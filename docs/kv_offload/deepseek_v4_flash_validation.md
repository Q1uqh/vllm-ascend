# DeepSeek-V4-Flash Mooncake offload 开发与整体验收

本实现对应方案 v1.8。只将 Attention compressor KV（随上下文增长的压缩历史缓存）放入跨层轮转槽。
SWA（近期滑窗缓存）、两个 compressor state（主注意力和索引压缩器各自的续算状态）和 Indexer K/scale（索引历史及缩放信息）保留本层内容。

## 实现与运行配置

模型缓存布局入口为 `vllm_ascend/models/deepseek_v4/cache_config.py`。
实际压缩历史注册名为 `.self_attn`；只按组件选择共享，不能把整层辅助缓存一起折叠。
普通布局继续使用既有跨组底层存储；开启共享时，将共享槽和驻留组件拆成独立分配，避免驻留数据与共享覆盖范围重叠。
容量计算、MRV2（第二版模型执行器）的实际分配和搬运地址均消费同一份描述。
每份分配额外预留 2 MiB（满足现有地址对齐分配器需要的空间），不会按逻辑别名重复收费。

配对 vLLM 提交：`24420b7a341af37a9bc3186e6fc6062512e0aa58`。
在该版本上，调度端通过 `kv_connector_block_state.get_block_ids()`（读取本轮真实有效分组块表的接口）替换请求跟踪表。
即使本轮没有分配新块也同步，保留滑窗回收后的空块位置。接口缺失或表缺失会报错。

首轮采用 eager（直接执行算子）、同步调度、关闭推测解码；PP（流水线并行）、PCP（预填充上下文并行）、DCP（解码上下文并行）均为1。
DP（独立数据并行副本）和 TP（张量并行）沿用现有服务配置，目标为 DP4/TP1＋EP（专家并行）。
必须启用 MRV2，并配置 `--enforce-eager --max-num-batched-tokens 4096`。
沿用原有模型、量化和 Mooncake 连接配置，Store（缓存池连接器）配置示例：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 2,
    "layerwise_independent_layers": [0, 1],
    "layerwise_prefetch_layers": 2
  }
}
```

在原有 `--additional-config` 中合并 `"kv_pool_range_debug": true`，打开诊断。
分别以 B（每个兼容域的共享槽数）=1/2/3/4 启动并保留完整服务日志。
全驻留对照可使用相同配置并将 `layerwise_independent_layers` 设置为 `"all"`。
这里的对照仍保留外部池能力，可另用未开启共享的原服务核对普通布局。

## 六项整体验收

| 项目 | 操作和证据 | 判定边界 |
| --- | --- | --- |
| 共享和驻留布局 | B=1/2/3/4；核对 `config`、`allocation`、`component` 的全部地址、组号、页大小、步长、前驱 | 仅压缩历史同域轮转；辅助组件和 L0/L1 不误共享 |
| 物理预算一致 | 汇总唯一 `allocation.address` 的 `bytes`，与 `config.planned_bytes` 比较；另核对对齐预留 | 相同底层分配只计一次，驻留组件未按 B 折叠 |
| 跨 chunk（一次调度处理的一段 token）续算 | 4096 分段，4096/4097/8193/12289 等输入；运行下面的确定性输出对照 | 读取准确 E（上一段计算结束的原始 token 位置）；同请求辅助数据不轮转换入，输出一致 |
| 外部前缀和并发 | 冷请求、重复、不同后缀和混合长度并发；核对 `restore` 来源及已有 commit（对象发布可读）日志 | 不能仅凭主压缩历史可读就认为完整命中；缺少必需组必须缩短命中或失败 |
| 槽位覆盖保护 | 新池或新输入下开启一次 PUT（从设备保存到池）延迟；核对 `source_done → slot_release → range_write_begin/compute_release` | 必须观察到真实 `slot_wait.blocked=true`；同进程使用单调时间比较 |
| 生命周期和错误 | 正常完成、客户端取消、GET（从池恢复）失败、PUT 失败、commit 失败及小容量池 | 必需数据失败后不能继续使用不完整缓存；核对撤销、读会话释放和 `terminal_cleanup` |

同请求续算时，混合组中的索引数据依旧可发布给外部读者，但恢复搬运只选择共享压缩历史。
外部前缀恢复保留全部必需组件任务，并复用基线协调器的端点与可达范围检查。
公共对象身份包含实际组件规格、格式、模型配置和并行分片，不添加 DP 专用隔离。

### 确定性输出与并发对照

在服务器仓库根目录运行；两个服务使用相同权重、分词器和采样配置。
脚本直接使用整数 token 输入，确保输入长度准确。并发批次包含重复输入及混合长度。
对照结果只证明已运行用例的文本一致，不代替异常路径验证。

```bash
python tests/e2e/cases/features/kv_pool/tools/compare_dsv4_offload.py \
  --baseline http://BASELINE_HOST:PORT \
  --candidate http://OFFLOAD_HOST:PORT \
  --model dsv4 --concurrency 4 > comparison.log
```

### 延迟与失败注入

以下参数放在 `kv_connector_extra_config` 中，仅用于服务器验收。
要求同时打开 `kv_pool_range_debug`；默认关闭。
每个后端实例只在第一次非空操作时注入一次，事件会明确标记为故障注入。

| 参数 | 值 | 用途 |
| --- | --- | --- |
| `offload_debug_put_delay_ms` | 0～5000 的整数，建议500 | 第一次实际 PUT 延迟，验证后继层等待 |
| `offload_debug_fail_once` | `"get"`、`"put"` 或 `"commit"` | 对应操作返回失败，验证错误处理；默认空字符串 |

各故障场景单独重启服务。GET 故障需要先保存再触发后续恢复。
延迟场景应使用未发布的新输入或新池，否则已存在对象可能跳过 PUT，不能得到有效延迟证据。
池容量不足通过服务器 Mooncake 容量配置复现，不用注入失败冒充容量不足。
取消可在长请求运行时中断客户端，再发送正常请求检查资源回收与后续可用性。

### 回传不超过300行

```bash
python tests/e2e/cases/features/kv_pool/tools/summarize_dsv4_offload.py \
  b1.log b2.log b3.log b4.log comparison.log delay.log get-failure.log put-failure.log \
  > dsv4-summary.txt
wc -l dsv4-summary.txt
```

汇总脚本逐行读取，输出硬上限280行，优先保留首批错误、配置、传输计数和六项验收摘要。
服务器完整日志保留；只需回传汇总文件。输出中的 `INSUFFICIENT` 表示证据不足，不等于通过。
故障注入运行中出现的预期错误需要结合场景核对，不能按正常成功场景解释。
不同文件按独立时钟范围处理，建议同一进程的完整日志不要拆成多个文件后再做顺序判定。

## 本机验证范围

```bash
python -m pytest --noconftest -q tests/host/test_dsv4_mooncake_offload.py
bash format.sh ci
```

主机测试直接执行生产代码中的规划、地址构建和范围选择函数，隔离框架导入，包含真实主机字节复制检查。
这不证明 NPU 算子、跨流同步、服务启动或多卡输出正确；这些结论由服务器六项整体验收提供。
现有平台布局补丁只增加对模型布局入口的调用。后续上游支持相同模型布局入口时可移除该版本桥接，模型侧规格仍保留。
