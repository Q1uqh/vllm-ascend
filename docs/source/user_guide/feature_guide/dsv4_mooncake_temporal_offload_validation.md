# DeepSeek V4 Flash Mooncake KV Offload 服务器测试说明

本次 D2D 实现参考 [PR #20940](https://github.com/vllm-project/vllm-ascend/pull/20940) 的 `49c3bad3ad5c79a78c86828042d03f677b99cdea`，并增加 temporal 共享槽、延迟前缀发送和逐层 WRITE 完成屏障。配置文件可直接使用 `examples/disaggregated_prefill_v1/dsv4_temporal_d2d_p.json` 和 `dsv4_temporal_d2d_d.json`。

在 eager 模式的 Model Runner V2 中，Mooncake 的预填充节点（P）可以让兼容的层轮流使用同一设备缓存槽位。解码节点（D）先分配驻留缓存，通过代理发布目标地址；P 在最终 prefill chunk 的每层计算结束后，将该层有效 KV 直接 D2D WRITE 到 D 的 HBM。继续处理下一分块时，P 从公共键恢复已写完的历史页，并从请求专属快照恢复仍在使用的 SWA 缓存和压缩器状态。

## 服务器配置

以已验证可运行的 Ascend A5 DeepSeek V4 Flash 部署为起点。P 和 D 使用相同的模型路径、模型版本、KV 缓存数据类型和 TP 大小。启用 eager MRV2（`VLLM_USE_V2_MODEL_RUNNER=1` 和 `--enforce-eager`），P、D 使用相同的 DP、TP 和 EP 配置，PP、PCP、DCP 都设为 1。

权重需要多卡时，首轮沿用已验证可加载权重的 `DP=4、TP=1 + EP`。P 和 D 各自设置 `--data-parallel-size 4 --data-parallel-size-local 4 --tensor-parallel-size 1 --enable-expert-parallel`。P 使用设备 `0,1,2,3`，D 使用设备 `4,5,6,7`。

DP 模式下，公共历史按 P 副本隔离；PD 目标由 D 的引擎及 worker 元数据确定，不要求 P、D 副本编号相同。

vLLM 使用 `.github/vllm-main-verified.commit` 指定的提交。Mooncake 使用支持会话与范围传输 API 以及 `batch_remove` 的 DRAM 版本。记录实际使用的版本和每个 rank 的 KV 缓存预算。

P 使用 `MultiConnector`，按顺序配置 `MooncakePushConnector` 和 `AscendStoreConnector`。前者负责 D2D，后者负责 P 的分块 offload。D 仅使用 `MooncakePushConnector`，不读取池中快照。首轮部署中两端均关闭推测解码，并设置 `--no-async-scheduling` 和 `--no-enable-prefix-caching`。关闭本地前缀缓存后，仍可查询 Mooncake 中的公共前缀。

### P 端配置

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakePushConnector",
        "kv_role": "kv_producer",
        "kv_port": 14579,
        "kv_connector_extra_config": {"dsv4_temporal_d2d": true}
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "backend": "mooncake",
          "use_layerwise": true,
          "layerwise_num_shared_buffers": 2,
          "layerwise_independent_layers": [0, 1],
          "layerwise_prefetch_layers": 2,
          "layerwise_dsv4_pd_transport": "d2d"
        }
      }
    ]
  }
}
```

Push 必须排在 Store 前面。每层 hook 先等待 D 元数据，并完成该层全部 D2D WRITE，再由 Store 释放共享槽。当前优先保证时序正确，该等待会限制传输与计算重叠；性能需要服务器实测。

### D 端配置

```json
{
  "kv_connector": "MooncakePushConnector",
  "kv_role": "kv_consumer",
  "kv_port": 15579,
  "kv_connector_extra_config": {"dsv4_temporal_d2d": true}
}
```

两端必须等 TP，PP=PCP=DCP=1，关闭推测解码、本地 prefix cache；P 关闭异步调度。D 保持各层 HBM 驻留。删除旧的 `layerwise_dsv4_handoff` 配置，不使用 snapshot handoff 代理。

### 单机 8 卡启动

保留现有服务器脚本的网卡、HCCL、CANN 和 Mooncake 环境变量。两端的 `MOONCAKE_CONFIG_PATH` 指向各自可用的配置，连接同一个 Mooncake master，配置 `enable_ssd_offload=false`。将上面两个 JSON 分别保存为 `kv-p.json` 和 `kv-d.json`。在两个终端加载同一套环境变量，再各自执行下面对应的命令。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONHASHSEED=0
local_ip="141.61.133.129"
model="/mnt/share/weight/DeepSeek-V4-Flash-0731"

common=(
  --host "$local_ip" --served-model-name dsv4
  --data-parallel-size 4 --data-parallel-size-local 4
  --tensor-parallel-size 1 --enable-expert-parallel
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching
  --no-disable-hybrid-kv-cache-manager --trust-remote-code
  --max-model-len 1048576 --block-size 32 --kv-cache-dtype fp8
  --attention_config.indexer_kv_dtype fp8 --tokenizer-mode deepseek_v4
  --additional-config '{"enable_cpu_binding":true,"kv_pool_range_debug":true}'
)
```

P 端：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve "$model" "${common[@]}" \
  --port 8099 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 --kv-transfer-config "$(cat kv-p.json)" \
  2>&1 | tee v4_flash_p.log
```

D 端：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve "$model" "${common[@]}" \
  --port 8066 --max-num-batched-tokens 1024 --max-num-seqs 56 \
  --gpu-memory-utilization 0.95 --kv-transfer-config "$(cat kv-d.json)" \
  2>&1 | tee v4_flash_d.log
```

从旧脚本切换时，替换两端整段 `--kv-transfer-config`，去掉 `--speculative-config`、D 端 `FULL_DECODE_ONLY` 和 `enable_npugraph_ex`。P 使用 Mooncake DRAM 存储支持中间 chunk 换出、换入；PD 的数据传输使用设备 TransferEngine 的 `batch_transfer_sync_write`，不经过 D 的池子恢复路径。

状态模型将最后一个原始提示词 token 留给 D 重算：P 处理前 N-1 个 token，并逐层传输 N-1 端点的压缩 KV、SWA、compressor 和 indexer 状态。D 收齐所有 P rank 的 DONE 后，处理原始第 N 个 token 并正常生成，不拼接 P 采样 token。

为 P 的活跃快照和历史预留足够 Mooncake DRAM；中间 chunk 必需对象缺失时明确报错。P 的私有快照在请求释放后回收，不等待 D 从池子读取。

### 通过代理发请求

在代理所在服务器使用本分支的 `load_balance_proxy_layerwise_server_example.py`。假设代理在 `141.61.133.128`，P、D 都在 `141.61.133.129`，按下面方式启动。 `--host` 是代理所在机器的监听地址。

```bash
python3 examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 141.61.133.128 --port 1999 \
  --prefiller-hosts 141.61.133.129 --prefiller-ports 8099 \
  --decoder-hosts 141.61.133.129 --decoder-ports 8066
```

```bash
curl -sS http://141.61.133.128:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4",
    "messages": [{"role": "user", "content": "请用一句话解释什么是 KV 缓存。"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }'
```

代理先请求 D；D 分配缓存后通过 `/v1/metaserver` 通知代理启动 P。用 `DSV4_D2D` 的 `write_complete`、`slot_released` 和 `receive_complete` 确认交接，D 不应出现 `handoff_bulk_h2d`。首次使用单条文本请求；参考 PR 的 metaserver 每个 D 引擎最多 32 个并行 POST，大型多 prompt batch 尚有限制。

## 与驻留模式对照

- **容量对照**：在相同 KV 缓存预算下重跑 P，省略 `layerwise_num_shared_buffers`，得到驻留模式的块数和显存基线。
- **数值对照**：使用模型和运行配置相同、已验证可运行的驻留生成服务。冷启动用例使用新的 Mooncake 命名空间或存储池。
- **分块用例**：将 `--max-num-batched-tokens` 设为小于至少一条提示词的 token 数，以触发多分块预填充。将测试输入写入 `prompts.json`，格式为 JSON 数组，每个元素是一条提示词的 token ID 数组。

```bash
python3 tests/e2e/cases/features/kv_pool/dsv4_mooncake_temporal_smoke.py \
  --proxy http://141.61.133.128:1999 \
  --model dsv4 \
  --prompt-tokens prompts.json --output-tokens 8 --repeat 2 | tee smoke.log
```

需要对照输出时，加上 `--resident http://resident:8000`。首次先用一条短请求检查 P、D 交接，再增加跨 chunk 输入。

## 验收项目

当前待验证项限定为以下四项。固定相同模型、KV dtype 和 TP，使用 B=2、L0/L1 独立、eager MRV2，PP=PCP=DCP=1；关闭推测解码和本地 prefix cache，P 关闭异步调度。精度对照作为跨 chunk 恢复的结果证据，不单独扩展为另一项验收。

| 待验证项 | 验证什么 | 如何验证 | 通过条件 |
| --- | --- | --- | --- |
| P 端共享槽轮转 | 不同逻辑层确实复用同一块 HBM，且执行顺序符合规划 | 检查 `DSV4 temporal slots` 的 predecessor、`layer_layout` 中的 group/base_addresses/stride，以及 `DSV4_SLOT_REUSE` 的实际轮转边；核对 L2→L6 等关系和 L0/L1 独立位置 | 同组兼容层共用地址，实际轮转边与规划一致，独立层未混用 |
| 跨 chunk 恢复 | 下一 chunk 恢复的是本请求、本层在上一端点的有效历史和状态 | P 设置 `--max-num-batched-tokens 4096`，发送 8193-token 原始输入；检查 E=4096 的 `snapshot_committed`、下一轮 `snapshot_prepare.restore_end`，以及各组 `layer_restore_begin`→`layer_restore_complete`→`layer_entry_ready`；重复并增加部分相同前缀请求，用驻留服务对照 token IDs | 请求身份与恢复端点一致，各层必需 GET 成功后才放行计算，输出与对照一致；重复请求须有真实池命中证据 |
| PD 确实走 D2D | 最终 chunk 的累计有效 KV 和状态从 P HBM 直接写入 D HBM | 检查 `write_complete` 的 requests/endpoints、源/目标地址样本、复制字节数及 D 的 `receive_complete`；结合设备 profiling 检查实际搬运类型及设备端点 | 最终端点 E=N-1 正确，各发送 rank 成功完成后 D 才运行；真实设备间传输可确认，D 的 `handoff_bulk_h2d` 为零；无 H2D 日志本身不能证明 D2D |
| 共享槽不会提前覆盖 | 后继 GET 或设备计算不会覆盖仍被前驱计算、PUT 或 D2D 使用的源位置 | 按同一 P Worker 的 pid、monotonic_ns、请求端点和层号关联 `write_complete`、`slot_released`、`layer_save_complete`、`overwrite_fence`、后继 `layer_restore_begin` 和 `layer_entry_ready`；覆盖冷请求、多 chunk，并单独延迟 PUT 或 WRITE | 前驱设备访问、所需 PUT 源复制及 D2D WRITE 完成后，后继才能首次写入共享位置；有延迟时实际等待，无提前覆盖 |

`slot_released` 的 scope 为 `d2d_source_reads_only`，只证明该层 D2D 源读取已经结束，不代表 Store PUT 或所有 reader 均已完成。`layer_save_complete` 位于正常保存路径成功返回后，失败唤醒不会记录成功事件；`layer_entry_ready` 是 Worker 放行 attention 的主机侧事件，不是 NPU kernel 起始时间。device event 同步和同步 WRITE 返回提供完成边界，真实设备时序仍以 profiling 补强。

使用四组用例：单 chunk 冷请求、8193-token 跨 chunk 冷请求、重复/部分相同前缀请求、延迟 PUT/WRITE 的请求。延迟注入不是冒烟脚本自动执行的功能，需要独立的测试注入或服务器调试手段。正常请求日志不能替代受控延迟用例。

池子范围打点通过 `--additional-config '{"kv_pool_range_debug":true}'` 开启（与现有 additional-config 字段合并）。D2D 布局与完成日志直接记录。统一事件包含 pid 和 monotonic_ns；只在同一主机时钟域内比较时间，不拿不同主机的 monotonic_ns 直接排序。启动前保留配置日志，再运行请求并收集 P、D 与 smoke 日志。

```bash
python3 tests/e2e/cases/features/kv_pool/dsv4_mooncake_temporal_smoke.py \
  --proxy http://代理IP:1999 --resident http://驻留对照IP:端口 \
  --model dsv4 --prompt-tokens prompts.json --output-tokens 32 --repeat 2 \
  2>&1 | tee smoke.log

python3 tests/e2e/cases/features/kv_pool/summarize_mooncake_range_logs.py \
  v4_flash_p.log v4_flash_d.log smoke.log > mooncake-range-summary.txt
```

`prompts.json` 为真实 tokenizer 产生的 token ID 数组的 JSON 列表。不传 `--resident` 时，冒烟脚本只检查长度和有限 logprobs，不能判定精度正确。摘要输出不超过 300 行，保留事件计数、共享地址样本、近期时序样本与错误。摘要是有限样本，不能自动证明所有请求、所有层时序安全；按四项分别记录“通过／失败／证据不足”，缺少完整关联链或设备 profiling 时保持证据不足，不凭事件计数宣称全部通过。

## 本机验证与证据边界

Python 3.11+ 安装 pytest、msgspec、pyzmq、httpx 后，可执行：

```bash
python3 tests/ut/kv_offload/mooncake_v2/run_host_tests.py
```

当前隔离 CPU 回归为 86 passed、27 skipped；跳过项需要本地 socket 监听，当前沙箱禁止绑定端口。隔离运行器保留实际 scheduler/worker 方法，以模拟对象替代 framework/NPU 初始化，并用 CPU 字节复制验证页跨度、窗口和地址映射；这不等同于完整依赖导入或真实 NPU 测试。修改文件通过 Ruff 和语法检查，日志摘要验证了字节计数和 300 行上限。真实 D2D 负载、输出正确性、DP/TP 多 rank 时序和性能由服务器验收。
