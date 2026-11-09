# DeepSeek V4 Flash Mooncake 共享缓存适配

本实现以 PR（代码合并请求）#21301 的 `e1c7eafab730ec87ffcd8069cf08d81a2b843230` 为基线，
工作分支为 `feat/dsv4-mooncake-layerwise-offload`。
配对 vLLM 源码为 `84030bbe3d74d99bad477a3d2e37a973ccd8865c`，需要其
`KVConnectorBlockState`（调度器向连接器提供当前有效块表的接口）。不支持此接口时明确报错，不能沿用已释放的滑窗块编号。

## 实现与边界

- 使用已有 Store（存储池连接器）的调度、读写会话、`offload_v1` 私有快照和错误处理。
- 模型声明完整组件组合，公共 planner（根据规格生成实际存储清单的规划器）同时规划所有组件。
  L0/L1 独立；CSA（压缩稀疏注意力）和 HCA（高压缩注意力）按真实规格分别轮转。
  多个缓存组继续 overlay（通过全局块编号排他分配，让不同组利用同一底层区域）。
- 容量预算、跨进程块数归一化和实际分配使用同一物理计划；整个共享 backing（张量实际占用的底层存储）只计一次。
- 中间 chunk（一次调度处理的一段输入）只存池；最后一个 chunk 经已有 PD（预填充与解码分离）连接器发送累计有效历史和当前状态。
  原始输入 N 个 token（模型处理的基本符号单位）时，P（预填充节点）计算 N-1 个，D（解码节点）计算最后一个原始 token。
  单 token 输入由 D 本地处理，P 不发送缓存，不构造空输入。
- 沿用已有异步 WRITE（把数据写入远端设备内存）及槽释放接口；Store PUT（把设备数据保存到存储池）和 PD 都读完后才能覆盖。
  多组组件在 post-attention（当前层注意力计算完成后）的共同设备事件后发送。
- 压缩器状态保存准确端点的活动窗口；对齐到完整页的端点也有私有快照。
  私有状态快照不作为跨请求公共命中。重复请求可能重新计算，不能据此推断存储故障。
- 首轮限定 eager（不捕获整图、直接执行算子）和 MRV2（第二版模型执行器），等 TP（张量并行度），
  PP（流水线并行度）=PCP（预填充上下文并行度）=DCP（解码上下文并行度）=1；关闭推测解码。
  首先使用 TP=1、DP（数据并行度）=1，再验证 DP=4、TP=1。

已有 patch（对上游接口的兼容修改）仍负责旧版本的预算入口；仅将 DSV4 分支接到公共物理计划。
没有新增模型执行器或另一套 offload（将暂时不用的设备缓存换出、需要时恢复）后端。
后续上游版本统一采用可插拔容量构建入口时，可移除旧版 patch 中的预算适配。

## 配置

沿用服务器现有模型路径、量化参数、Mooncake master/client（存储服务主节点及提供存储空间的进程）配置。
Mooncake 客户端必须支持 PR #21301 要求的批量 session（受管理的读写会话）与 range（对象内字节范围）接口。
`MOONCAKE_CONFIG_PATH` 指向现有 JSON 配置文件。

P 的 `--kv-transfer-config`：PD 子连接器排在前面，确保先确定输入截断端点，再查询 Store。

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "MooncakeLayerwiseConnector", "kv_role": "kv_producer"},
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "backend": "mooncake",
          "use_layerwise": true,
          "load_async": false,
          "layerwise_num_shared_buffers": 2,
          "layerwise_independent_layers": [0, 1],
          "layerwise_prefetch_layers": 2
        }
      }
    ]
  }
}
```

D 的 `--kv-transfer-config`：

```json
{"kv_connector": "MooncakeLayerwiseConnector", "kv_role": "kv_consumer"}
```

两端使用相同模型、量化格式与 TP；不同实例设置原有 `kv_port`，避免控制端口冲突。
P 设置 `--no-enable-prefix-caching --max-num-batched-tokens 4096 --enforce-eager`，关闭异步调度和推测解码。
两端在原有 `--additional-config` 中合并 `"kv_pool_range_debug": true`，不要覆盖其他模型所需参数。

启动已有代理，替换地址与端口：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 141.61.133.128 --port 1999 \
  --prefiller-hosts 141.61.133.128 --prefiller-ports 8099 \
  --decoder-hosts 141.61.133.128 --decoder-ports 8066
```

## 验证

本机检查：

```bash
python tests/ut/kv_offload/test_dsv4_shared_components_host.py
```

该测试隔离框架导入，直接执行生产源码的规划、快照选择和发送完成保护；包含真实 CPU（中央处理器）字节复制。
它不能证明框架初始化、NPU（神经网络处理器）算子或设备搬运正确。
完整框架回归应在已安装配对 vLLM、torch 和依赖的环境运行：

```bash
pytest tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py \
  tests/ut/distributed/ascend_store/test_mooncake_offload.py
```

服务器验收保持四项：

| 项目 | 怎么检查 |
| --- | --- |
| P 端共享槽轮转 | `layout`/`pd_layout` 中逐组件地址、步长与计划相符；B=2；CSA 的 L2→L6、HCA 的 L3→L7；L0/L1 不进入同组轮转 |
| 跨 chunk 恢复 | 8193 个原始输入 token、chunk=4096；检查 `snapshot_plan` 的恢复端点及活动页；覆盖非对齐端点、重复前缀；与驻留服务比较确定性输出 |
| PD 确实走 D2D（设备内存直接到设备内存） | 中间 chunk 不出现请求的 `pd_write`，最终发送累计有效上下文；检查所有发送进程完成；附设备 profiling（实际算子与搬运时序采集），排除 D 从存储池 H2D（主机内存到设备内存）接收 |
| 共享槽不会提前覆盖 | 检查前驱 Store 完成、PD `pd_reader_done` 与后继 `reuse_ready`；分别延迟 PUT/WRITE，实际观察等待；不能只看正常请求的事件计数 |

沿用输出对照脚本，对照服务与候选服务使用同一模型名称：

```bash
python tests/e2e/cases/features/kv_pool/verify_mooncake_offload.py \
  --baseline-url http://BASELINE:PORT/v1 --candidate-url http://141.61.133.128:1999/v1 \
  --model dsv4 --max-tokens 16 --concurrency 4
```

该脚本按文本构造请求，不保证恰好 8193 token；精确边界另用 tokenizer（把文本转换成模型符号编号的工具）生成 token ID 列表后调用 `/v1/completions`。

日志汇总：

```bash
python tests/e2e/cases/features/kv_pool/tools/summarize_component_offload.py \
  v4_flash_p.log v4_flash_d.log > component-offload-summary.txt
```

汇总最多 280 行，优先保留错误。把对照结果与设备搬运证据控制在额外 20 行内，总计不超过 300 行。
日志时间为同一进程的单调时钟；不同机器不能直接比较。
汇总只报告观察证据或失败，不凭采样日志自动宣布四项验收通过。
