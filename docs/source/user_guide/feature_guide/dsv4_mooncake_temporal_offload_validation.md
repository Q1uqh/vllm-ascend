# DeepSeek V4 Flash Mooncake KV Offload 服务器测试说明

在 eager 模式的 Model Runner V2 中，Mooncake 的预填充节点（P）可以让兼容的层轮流使用同一设备缓存槽位。解码节点（D）使用驻留缓存，从 P 的最终检查点恢复。继续处理下一分块时，P 从公共键恢复已写完的历史页，并从请求专属快照恢复仍在使用的 SWA 缓存和压缩器状态。

## 服务器配置

以已验证可运行的 Ascend A5 DeepSeek V4 Flash 部署为起点。P 和 D 使用相同的模型路径、模型版本、KV 缓存数据类型和 TP 大小。启用 eager MRV2（`VLLM_USE_V2_MODEL_RUNNER=1` 和 `--enforce-eager`），P、D 使用相同的 DP、TP 和 EP 配置，PP、PCP、DCP 都设为 1。

权重需要多卡时，首轮沿用已验证可加载权重的 `DP=4、TP=1 + EP`。P 和 D 各自设置 `--data-parallel-size 4 --data-parallel-size-local 4 --tensor-parallel-size 1 --enable-expert-parallel`。P 使用设备 `0,1,2,3`，D 使用设备 `4,5,6,7`。

DP 模式下，公共历史按 P 副本隔离。D 按交接元数据中的 P 副本编号读取，不要求 P、D 的副本编号相同。

vLLM 使用 `.github/vllm-main-verified.commit` 指定的提交。Mooncake 使用支持会话与范围传输 API 以及 `batch_remove` 的 DRAM 版本。记录实际使用的版本和每个 rank 的 KV 缓存预算。

单独使用 `AscendStoreConnector`。P 端关闭推测解码，并设置 `--no-async-scheduling` 和 `--no-enable-prefix-caching`。关闭本地前缀缓存后，仍可查询 Mooncake 中的公共前缀。

### P 端配置

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

### D 端配置

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true,
    "layerwise_dsv4_handoff": true
  }
}
```

向 P 发送请求时设置 `max_tokens=1`。将 P 返回的 `kv_transfer_params`、原始提示词 token ID 序列以及 P 采样出的第一个 token 一并传给 D。D 从 P 实际完成预填充的位置开始解码，并在读完后按后端租约规则删除私有对象，完成交接确认。

为活跃快照和历史数据预留足够的 Mooncake DRAM。GET 会话保护正在读取的对象，两次读取之间的数据保留依赖可用容量。必需对象在分块之间或交接前被驱逐时，传输会报错。如果交接在 D 确认前被放弃，未确认的快照仍可由 Mooncake 驱逐。

## 与驻留模式对照

- **容量对照**：在相同 KV 缓存预算下重跑 P，省略 `layerwise_num_shared_buffers`，得到驻留模式的块数和显存基线。
- **数值对照**：使用模型和运行配置相同、已验证可运行的驻留生成服务。冷启动用例使用新的 Mooncake 命名空间或存储池。
- **分块用例**：将 `--max-num-batched-tokens` 设为小于至少一条提示词的 token 数，以触发多分块预填充。将测试输入写入 `prompts.json`，格式为 JSON 数组，每个元素是一条提示词的 token ID 数组。

```bash
python3 tests/e2e/cases/features/kv_pool/dsv4_mooncake_temporal_smoke.py \
  --prefill http://prefill:8000 --decode http://decode:8000 \
  --resident http://resident:8000 --model dsv4 \
  --prompt-tokens prompts.json --output-tokens 8 --repeat 2 | tee smoke.log
```

## 验收项目

- **输出正确性**：对比冷请求、部分命中和热请求的 token ID，并检查 logprobs 是否为有限值。覆盖交错请求、32K 和 128K 输入，以及 3/4/5、127/128/129 和各缓存组实际块边界附近的分块端点。
- **容量和布局**：对比每个 rank 的块数、去重后的底层存储字节数和 HBM 峰值。测试共享缓冲区数量为 1 到 4 的配置及独立驻留层，核对规划器使用各组实际页格式和逻辑块表。
- **异步时序**：分别延迟 PUT、GET 和 commit。确认无需 GET 的后继层及不负责上传的 rank 也会等待前一层设备访问和源数据复制完成。确认下一分块和 D 只读取已提交的数据。
- **P/D 交接**：测试对齐和非对齐的预填充端点，并延迟向 D 提交请求。确认最终私有对象保留到 D 读完，并在读取租约允许后删除。
- **失败和回收**：注入对象缺失、传输字节数不足、取消和抢占。确认遇到必需数据缺失时明确报错，且撤销只针对本端拥有的 PUT 会话。至少跑 1,000 个交错或多分块请求，或连续运行 1 小时。
- **性能**：预热后至少测 3 轮，记录首 token 延迟（TTFT）的 P50/P95、吞吐和 HBM 峰值。分别报告请求传输字节数和实际复制字节数。冒烟脚本会记录请求耗时。

## 提取回传日志

诊断时，在 Ascend additional config 中开启 `kv_pool_range_debug`。汇总 P 端日志、D 端日志和冒烟测试报告。脚本保留近期错误和失败传输事件，合计输出不超过 300 行。

```bash
python3 tests/e2e/cases/features/kv_pool/summarize_mooncake_range_logs.py \
  prefill.log decode.log smoke.log > mooncake-range-summary.txt
```
