# Router部署与配置 · GLM5.3 Flash 1103

按P / D服务角色登记实例，扩容时增加对应地址和bootstrap端口。Router负责路由与流式转发，推理服务负责chunk调度、FP8、MTP和CUDA graph。

## 1. 构建与启动

分支使用固定Cargo.lock；本地路径依赖需要相邻的vLLM Rust源码目录：

```bash
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git ../vllm-rust-v029
cargo build --locked --release --bin vllm-router
```

以下地址是展开示例。增加实例时追加对应的--prefill / --decode，填写实际bootstrap端口；DP由后端服务发现，不把实例地址数量当作DP数量。

```bash
export P_URL=http://127.0.0.1:8001
export D_URL=http://127.0.0.1:8002
export P_BOOTSTRAP_PORT=9001
read -rs -p 'API key: ' API_KEY; echo; export API_KEY
export OPENAI_API_KEY="$API_KEY"
export VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1
./target/release/vllm-router \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill "$P_URL" "$P_BOOTSTRAP_PORT" --decode "$D_URL" \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --balance-abs-threshold 4 --balance-rel-threshold 1.5 \
  --host 0.0.0.0 --port 8000 \
  --prometheus-host 0.0.0.0 --prometheus-port 8003 \
  --disable-retries --worker-startup-timeout-secs 600 \
  --worker-startup-check-interval 5 --api-key "$API_KEY"
```

`/workers`中应包含全部预期rank并处于健康状态，再接入业务。不要在已有服务端口上重复启动。`glm53flash-integration.json`保留历史构建凭据，不代表新源码已自动进入旧镜像。

## 2. 默认复用P的分词结果

长多轮请求在P完成分词后，D直接使用P返回的token ID，减少重复渲染模板和分词带来的CPU开销。此优化移植自`08d71951`，聊天接口默认开启，也可显式配置：

```bash
export VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1  # 开启；未设置时也开启
# export VLLM_ROUTER_PD_REUSE_PROMPT_IDS=0  # 关闭
```

Router向P请求`return_token_ids=true`，检查返回的`prompt_token_ids`后，连同原KV元数据放入D的`kv_transfer_params`。D保留客户端的输出预算和流式选项。P/D使用相同的模型、Tokenizer和chat template，vLLM后端需支持原生`OnlineRenderer._reused_prompt_token_ids`接口；本套1103引擎已包含该接口。

复用只作用于`/v1/chat/completions`。D等待P返回结果再派发，vLLM引擎内部仍保持异步调度。空或非法ID、缺少KV元数据会在派发D前返回错误。

## 3. 按计算量与KV容量路由

`f2f1bb6e`用估算负载分配请求，不替代推理引擎的实际调度。设置`VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG`指向JSON后启用，容量按每路实际DP填写。示例数值仅表示配置格式，应从后端server_info读取真实值。

```json
{
  "tokenizer_path": "/model/tokenizer.json",
  "kv_fraction": 0.85,
  "prefill_kv_guard": false,
  "workers": {
    "http://127.0.0.1:8001": {"kv_tokens": 100000, "max_seqs": 24},
    "http://127.0.0.1:8002": {"kv_tokens": 200000, "max_seqs": 16}
  }
}
```

```bash
export VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG=/path/to/token-balance.json
```

- P按未完成请求的估算计算量分流：`q + q*(c+q/2)/131072`，q为新增输入，c为估算可复用前缀。
- 成功P响应形成前缀观察，跨P共享，按模型及cache_salt隔离，有效期60秒。
- D同时比较请求数和容量归一化KV需求，包含输出预算和余量；在流结束或取消时释放预占。
- 分词最多四路并行，避免重复保留大请求BPE缓存。估算只用于选路，统计使用后端实际usage。
- 可选`prefill_kv_guard=true`参考P真实KV压力，每秒刷新，五秒以上旧观测不参与选路。它是选路优先级，不是硬性准入或GPU抢占修复。

## 4. 耦合实例逐DP路由

`97b00848`适用于耦合服务，不用于替代上述PD容量路由：

```bash
export VLLM_ROUTER_COUPLED_KV_BALANCE=1
```

每秒读取各DP的KV、running和waiting，结合Router在途负载选路；只使用五秒内观测。空闲缓存可淘汰，不当作活跃请求压力。默认不启用。

## 5. 开关汇总

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `VLLM_ROUTER_PD_REUSE_PROMPT_IDS` | 未设置时开启 | 聊天请求让D复用P的分词结果；精确`1`开启，`0`关闭。 |
| `VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG` | 未设置 | 指向容量路由JSON时启用P计算量 / D容量路由。 |
| JSON中的`kv_fraction` | `0.85` | 路由估算可用KV比例。 |
| JSON中的`prefill_kv_guard` | `false` | 可选P KV压力反馈。 |
| `VLLM_ROUTER_COUPLED_KV_BALANCE` | 未设置，关闭 | 精确设为`1`开启耦合实例逐DP路由。 |

其余负载生命周期、usage合并、错误透传和估算阶段计时没有新增独立开关。

## 6. 观测与验收

| 指标 / 接口 | 用途 |
| --- | --- |
| `/workers` | 核对服务发现、角色、DP rank及健康状态。 |
| `vllm_router_pd_prompt_ids_reused_total` | Router已准备的P→D分词结果交接次数；D实际复用和耗时看后端。 |
| `vllm_router_pd_assigned_requests` / `vllm_router_pd_assigned_kv_tokens_estimate` | 观察各路预占请求数和估算KV负载。 |
| `vllm_router_pd_live_kv_usage` | P KV压力反馈采样。 |
| `vllm_router_coupled_live_kv_usage` / `vllm_router_coupled_routing_score` | 耦合路由实时KV及评分。 |
| `vllm_router_pd_estimation_seconds{phase=...}` | 区分slot_wait、blocking_wait、compute、total；是CPU墙钟耗时，不是GPU时间。 |

先核对全部rank、流式usage和取消后的负载释放，再接入业务。4P4D使用16核/16GiB Router，登记8个EP4 P实例和4个EP8 D实例，`/workers`应出现64路健康rank；容量配置按每路P seq24、D seq32及后端实际KV容量生成。完整引擎、Mooncake与Tokenizer命令集中在vLLM同名1103分支的`deployment4p4d.md`。

当前配套镜像是`local/glm53flash-router:native-ids-20261103`，ID为`sha256:c07f71ef603b3c28c2f2b48e0657c4c96fcee285428dcd64747ed53a3d3aeec5`，ELF SHA256为`6c51e70abc6a668a716b4738f448b5c2034559d63c3fe2c18d73352a95f99448`。这是本地镜像标签，跨机器使用先同步镜像并核对ID；源码构建时记录新产物的指纹。

本机CPU假后端合同检查覆盖默认开启、显式开启和关闭：

```bash
uv venv --python 3.12
for mode in default on off; do
  .venv/bin/python scripts/check_pd_prompt_ids.py \
    --binary target/release/vllm-router --mode "$mode" \
    --output "/tmp/pd-prompt-ids-${mode}.json"
done
```
