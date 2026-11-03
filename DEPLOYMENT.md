# Router部署与配置

按P / D服务角色登记实例，不绑定并发数或固定P/D数量。模型、Mooncake及推理参数由对应推理部署管理，Router不修改引擎chunk调度、FP8、MTP或CUDA graph。

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

## 2. 按计算量与KV容量路由

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

## 3. 耦合实例逐DP路由

`97b00848`适用于耦合服务，不用于替代上述PD容量路由：

```bash
export VLLM_ROUTER_COUPLED_KV_BALANCE=1
```

每秒读取各DP的KV、running和waiting，结合Router在途负载选路；只使用五秒内观测。空闲缓存可淘汰，不当作活跃请求压力。默认不启用。

## 4. D复用P分词结果

`08d71951`对`/v1/chat/completions`默认开启。P需支持`return_token_ids`，D需支持原生`OnlineRenderer._reused_prompt_token_ids`，两端模型、Tokenizer和模板设置必须一致。

Router验证P返回的ID，将其放入D的`kv_transfer_params.prompt_token_ids`，保留原有传输信息和输出选项。缺失或非法ID会在发送D前返回错误；后端继续执行词表等校验。echo、多模态或批量chat可能需要额外渲染，不能仅凭Router计数确认省掉了D端分词。

可显式关闭或开启：

```bash
export VLLM_ROUTER_PD_REUSE_PROMPT_IDS=0  # 关闭
# export VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1  # 显式开启；未设置时同样开启
```

**异步边界**：此路径需等P返回ID再发送D，因此会关闭该聊天请求在Router层的P/D并发派发；它不关闭vLLM引擎的异步调度。原本依赖MoRI-IO WRITE或NIXL push并发派发的部署需验证这一变化。不兼容的后端请显式关闭。其他API路径不受此开关影响。

## 5. 开关汇总

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG` | 未设置 | 指向容量路由JSON时启用P计算量 / D容量路由。 |
| JSON中的`kv_fraction` | `0.85` | 路由估算可用KV比例。 |
| JSON中的`prefill_kv_guard` | `false` | 可选P KV压力反馈。 |
| `VLLM_ROUTER_COUPLED_KV_BALANCE` | 未设置，关闭 | 精确设为`1`开启耦合实例逐DP路由。 |
| `VLLM_ROUTER_PD_REUSE_PROMPT_IDS` | 未设置，开启 | 聊天请求默认复用P的ID；显式`0`关闭，`1`开启。 |

其余负载生命周期、usage合并、错误透传和估算阶段计时没有新增独立开关。

## 6. 观测与验收

| 指标 / 接口 | 用途 |
| --- | --- |
| `/workers` | 核对服务发现、角色、DP rank及健康状态。 |
| `vllm_router_pd_assigned_requests` / `vllm_router_pd_assigned_kv_tokens_estimate` | 观察各路预占请求数和估算KV负载。 |
| `vllm_router_pd_live_kv_usage` | P KV压力反馈采样。 |
| `vllm_router_coupled_live_kv_usage` / `vllm_router_coupled_routing_score` | 耦合路由实时KV及评分。 |
| `vllm_router_pd_estimation_seconds{phase=...}` | 区分slot_wait、blocking_wait、compute、total；是CPU墙钟耗时，不是GPU时间。 |
| `vllm_router_pd_prompt_ids_reused_total` | 已准备的ID交接次数；D后续HTTP失败也可能包含在内。 |

先核对rank、流式usage、取消后的负载释放及真实后端ID复用，再做性能对照。Router新功能需要新二进制；整理Git历史不会自动更新已运行容器。
