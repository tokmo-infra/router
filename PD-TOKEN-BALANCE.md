# P/D token 负载路由

环境变量 VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG 指向 JSON 配置。配置包含 tokenizer_path、kv_fraction（本轮 0.85）、workers。workers 以实例基础 URL 为键，填写实际 server_info 的每路 kv_tokens 和 max_seqs。

- P：按未完成请求的估算计算量分流；公式 q + q*(c+q/2)/131072，q 为估算新增输入，c 为估算可复用前缀。
- 所有 P 共享成功响应的前缀观察，按模型及 cache_salt 隔离，有效期 60 秒。
- D：同时比较请求数与估算 KV 总长度，按各路实际 GPU KV 容量归一化；长请求更偏向 KV 余量大的卡。
- D 在选路时预占，流式结束、取消或连接释放时归还；P 的计算需求在 P 阶段完成时归还。
- 使用模型 tokenizer 对请求内容估算长度，额外计入输出预算和 10% 余量。估算用于选路，性能报告仍使用原生 usage。
- 分词最多 4 路并行，关闭额外 BPE 缓存，不重复序列化大请求文本。
- 可选 `prefill_kv_guard: true`：P 同时预占估算 KV，并参考每路真实 GPU KV 使用率；每秒刷新，超过 5 秒的观测不参与选路。默认关闭。
- P 返回非成功 HTTP 状态时，保留上游错误原因，不再误报为缺少 usage。
- 这是 Router 层启发式估算，不改变 vLLM 的 chunk 调度，不包含专家副本 EPLB 或离线拟合时间模型。

指标：vllm_router_pd_token_balance_selected_total、vllm_router_pd_assigned_requests、vllm_router_pd_assigned_kv_tokens_estimate、vllm_router_pd_live_kv_usage。

`vllm_router_pd_estimation_seconds{phase=...}` 拆分分词槽等待（slot_wait）、阻塞线程池等待（blocking_wait）、分词及前缀估算（compute）和估算全程（total）。这些是 Router 墙钟耗时，不是 GPU 计算或首 token 延迟；不改变选路公式。

## P KV 压力试验

在原 JSON 中加入 `"prefill_kv_guard": true` 即可启用；删除该项恢复原计算量选路。它调整选路优先级，不是硬性 KV 准入限流。

- CPU 测试：717 项通过，另 12 项 WASM 示例测试未运行。
- 模型仍用原 C230 镜像与 FP8/MTP3；不改 Mooncake 的读租约或缓存格式。
- 基线终结后更换 Router、独立 DB 对照，记录 P/D 抢占、失败及 55% 后完整分钟性能。

## API 分词缓存亲和代理

外层仍负责GPU/DP选路；新增代理只选择同机API进程。共享分词池属于配套vLLM，代理不重新分词，也不注册额外GPU rank。

~~~bash
docker build -f Dockerfile.glm53flash-cache-repair -t local/glm53flash-router:0.1.16 .
docker run --rm --network host --entrypoint python3 local/glm53flash-router:0.1.16 /opt/glm-cache-repair/glm_api_cache_router.py --help
~~~

P使用32110→31010，D主节点使用32111→31011；传入各API的--api-socket /run/glm-tokenizer/api-N.sock和--shared-cache-path /run/glm-tokenizer/segments.sqlite3。前缀阈值0.3、活跃请求差阈值8/比例1.5，前缀观察TTL1800秒。
