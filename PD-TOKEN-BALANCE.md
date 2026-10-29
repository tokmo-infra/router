# P/D token 负载路由

环境变量 VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG 指向 JSON 配置。配置包含 tokenizer_path、kv_fraction（本轮 0.85）、workers。workers 以实例基础 URL 为键，填写实际 server_info 的每路 kv_tokens 和 max_seqs。

- P：按未完成请求的估算计算量分流；公式 q + q*(c+q/2)/131072，q 为估算新增输入，c 为估算可复用前缀。
- 所有 P 共享成功响应的前缀观察，按模型及 cache_salt 隔离，有效期 60 秒。
- D：同时比较请求数与估算 KV 总长度，按各路实际 GPU KV 容量归一化；长请求更偏向 KV 余量大的卡。
- D 在选路时预占，流式结束、取消或连接释放时归还；P 的计算需求在 P 阶段完成时归还。
- 使用模型 tokenizer 对请求内容估算长度，额外计入输出预算和 10% 余量。估算用于选路，性能报告仍使用原生 usage。
- 分词最多 4 路并行，关闭额外 BPE 缓存，不重复序列化大请求文本。
- 这是 Router 层启发式估算，不改变 vLLM 的 chunk 调度，不包含专家副本 EPLB 或离线拟合时间模型。

指标：vllm_router_pd_token_balance_selected_total、vllm_router_pd_assigned_requests、vllm_router_pd_assigned_kv_tokens_estimate。
