# Router 0.1.16 · GLM-5.3 Flash

- **P/D 路由**：支持 Mooncake PD 分离及多 P、多 D 实例。
- **D 负载预占**：选择 D 时原子预占，计数持续至流式结束；取消和连接释放时归还。
- **P/D token 负载**：P 按估算计算量分流，D 同时均衡请求数与容量归一化的 KV 长度；配置见根目录 PD-TOKEN-BALANCE.md。
- **P KV 压力反馈**：可选结合真实 GPU KV 使用率和长请求 KV 预占估算选择 P。
- **上游错误**：保留 P 端失败响应的错误原因，便于定位缓存与推理异常。
- **API 缓存亲和**：新增轻量网关，通过私有 Unix socket 选择 API 进程；保留 GPU rank 头、原请求和 SSE，结合前缀复用与 API 在途负载，不重试生成请求。
- **缓存感知**：保留历史输入和 cache_salt，使用一致的前缀哈希参与路由。
- **耦合实例负载**：可选每秒读取各 DP 的真实 GPU KV、running/waiting，结合在途请求选择低压力路。
- **估算计时**：区分分词槽等待、线程池等待和 token/前缀估算耗时。
- **Usage 统计**：使用 P 端实际命中 token，合并 D 输出，保留缓存与思考统计。
- **Worker 健康检查**：调整节点健康与负载刷新，匹配分离部署。
- **依赖版本**：固定 Cargo.lock，对齐实际使用的 Router 0.1.16 镜像。

配套 vLLM 分支同名：vllm-0.32.0-0.1.16-glm5.3-flash。
通用 1P2D 的 CX8 启动命令、Router 和 Mooncake 配置保存在配套 vLLM 仓库根目录 DEPLOYMENT.md。
耦合实例使用 VLLM_ROUTER_COUPLED_KV_BALANCE=1；PD 分离使用 PD-TOKEN-BALANCE.md 的 JSON 配置。
