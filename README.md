# Router 0.1.16 · GLM-5.3 Flash

当前 4P4D 部署统一使用 **`router:vllm-yyf`，P `cache_aware` / D `round_robin`**。保留 prompt ID 复用与 P length-capped 交接，关闭 TokenBalancer、program scheduling 及关联分词槽位；不设 CPU quota，64 GiB RAM / 96 GiB RAM+swap、mems0。唯一 Router 启动方案见 [DEPLOYMENT.md](DEPLOYMENT.md)，旧容量路由启动方案已移除。

2026-11-07 SWE130k / C550 六策略冷测中，轮询为 16037.53 万 TPM，TTFT Mean / P50 / P99 为 4.257 / 3.267 / 18.857 秒，全程 HTTP500 为 0。

## 源码功能

- **P/D 路由**：支持 Mooncake PD 分离及多 P、多 D 实例。
- **D 负载预占**：选择 D 时原子预占，计数持续至流式结束；取消和连接释放时归还。
- **可选 P/D token 负载**：源码保留 P 计算量与 D 容量估算；当前轮询部署不启用，功能说明见根目录 PD-TOKEN-BALANCE.md。
- **P KV 压力反馈**：可选结合真实 GPU KV 使用率和长请求 KV 预占估算选择 P。
- **上游错误**：保留 P 端失败响应的错误原因，便于定位缓存与推理异常。
- **缓存感知**：保留历史输入和 cache_salt，使用一致的前缀哈希参与路由。
- **耦合实例负载**：可选每秒读取各 DP 的真实 GPU KV、running/waiting，结合在途请求选择低压力路。
- **估算计时**：区分分词槽等待、线程池等待和 token/前缀估算耗时。
- **Usage 统计**：使用 P 端实际命中 token，合并 D 输出，保留缓存与思考统计。
- **Worker 健康检查**：调整节点健康与负载刷新，匹配分离部署。
- **依赖版本**：固定 Cargo.lock，对齐实际使用的 Router 0.1.16 镜像。

配套 vLLM 分支：`glm5.3-flash-1103`。完整 4P4D 模型、Mooncake 与共享分词配置见该分支根目录 `deployment4p4d.md`；本仓库 [PD-TOKEN-IDS.md](PD-TOKEN-IDS.md) 说明 prompt ID 复用与 length-cap 交接。
