# Router 0.1.16 · GLM-5.3 Flash

- **P/D 路由**：支持 Mooncake PD 分离及多 P、多 D 实例。
- **D 负载预占**：选择 D 时原子预占，计数持续至流式结束；取消和连接释放时归还。
- **P/D token 负载**：P 按估算计算量分流，D 同时均衡请求数与容量归一化的 KV 长度；配置见根目录 PD-TOKEN-BALANCE.md。
- **P KV 压力反馈**：可选结合真实 GPU KV 使用率和长请求 KV 预占估算选择 P。
- **上游错误**：保留 P 端失败响应的错误原因，便于定位缓存与推理异常。
- **缓存感知**：保留历史输入和 cache_salt，使用一致的前缀哈希参与路由。
- **Usage 统计**：使用 P 端实际命中 token，合并 D 输出，保留缓存与思考统计。
- **Worker 健康检查**：调整节点健康与负载刷新，匹配分离部署。
- **依赖版本**：固定 Cargo.lock，对齐实际使用的 Router 0.1.16 镜像。

配套 vLLM 分支同名：vllm-0.32.0-0.1.16-glm5.3-flash。
C230 的三机 CX8 启动命令、Router 和 Mooncake 配置统一保存在配套 vLLM 仓库的 integrations/glm53flash/DEPLOYMENT.md。
