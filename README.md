# Router 0.1.16 · GLM-5.3 Flash

GLM分离部署使用P `cache_aware` / D `round_robin`，支持1P2D和4P4D。镜像从本仓库 `glm5.3-flash-1103` 分支构建为 `router:glm53flash-1103`，构建与启动命令统一见[DEPLOYMENT.md](DEPLOYMENT.md)。

## 路由与交接

- **P/D路由**：Mooncake PD分离，按各实例实际DP rank登记worker。
- **低并发分流**：C8、C16、C32也按当前负载纠偏；同等前缀命中按负载和累计请求数选P，D按DP rank交错轮询各实例。
- **P/D请求计数**：选路时同步登记在途请求；P完成交接后归还，D持续到流式结束；错误、取消和连接释放均归还。
- **缓存感知**：保留历史输入与cache_salt，按一致的前缀哈希选择P。
- **Prompt ID复用**：D复用P返回的精确token IDs，避免重复分词。
- **P交接**：单token length-capped请求只作用于P副本，D保留原始输出参数。
- **Usage合并**：使用P端真实缓存命中和D端输出用量，保留思考统计。
- **LRU**：缓存感知前缀树按字符预算和清理周期淘汰，不永久保留全部请求历史。

PD路由直接执行所选策略，不加载估算Tokenizer、token容量配置或估算分词槽位。独立Program scheduling默认关闭；可选绑定策略见[说明](docs/program_scheduling.md)。

配套vLLM同名分支包含[1P2D部署](https://github.com/tokmo-infra/vllm/blob/glm5.3-flash-1103/deployment1p2d.md)与[4P4D部署](https://github.com/tokmo-infra/vllm/blob/glm5.3-flash-1103/deployment4p4d.md)。字段作用和使用示例位于各部署文档开头；[PD-TOKEN-IDS.md](PD-TOKEN-IDS.md)说明token-ID和length-cap交接。
