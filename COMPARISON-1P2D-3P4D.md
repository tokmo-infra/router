# 1P2D 与 3P4D 参数和代码差异

1P2D列为原部署参数；3P4D列为本次用户指定的目标参数，不是仍在运行的原C718参数。两者使用同一套已合并的GLM镜像代码，可选能力由启动参数启用。

| 参数名 / 项目 | 原1P2D | 新3P4D |
| --- | --- | --- |
| P布局 | 82两组GPU0–3/4–7 | 81/83/84各8GPU |
| P tensor-parallel-size / data-parallel-size | 1 / 4，每组EP4 | 1 / 8，每组EP8 |
| D布局 | 83/84两个独立EP8 | 85—88一个全局EP32 |
| D data-parallel-size / data-parallel-size-local | 8 / 8 | 32 / 8 |
| D data-parallel-start-rank | 无跨机组 | 85主节点，86/87/88为8/16/24且headless |
| 总GPU数 | 24 | 56 |
| P max-num-seqs | 24/DP | 32/DP |
| D max-num-seqs | 16/DP | 32/DP |
| P max-num-batched-tokens | 2048/DP | 2048/DP |
| D max-num-batched-tokens | 1024/DP | 2048/DP |
| gpu-memory-utilization | P/D均0.95 | P/D均0.95 |
| P compilation-config.cudagraph_mode | 未显式覆盖 | NONE |
| D compilation-config | FULL_DECODE_ONLY，max_cudagraph_capture_size=64 | FULL_DECODE_ONLY，max_cudagraph_capture_size=128 |
| P api-server-count | 每EP4实例4 | 每EP8实例4 |
| D api-server-count | 每EP8实例1 | D85为8；86—88为0 |
| VLLM_MOONCAKE_LOAD_RECV_THREADS | P=4，D原生默认 | P/D=1 |
| scheduler-reserve-full-isl | true | false |
| P all2all-backend | 未显式覆盖 | allgather_reducescatter |
| D all2all-backend | 未显式覆盖 | deepep_v2 |
| AI_INFRA_DEEPEP_V2_NUM_SMS | 未设置 | D=32 |
| VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE | 未设置 | D=0 |
| P NCCL_IB_HCA | 每EP4组对应4张CX8 | mlx5_0/2/4/6 |
| P Mooncake device_name | 每EP4组对应4张CX8 | mlx5_1/3/5/7 |
| D NCCL_IB_HCA | 全部8张CX8 | mlx5_0/1/5/6 |
| D Mooncake device_name | 全部8张CX8 | mlx5_2/3/4/7 |
| NCCL_P2P_DISABLE | 未显式覆盖 | 0（未关闭） |
| NCCL_P2P_LEVEL | 未显式覆盖 | P=PXB，D=PIX |
| NCCL_SHM_DISABLE | 未显式覆盖 | 1（关闭CPU SHM传输，不是P2P） |
| Tokenizer容量 | P每API2GiB；D每API16GiB，合计48GiB | P每EP8组16GiB；D85共享8GiB＋8个64MiB L1 |
| VLLM_GLM53_TOKENIZER_SHARED_GIB | 未设置，独立API缓存 | P=15.75，D85=8 |
| VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB | 未设置 | 64 |
| API亲和代理 / VLLM_API_PRIVATE_SOCKET_DIR | 未启用 | 32110/32111网关＋/run/glm-tokenizer私有socket |
| Mooncake global_segment_size | 每台805306368000（750GiB），共2250GiB | 每台805306368000（750GiB），共5250GiB |
| Mooncake GPU客户端local_buffer_size | 每rank2147483648 | P每rank536870912；D每rank2147483648 |
| Mooncake enable_offload / offload_on_evict | false / 未启用 | Store/master true / true；GPU客户端仍false |
| L3容量 | 关闭 | P每台1TiB，D每台3TiB，共15TiB |
| remote_store_read_batch_bytes | 未设置 | 67108864 |
| placement_segments | 未设置，local_first | P首选权重1、D权重3，保留全局回退 |
| kv_load_failure_policy | fail | recompute |
| Store/Master eviction_high_watermark_ratio | 0.98 | 0.98 |
| prefix-match-unit / block-size | 256 / 64，实际物理页8960 | 相同 |
| kv-cache-dtype / speculative-config | fp8 / method=mtp,num_speculative_tokens=3 | 相同 |
| max-model-len / MoE / FlashInfer autotune | auto / 原生自动 / 原生调优 | 相同 |
| Router prefill-policy / decode-policy | cache_aware / cache_aware | 相同 |
| Router balance-abs-threshold / balance-rel-threshold | 4 / 1.5 | 相同 |

## 本次新增代码

| 模块 | 原镜像行为 | 合并后的行为 |
| --- | --- | --- |
| Tokenizer | 每API独立缓存 | 可选共享精确token-ID池＋私有L1；保留原生编码fallback与epoch隔离 |
| API listener / gateway | 共享TCP端口，API选择不保持缓存亲和 | 私有UDS锁定API；前缀亲和＋实时API负载选择，保留rank、body和SSE，不重试生成 |
| Mooncake GET预算 | 依赖GPU客户端enable_offload | 支持独立remote_store_read_batch_bytes，覆盖远端SSD读回 |
| Mooncake PUT placement | 本地优先/原生分配 | 可选按前缀hash和权重指定首选Owner，保留原生分配回退 |
| 部分读回失败 | 部分block错误，可能继续碰同一坏KV | 整请求相关KV失效；recompute后禁后续缓存读取 |
| DeepEP SM选择 | 原生带宽估算 | 可选显式AI_INFRA_DEEPEP_V2_NUM_SMS；原生kernel不改 |
| Rust Router核心 | 原21c9cfa19e34二进制 | 不改二进制；新镜像增加Python API亲和代理 |

GitHub分支主源码保留较新的scheduler/launcher，仅移植上述功能；固定0.32运行镜像通过小补丁叠加，不用旧版运行文件覆盖整个仓库。两套配置都没有恢复P-only草稿槽实验，也没有重新启用实验性SM120融合索引器。
