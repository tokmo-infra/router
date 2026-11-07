# Router 部署 · GLM5.3 Flash 1103 / 4P4D

当前只使用这一套启动方案：**P `cache_aware`，D `round_robin`**。使用已经实测的 `router:vllm-yyf`，保留 P→D prompt ID 复用；关闭 TokenBalancer、program scheduling 和关联分词槽位分配。旧的容量路由、16 CPU / 16 GiB 启动方案不再使用。

## 1. 镜像与配置

| 项目 | 配置 |
| --- | --- |
| 镜像 | `router:vllm-yyf`，本地标签，先同步镜像再启动 |
| image ID | `sha256:4322e704665f87ef92204d63af35a92c02baecc8f522a8f3803bf6d17dffe687` |
| Router 二进制 SHA256 | `b366ce030714f6a95c5b30ffc93f5287bf611fd6d5161c20beb84c0ea73b04de` |
| 位置与端口 | 6000D-86；API 18300，metrics 18301 |
| 实例 | 81–84 每台双 EP4 P，共 8P；85–88 每台 EP8 D，共 4D，合计 64 rank |
| CPU / 内存 | 不设 CPU quota；64 GiB RAM / 96 GiB RAM+swap，`cpuset-mems=0` |
| 交接 | Mooncake，prompt ID 复用=1，P force length cap=1 |
| 请求上限 | 总并发 4096，请求超时 600 秒，worker 启动等待 7200 秒，不重试 |
| 监听 / 日志 | backlog 4096；local 日志每份 128 MiB，保留 5 份 |

P/D 的模型、Tokenizer 和 chat template 保持一致。Router 不加载估算 Tokenizer，不挂载模型目录或 token-balance JSON；P/D 服务自身的共享分词和 L1 缓存继续使用。`4096` 是 Router 总请求并发上限，不是分词槽位数。

### Router 内存与 LRU

缓存感知前缀树有后台 LRU，默认每 120 秒淘汰；`--max-tree-size 2147483648` 是每个 worker/tenant 的保留字符预算，不是 Router 总 RAM 字节数。旧前缀超过预算后在下一轮清理，不会永久保存全部请求历史。

64 GiB 仍是容器硬内存上限：在途请求、树节点和各 worker 的条目、两个淘汰周期间的增长都会占内存，LRU 不按 RSS 达到 64 GiB 才触发，所以仍存在 OOM 的可能。长期运行观察 RSS/峰值；若需更小占用，可调低 `--max-tree-size` 或缩短 `--eviction-interval`，再比较命中和性能。本文命令保留实测参数。

### 模型的 L2 / L3 KV 缓存

L2 是 Mooncake Store 主机内存池，建议模型、Router、共享分词等服务预热后，将剩余可用内存的 **85%** 用作 L2，留下 15% 余量；已有池时先加回当前池容量，再计算新的总预算。750 GiB/台是本次基线，L2 的淘汰由配套 Master 管理。

L3 是可选的本地 NVMe SSD offload，本轮未启用。配置包括 Master/Store 的 `enable_offload`、Store SSD 目录挂载、bucket 后端、LRU 与磁盘字节配额，以及 P/D 配套的 SSD 读回设置。完整计算命令和可选 L3 参数见 vLLM 同名分支 `deployment4p4d.md` 第 5 节；两者不需要修改 Router 的轮询配置。

## 2. 启动

先按配套 vLLM 同名 `glm5.3-flash-1103` 分支的 `deployment4p4d.md` 启动 Master、全部 Store 和 12 个模型实例，再在 Router 主机的 root Bash 会话执行。API key 与 P/D 一致，通过环境或交互输入，不写入版本库。以下参数与六策略冷测的轮询组一致。

```bash
set -euo pipefail
export ROUTER_IMAGE="${ROUTER_IMAGE:-router:vllm-yyf}"
export ROUTER_IP="${ROUTER_IP:-11.8.0.86}"
export SERVICE_PREFIX="${SERVICE_PREFIX:-glm53flash-1103-4p4d}"
export P_HOSTS="${P_HOSTS:-11.8.0.81 11.8.0.82 11.8.0.83 11.8.0.84}"
export D_HOSTS="${D_HOSTS:-11.8.0.85 11.8.0.86 11.8.0.87 11.8.0.88}"
if [ -z "${API_KEY:-}" ]; then
  read -r -s -p 'API key: ' API_KEY; printf '\n'
fi
: "${API_KEY:?API key 必须非空}"
export API_KEY

read -r -a p_hosts <<< "$P_HOSTS"
read -r -a d_hosts <<< "$D_HOSTS"
test "${#p_hosts[@]}" -eq 4
test "${#d_hosts[@]}" -eq 4
routes=()
for ip in "${p_hosts[@]}"; do
  routes+=(--prefill "http://$ip:18483" 19083 --prefill "http://$ip:18493" 19093)
done
for ip in "${d_hosts[@]}"; do
  routes+=(--decode "http://$ip:18483")
done

docker run -d --pull=never --name "$SERVICE_PREFIX-router" \
  --runtime=runc --network host --restart no \
  --memory 64g --memory-swap 96g --cpuset-mems 0 \
  --ipc private --shm-size 64m --ulimit nofile=1048576:1048576 \
  --log-driver local --log-opt max-size=128m --log-opt max-file=5 \
  -e OPENAI_API_KEY="$API_KEY" -e NVIDIA_VISIBLE_DEVICES=void \
  -e NO_PROXY='*' -e no_proxy='*' \
  -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
  -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1 \
  -e LD_PRELOAD=/usr/local/lib/router-listener.so \
  -e AI_INFRA_LISTEN_BACKLOG=4096 \
  --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
  --host 0.0.0.0 --port 18300 \
  --prometheus-host 0.0.0.0 --prometheus-port 18301 \
  --vllm-pd-disaggregation --kv-connector mooncake \
  "${routes[@]}" \
  --prefill-policy cache_aware --decode-policy round_robin \
  --max-tree-size 2147483648 --max-concurrent-requests 4096 \
  --disable-retries --api-key "$API_KEY" --log-level warn \
  --request-timeout-secs 600 --worker-startup-timeout-secs 7200
```

不传 `VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG`，不加 `--enable-program-scheduling`，不设置估算分词槽位。镜像自带 `/usr/local/lib/router-listener.so`。prompt ID 复用与 length-cap 两项均在部署中显式开启；length-cap 只修改发往 P 的单 token 请求副本，D 保留客户端原始输出参数。

## 3. 核验与实测结果

```bash
docker image inspect "$ROUTER_IMAGE" --format '{{.Id}}'
curl -fsS -H "Authorization: Bearer $API_KEY" \
  "http://$ROUTER_IP:18300/workers" |
  jq -e '.total == 64 and (.workers | length == 64) and all(.workers[]; .is_healthy == true)'
```

image ID 应与上表一致，`/workers` 应显示 64 个健康 rank。核对容器为 D 轮询、无 CPU quota、64 GiB / 96 GiB 内存配置，prompt ID 复用及 length-cap 为 1，TokenBalancer 和 program scheduling 未启用。

2026-11-07，SWE130k 原序前 825 完整会话、73295 轮，C550/U825/N825；六组每组重启服务并清空 KV、Tokenizer 和 Router 缓存，P 固定 `cache_aware`，仅改变 D 策略。统计完成 55% 后至持续欠载起点的完整区间。

| D 策略 | 总 TPM / 万 | TTFT Mean / P50 / P90 / P99（秒） | TPOT Mean（毫秒） | 全程 HTTP500 |
| --- | ---: | --- | ---: | ---: |
| round_robin | 16037.53 | 4.257 / 3.267 / 7.506 / 18.857 | 24.946 | 0 |

轮询为六组中吞吐最高、TTFT Mean 和 P99 最低的方案。完整 P/D、Mooncake 和共享 Tokenizer 启动命令以配套 vLLM 分支的 `deployment4p4d.md` 为准。
