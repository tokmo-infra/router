# Router 部署 · GLM5.3 Flash 1103

P使用 `cache_aware`，D使用 `round_robin`。同一套Router命令支持1P2D和4P4D，配套模型服务使用vLLM同名分支的部署文档。Router不加载估算Tokenizer或token容量JSON，Program scheduling保持关闭。

## 字段作用与使用案例

| 字段 | 配置 | 作用与使用案例 |
| --- | --- | --- |
| `PD_TOPOLOGY` | `1p2d` / `4p4d` | 本文Bash的部署选择变量。1P2D登记2个EP4 P、2个EP8 D，共24rank；4P4D登记8个EP4 P、4个EP8 D，共64rank。 |
| `P_HOSTS` / `D_HOSTS` | 按主机填写IP列表 | 每台P登记18483、18493两个origin，每台D登记18483；换到另一组机器时修改这些变量。 |
| `--prefill-policy` / `--decode-policy` | `cache_aware` / `round_robin` | P按前缀缓存选择；D在健康worker之间轮询，不经过估算token容量策略。 |
| `VLLM_ROUTER_PD_REUSE_PROMPT_IDS` | 1 | 将P返回的精确token IDs交给D，长输入不重复分词；两端tokenizer与chat template一致。 |
| `VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP` | 1 | P的单token请求忽略EOS/stop以发布Mooncake KV；D保留客户端的输出长度和停止条件。 |
| `--balance-abs-threshold` / `--balance-rel-threshold` | 64 / 1.5 | P纠偏阈值。绝对阈值按健康worker平均在途数向下缩放，低并发也能避开忙碌rank；同等缓存命中优先空闲、累计请求较少的worker。一般保留默认值。 |
| `--max-tree-size` | 2147483648 | 每worker/tenant前缀树保留字符预算，不是进程RAM字节数；需要较小缓存时降低该值。 |
| `--eviction-interval` | 120秒 | 后台LRU清理周期；希望更快回收旧前缀时缩短。 |
| `--max-concurrent-requests` | 4096 | Router总在途请求上限，不是分词槽位；测试并发由客户端设置。 |
| `AI_INFRA_LISTEN_BACKLOG` | 4096 | listener shim的监听backlog；运行镜像包含对应共享库。 |
| `RUST_BUILD_IMAGE` / `ROUTER_RUNTIME_IMAGE` | `rust:1.99.1-bookworm` / `router:vllm-yyf` | 构建参数：前者提供编译器，后者提供运行库和listener shim；运行目标使用下文的`router:glm53flash-1103`。 |
| `CARGO_BUILD_JOBS` | 4 | 构建并行度；CPU资源充足时可传 `--build-arg CARGO_BUILD_JOBS=8`，不限制Router运行CPU。 |

选择1P2D的例子：

~~~bash
export PD_TOPOLOGY=1p2d
export P_HOSTS="11.8.0.82"
export D_HOSTS="11.8.0.83 11.8.0.84"
export ROUTER_IP=11.8.0.83
~~~

P侧MTP精简、NVFP4融合、前缀缓存和L2/L3属于引擎配置，见vLLM仓库 `deployment1p2d.md` / `deployment4p4d.md` 开头与对应章节。Router无需为这些优化加载模型权重或tokenizer。

## 1. 构建镜像

在Linux构建机的本仓库根目录执行。先导入 `router:vllm-yyf` 作为运行时基底，继承其Debian Bookworm共享库与 `/usr/local/lib/router-listener.so`；Rust二进制由本分支源码重新编译并覆盖。运行时基底不是要直接启动的目标镜像。Rust版本与依赖由构建镜像及Cargo.lock固定。

~~~bash
docker build -f Dockerfile.router \
  --build-arg RUST_BUILD_IMAGE=rust:1.99.1-bookworm \
  --build-arg ROUTER_RUNTIME_IMAGE=router:vllm-yyf \
  -t router:glm53flash-1103 .
docker image inspect router:glm53flash-1103 --format '{{.Id}}'
docker run --rm --runtime runc --network none \
  --entrypoint /usr/local/bin/vllm-router router:glm53flash-1103 --version
~~~

把构建产物分发到Router主机，并核对实际image ID。下面的 `--pull=never` 要求镜像预先就绪。模型、Store和Master先按配套vLLM部署文档启动；本节不重启它们。

## 2. 启动

在目标Router主机的root Bash会话执行，密钥与P/D一致，通过环境或交互输入。Router不设CPU quota，使用64GiB RAM / 96GiB RAM+swap、NUMA mems0；不启用Program scheduling。

~~~bash
set -euo pipefail
export PD_TOPOLOGY="${PD_TOPOLOGY:-4p4d}"
export ROUTER_IMAGE="${ROUTER_IMAGE:-router:glm53flash-1103}"
export SERVICE_PREFIX="${SERVICE_PREFIX:-glm53flash-1103-$PD_TOPOLOGY}"
case "$PD_TOPOLOGY" in
  1p2d)
    export P_HOSTS="${P_HOSTS:-11.8.0.82}"
    export D_HOSTS="${D_HOSTS:-11.8.0.83 11.8.0.84}"
    export ROUTER_IP="${ROUTER_IP:-11.8.0.83}"
    expected_p=1; expected_d=2
    ;;
  4p4d)
    export P_HOSTS="${P_HOSTS:-11.8.0.81 11.8.0.82 11.8.0.83 11.8.0.84}"
    export D_HOSTS="${D_HOSTS:-11.8.0.85 11.8.0.86 11.8.0.87 11.8.0.88}"
    export ROUTER_IP="${ROUTER_IP:-11.8.0.86}"
    expected_p=4; expected_d=4
    ;;
  *) echo 'PD_TOPOLOGY只支持1p2d或4p4d' >&2; exit 1 ;;
esac
if [ -z "${API_KEY:-}" ]; then
  read -r -s -p 'API key: ' API_KEY; printf '\n'
fi
: "${API_KEY:?API key必须非空}"
export API_KEY
read -r -a p_hosts <<< "$P_HOSTS"
read -r -a d_hosts <<< "$D_HOSTS"
test "${#p_hosts[@]}" -eq "$expected_p"
test "${#d_hosts[@]}" -eq "$expected_d"
export EXPECTED_DP_RANKS=$((8 * (expected_p + expected_d)))
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
  --max-tree-size 2147483648 --eviction-interval 120 \
  --max-concurrent-requests 4096 --disable-retries \
  --api-key "$API_KEY" --log-level warn \
  --request-timeout-secs 600 --worker-startup-timeout-secs 7200
~~~

## 3. 缓存与核验

缓存感知前缀树按上述字符预算和120秒周期进行LRU。64GiB是容器RAM硬上限，树节点、在途请求和两个清理周期之间的增长也占内存；长期服务结合RSS峰值调整字符预算与周期。P/D自身的共享分词与64MiB/API私有L1继续使用，与Router不执行的估算分词是两回事。

L2为Mooncake Store的主机内存池。建议模型、Router和分词服务启动并预热后，将剩余可用内存的85%用于L2；已经分配池时先加回当前池容量再算总预算。L3 SSD默认关闭，可选目录、磁盘配额、LRU和读回参数见配套vLLM文档。

~~~bash
docker image inspect "$ROUTER_IMAGE" --format '{{.Id}}'
curl -fsS -H "Authorization: Bearer $API_KEY" \
  "http://$ROUTER_IP:18300/workers" |
  jq -e --argjson n "$EXPECTED_DP_RANKS" \
    '.total == $n and (.workers | length == $n) and all(.workers[]; .is_healthy == true)'
~~~

1P2D应有24个健康rank，4P4D应有64个。D按rank先后交错各实例，低并发也会轮到多台机器；C8等并发小于总rank数时，后续请求继续轮询其余rank。核对D轮询、复用和length-cap开关、无CPU quota及内存限制。P/D的MTP、异步调度、图和缓存由引擎部署命令核验。
