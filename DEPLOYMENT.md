# Router 部署 · GLM5.3 Flash 1103

P使用 `cache_aware`，D使用 `round_robin`。同一套Router命令支持1P2D和4P4D，配套模型服务使用vLLM同名分支的部署文档。Router不加载估算Tokenizer或token容量JSON，Program scheduling保持关闭。

`/v1/responses` 不在服务端保存对话。续聊时在 `input` 里发送完整历史；`previous_response_id` 留空（省略或 `null`）。传入非 null 值（含空字符串）时，Router 返回 HTTP 400，响应中的 `error.code` 为 `previous_response_id_unsupported`。同一请求不会进入 P/D；不影响 Mooncake KV 缓存。

Responses 的 `tools` 同时接受扁平函数定义和 `{"type":"function","function":{...}}` 包装。Router 转发前展开 `name`、`description`、`parameters`、`strict`；同名字段值冲突时返回 HTTP 400（`invalid_function_tool`）。续聊仍在 `input` 中携带完整历史。

Chat Completions、Responses、Messages 和 Messages token 计数入口支持图片和视频。Responses、Messages 的视频块需配套支持该输入扩展的 vLLM；旧引擎不能仅靠升级 Router 获得支持。Prefill 的输入校验状态（400、413、415、422）直接保留；响应提供可识别的参数名，不带原始提示词或媒体数据。

单张内联图片默认上限为 4 MiB（4,194,304 字节，按 Base64 解码后的图片大小计算）。超限 JPEG/PNG 在选路前压缩到上限内，P/D 使用同一份字节；无法压到上限或超限格式不受支持时返回 HTTP 400。上限可通过 `VLLM_ROUTER_GLM_INLINE_IMAGE_MAX_BYTES` 调整。纯文本、外链和上限内的图片原样转发。

视频放在用户消息的 `content` 数组中；下表 `URL` 可替换成后端允许访问的外链或内联 `data:video/mp4;base64,...`。

| 入口 | 单个视频块示例 |
| --- | --- |
| Chat Completions | `{"type":"video_url","video_url":{"url":"URL"}}` |
| Responses | `{"type":"input_video","video_url":"URL"}`，放在 `input` 中的 user 消息。配套 vLLM 也识别 `video_url` 类型及只含 `url` 的包装。 |
| Messages / count_tokens | `{"type":"video","source":{"type":"url","url":"URL"}}`，或 `{"type":"video","source":{"type":"base64","media_type":"video/mp4","data":"..."}}`。用户消息的 `tool_result.content` 也适用。 |

每请求最多处理一段内联视频。Router 在选路、P/D 拆分和重试前检查一次：源文件 ≤200,000,000 字节（解码 Base64 后），接受 MP4、MOV、MKV、WebM；视频边长 ≤8192、单帧 ≤16,777,216 像素、时长 ≤600 秒、平均帧率 ≤240 fps。符合输出预算的 H.264 MP4 验证可解码后保留原字节和 UUID；其余支持的输入按全时长采样并转为仅含画面的 H.264 MP4，移除旧 UUID。输出 ≤32 MiB、最多 96 帧，画面适配横屏或竖屏的 1280×720 包围框，全部 RGB 帧估算合计 ≤128 MiB。P、D 使用同一份处理结果；静态 Mooncake 的 token 计数只发 P。HTTP(S) 和后端允许的 `file:///` 走原路径，Router 不打开、不下载，也不占本地媒体槽。

视频与超限图片共用两个 CPU 槽，满载返回 503。视频子进程共用 45 秒时限，超时终止并回收；无效、超限或无法保留完整时间轴的视频返回协议对应的 400。临时文件随请求清理，运行容器需有可写临时目录；预留至少 500 MiB 临时空间供两个最大输入同时处理。镜像运行层需有 Debian `ffmpeg` 包（含 `ffprobe`、`libx264` 编码器）。在 CPU 构建机可执行 `cargo build --locked --bin vllm-router` 和 `python3 tests/glm_media_regression.py target/debug/vllm-router`，使用本机回环 P/D 验证真实视频、图片阈值与准入行为。

Chat、Responses、Messages 与 Messages token 计数均使用请求体大小和请求准入限制。模型不存在返回协议对应的 HTTP 404；上游接口不可用返回 502。Messages 响应包含 request-id 和 x-request-id；转发时另用 x-router-request-id 关联客户端请求，原 P/D 交接 ID 保持不变。

GLM-5.3-Flash 的原厂行为是始终启用思考，可选择 low、high 或 max。本 Router 不改写思考或停止参数。外链图片由 vLLM 按后端网络和媒体域名配置获取。

Chat 的工具终态和 usage 由配套 vLLM 生成，Router 透传。新版引擎正常完成工具调用时返回 `finish_reason=tool_calls`，预算截断保留 `length`。流式 Chat 的计费以 `[DONE]` 前 `choices: []` 的最终 usage 为准；显式开启连续计量时，中间块只用于展示累计进度。生产是否已使用新版引擎以运行镜像为准。

`--request-timeout-secs` 是 Router 每次上游 HTTP 调用（含读取响应体）的总超时。本文 1P2D 默认 3600 秒，4P4D 保持 600 秒；持续出流不会重置计时。调整时按原 IP 和端口重建 Router 容器，网关和客户端须一并核对超时；一小时端到端能力需另行验收。

联网搜索：本部署不支持。Responses 不存储会话，续聊使用完整历史。

内网推理入口默认无需客户端 key。需要鉴权的交付环境可配置可信 HTTPS 身份校验端点（ROUTER_CLIENT_VALIDATION_URLS，多个以逗号分隔）。Router 将客户端 Bearer key 提交校验：身份服务返回 200 放行，401/403 拒绝；校验结果不跨请求缓存，身份服务不可用时返回 503。此 key 与 P/D 内部 key 独立。启用后示例中的 Router 请求和健康检查使用该环境的有效客户端凭证。

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
| `ROUTER_REQUEST_TIMEOUT_SECS` | 1P2D：3600 秒；4P4D：600 秒 | 本文启动变量，传给 `--request-timeout-secs`；可按部署覆盖。 |
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
    ROUTER_REQUEST_TIMEOUT_SECS="${ROUTER_REQUEST_TIMEOUT_SECS:-3600}"
    expected_p=1; expected_d=2
    ;;
  4p4d)
    export P_HOSTS="${P_HOSTS:-11.8.0.81 11.8.0.82 11.8.0.83 11.8.0.84}"
    export D_HOSTS="${D_HOSTS:-11.8.0.85 11.8.0.86 11.8.0.87 11.8.0.88}"
    export ROUTER_IP="${ROUTER_IP:-11.8.0.86}"
    ROUTER_REQUEST_TIMEOUT_SECS="${ROUTER_REQUEST_TIMEOUT_SECS:-600}"
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
  -e API_KEY_VALIDATION_URLS="${ROUTER_CLIENT_VALIDATION_URLS:-}" \
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
  --request-timeout-secs "$ROUTER_REQUEST_TIMEOUT_SECS" --worker-startup-timeout-secs 7200
~~~

## 3. 缓存与核验

缓存感知前缀树按上述字符预算和120秒周期进行LRU。64GiB是容器RAM硬上限，树节点、在途请求和两个清理周期之间的增长也占内存；长期服务结合RSS峰值调整字符预算与周期。P/D自身的共享分词与64MiB/API私有L1继续使用，与Router不执行的估算分词是两回事。

L2为Mooncake Store的主机内存池。建议模型、Router和分词服务启动并预热后，将剩余可用内存的85%用于L2；已经分配池时先加回当前池容量再算总预算。L3 SSD默认关闭，可选目录、磁盘配额、LRU和读回参数见配套vLLM文档。

~~~bash
docker image inspect "$ROUTER_IMAGE" --format '{{.Id}}'
curl -fsS -H "Authorization: Bearer ${ROUTER_CLIENT_API_KEY:-$API_KEY}" \
  "http://$ROUTER_IP:18300/workers" |
  jq -e --argjson n "$EXPECTED_DP_RANKS" \
    '.total == $n and (.workers | length == $n) and all(.workers[]; .is_healthy == true)'
~~~

1P2D应有24个健康rank，4P4D应有64个。D按rank先后交错各实例，低并发也会轮到多台机器；C8等并发小于总rank数时，后续请求继续轮询其余rank。核对D轮询、复用和length-cap开关、无CPU quota及内存限制。P/D的MTP、异步调度、图和缓存由引擎部署命令核验。
