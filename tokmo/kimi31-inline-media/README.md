# Kimi K3.1 Chat 内联媒体压缩

在 vLLM P/D Router 的 `POST /v1/chat/completions` 入口处理一次媒体，覆盖静态和服务发现两种路由模式。P、D 收到同一份最终字节。接口支持单个内联 MP4 和 JPEG、PNG、静态 WebP 图片，覆盖历史消息、图文混排及流式输出。适配的 checkpoint 是 Kimi-K3.1-Pro，patch 为 14 像素，空间合并为 2×2，时间块最多四帧。策略版本为 `kimi31-inline-v2`。

## 压缩条件

| 输入 | 触发条件及处理 |
| --- | --- |
| 图片 | 每张上限 4 MiB、4,096 个视觉 embedding。对超额输入等比缩放或重新编码。PNG、WebP 输出保留透明度。输入和输出尺寸相同时也可以只降低文件字节数。 |
| 视频 | 每段文件及转码输出上限 100 MB（100,000,000 字节，base64 编码前），画幅适配横屏或竖屏的 1920×1080 包围框。按实际帧率、总时长、checkpoint 采样及原始尺寸估算帧缓冲；预计超过 256 MiB 或当前视觉配额时也触发处理。处理时优先保留完整时间轴，最多 8 fps、96 帧，按剩余配额缩放，以 H.264 MP4 输出。模型使用画面，转码时移除音轨。 |
| 混合请求 | 依消息顺序分配共 8,192 个视觉 embedding，为后续媒体预留配额。文字、时间戳、媒体头和生成输出另占上下文。符合当次配额的小素材沿用原始字节。 |
| 接入上限 | 整个请求原始媒体合计最多 256 MiB，单张图片或视频源帧最多 178,956,970 像素。本入口只接受 base64 data URL。对象存储的 URL 可由受信的上传服务处理后传入。损坏、超出接入上限或无法在限时内完整转码的输入返回明确错误。 |

相同输入、配额和编解码器版本生成稳定的媒体字节。改写媒体时移除原输入的显式 UUID，后端对最终内容计算缓存身份。追加媒体触发重新分配、更新编码器或改变压缩策略时，受影响的历史媒体可能需要重新建立 KV。建议同一灰度池固定 CPU 架构和编解码器镜像。

## Router 内置模式

构建 Router 镜像时使用 `--build-arg INSTALL_K31_MEDIA=1`，或者在自建镜像的 Python 3.12 环境安装 `media-requirements.txt`。启动专用 K31 Router 时设置 `VLLM_ROUTER_K31_MEDIA_WORKERS=2`。取值 1–16，默认 0 保留原有 Chat 行为。Router 启动时校验依赖并创建指定数量的本地子进程，通过管道调用随二进制打包的同版压缩策略，无需额外端口。启用时此 Chat 入口适用上表的内联媒体契约。

媒体处理先于选 P/D、预留负载、拆分和重试。纯文本直接走原有路由。每个 worker 同时处理一个请求，另有每 worker 两个等待位置，满载时媒体请求返回 429。取消或超时会终止本次 worker 及其 FFmpeg 子进程，下次使用时补齐。请求成功后可从响应头 `X-K31-Visual-Tokens` 和 `X-K31-Rewritten-Items` 查看处理结果。入口保留普通 JSON 和 SSE 返回方式。

## 独立对照入口

安装依赖后将 `K31_ROUTER_URL` 指向已有的 P/D Router。下面的命令在本机回环地址启动独立入口。接入业务时，调用方鉴权、请求体上限和并发准入应放在压缩之前。适配器向上游传递 Authorization 和 X-Request-Id。

```sh
cd tokmo/kimi31-inline-media
export K31_ROUTER_URL=http://127.0.0.1:20700
uv run --no-project --with-requirements requirements.txt uvicorn app:app \
  --host 127.0.0.1 --port 20710 --limit-concurrency 4
```

响应头 `X-K31-Visual-Tokens` 为视觉 embedding 预算核算值，`X-K31-Rewritten-Items` 为改写项数，`X-K31-Normalized-Bytes` 为转发的媒体总字节数。真实输入 token 数使用响应的 `usage.prompt_tokens`。同一进程串行执行 CPU 压缩。系统安装的 FFmpeg 优先，也可显式设置 `FFMPEG_BINARY`，或使用依赖包内的二进制。视频使用 FFmpeg 的 fps 和 scale 滤镜，图片使用 Pillow 等比缩图。[FFmpeg 滤镜](https://ffmpeg.org/ffmpeg-filters.html) · [Pillow Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail)

```sh
uv run --no-project --with-requirements requirements.txt python -m unittest -v test_normalize
```

## 内置入口的隔离 1P1D 验收

P57/D90 实测，53,862,085 字节的高码率 MP4 原样通过。给同一 MP4 添加合法的填充 box 至 100,000,000 字节仍原样通过，100,000,001 字节时转码为 2,358,868 字节，三次均返回正确颜色。对 7,683,692 字节透明 PNG，输出 10,293 字节，保留透明度。

在两次 40 秒的混跑窗口内，每秒到达 6 个纯文本、1.5 个图片、1.5 个视频请求，固定输出 16 tokens。对照 Router 接收已按同一策略压好的素材，候选 Router 接收原图、原片。两组均使用相同资源配额，2,400 个定时请求（含纯文本基线）全部成功。合并窗口后的端到端 P99 如下。

| 请求 | 对照（ms） | 内置（ms） |
| --- | ---: | ---: |
| 纯文本基线 | 228 | 227 |
| 混跑中的文本 | 1,399 | 1,331 |
| 混跑中的 4032×3024 JPEG | 1,456 | 1,446 |
| 混跑中的 7.5 秒 4K MP4 | 1,592 | 1,797 |

另外以每 8 秒一段恰好 100 MB 的视频与文本、图片混跑。每组 192 个文本、48 张图片、4 段视频，全部成功。文本端到端 P99 为对照 881 ms、内置 905 ms。内置路径上这 4 次视频耗时 4.80–4.94 秒，对照为 3.77–3.89 秒。

## 模型与 KV 核对

以 7.5 秒、3840×2160 的 MP4 对照，原文件只有 63,368 字节，但 36 个原始 RGB 采样帧需要 895,795,200 字节，超过后端的 715,827,880 字节请求解码预算。压缩后保留全部 60 帧和 7.5 秒时长，画幅为 846×476，正放和倒放返回相反颜色顺序。30 秒素材压到 96 帧，模型仍将转场定位在第 15 秒。这些测试使用合成恒帧率素材。

同一问题下，1080p 输入从 40,581 降到 8,119 个 prompt tokens，4032×3024 图片从 15,666 降到 4,184，两次读取图片中的数字均为 73。26,135-token 请求重复时 D 本地 KV 命中 16,384 tokens；相同形状的视频倒序后，D 本地命中为 0，返回倒序答案。D 端 GPU KV 池在原版与视频版均为 32,315,205 tokens。
