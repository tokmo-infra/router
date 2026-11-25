# DeepSeek-V4.1 图片输入

共享 vLLM P/D Router 的 DS4.1 实例可设置 `VLLM_ROUTER_DS41_IMAGE_WORKERS=2`。在标准 Dockerfile 中使用构建参数 `INSTALL_DS41_IMAGE=1` 安装 Pillow。默认 worker 数为 0。DS 策略匹配 `deepseek-v4.1-flash-1017`、`deepseek-v4.1-flash` 和 `deepseek-4.1-flash`（忽略大小写）。其他模型名沿用原有路径。专用 DS 入口可设置 `VLLM_ROUTER_DS41_IMAGE_DEFAULT_MODEL=deepseek-v4.1-flash-1017`，使省略模型名的请求也启用图片处理；共享入口默认透传省略模型名的请求。每个实例选择一个模型策略，K31 继续使用已有的 `VLLM_ROUTER_K31_MEDIA_WORKERS`。

Chat Completions 的 `image_url`、Responses 的 `input_image`、Messages 的 base64 图片以及 Messages 计数接口共用 CPU 工作池。对内联图片处理一次后再分发 P/D。外部 HTTP(S) 链接沿用后端取图路径。DS 支持静态 JPEG、PNG、WebP、GIF。视频输入返回 4xx。纯文本绕过媒体队列。响应头 `x-ds-rewritten-items` 记录本次改写的图片数，`x-ds-visual-tokens` 汇总本次处理的内联图片预算。保存 Responses 历史时保留处理后的图片。

每张图片按 base64 前的原始字节判定，文件超过 4 MiB 或原始像素网格超出模型预算时重新编码。预算采用 DeepSeek-V4.1-Flash-1017 的 14 像素 patch、3 倍下采样、295,936 最小像素和每图 1,024 token 上限，计算包含行分隔符和首尾标记。符合预算的小图原样转发。CPU worker 的源图保护上限为每图 32 MiB、67,108,864 像素，超出时返回 4xx。需要缩小时按模型网格选择目标画布并保留长宽比，透明度使用 PNG 保存。改写结果由实际输入内容决定，以保持重复请求的媒体缓存身份。

Chat、Responses、Messages 和 Messages 计数接口按 router 收到的原始 JSON 字节数限制为 48 MiB（含边界）。若上游网关会重写 JSON，上游也需对原始客户请求体执行同一限制。

在本目录执行 `python3 -m unittest -v test_normalize` 可复验边界值、透明度和跨接口生成字节的一致性。

## DeepSeek PD 混部

单池 HTTP Router 使用 `--worker-urls` 时，设置 `VLLM_ROUTER_DS41_COUPLED_FRONTEND=1` 和 `VLLM_ROUTER_DS41_IMAGE_WORKERS=2` 可启用与 PD 分离相同的图片处理。启用内存历史时，Responses 续写在 Router 恢复已完成的输入和输出，可跨 DP rank 和 API 进程。匹配的 Messages 和 Responses 请求会计入所选 rank 的在途负载，直到响应流完成或断开。开关默认为关闭，仅作用于普通 Router 的 DeepSeek Flash 请求。省略请求模型名时依照上述默认模型配置。

执行 `VLLM_ROUTER_TEST_BINARY=/path/to/vllm-router python3 tokmo/ds41-inline-image/test_coupled_frontend.py` 可用本地模拟后端验证启用、关闭和同入口其他模型三种情况。
