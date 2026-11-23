# DeepSeek-V4.1 图片输入

共享 vLLM P/D Router 的 DS4.1 实例可设置 `VLLM_ROUTER_DS41_IMAGE_WORKERS=2`。在标准 Dockerfile 中使用构建参数 `INSTALL_DS41_IMAGE=1` 安装 Pillow。默认 worker 数为 0。DS 策略只处理明确指定模型名 `deepseek-v4.1-flash-1017` 的请求。其他模型名沿用原有路径。专用 DS 入口可设置 `VLLM_ROUTER_DS41_IMAGE_DEFAULT_MODEL=deepseek-v4.1-flash-1017`，使省略模型名的请求也启用图片处理；共享入口默认透传省略模型名的请求。每个实例选择一个模型策略，K31 继续使用已有的 `VLLM_ROUTER_K31_MEDIA_WORKERS`。

Chat Completions 的 `image_url`、Responses 的 `input_image`、Messages 的 base64 图片以及 Messages 计数接口共用 CPU 工作池。对内联图片处理一次后再分发 P/D。外部 HTTP(S) 链接沿用后端取图路径。DS 支持静态 JPEG、PNG、WebP、GIF。视频输入返回 4xx。纯文本绕过媒体队列。响应头 `x-ds-rewritten-items` 记录本次改写的图片数，`x-ds-visual-tokens` 汇总本次处理的内联图片预算。保存 Responses 历史时保留处理后的图片。

每张图片按 base64 前的原始字节判定，文件超过 4 MiB 或原始像素网格超出模型预算时重新编码。预算采用 DeepSeek-V4.1-Flash-1017 的 14 像素 patch、3 倍下采样、295,936 最小像素和每图 1,024 token 上限，计算包含行分隔符和首尾标记。符合预算的小图原样转发。CPU worker 的源图保护上限为每图 256 MiB、67,108,864 像素，超出时返回 4xx。需要缩小时按模型网格选择目标画布并保留长宽比，透明度使用 PNG 保存。改写结果由实际输入内容决定，以保持重复请求的媒体缓存身份。

在本目录执行 `python3 -m unittest -v test_normalize` 可复验边界值、透明度和跨接口生成字节的一致性。
