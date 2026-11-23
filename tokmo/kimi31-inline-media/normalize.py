"""CPU admission policy for inline visual media entering a Kimi K3.1 Chat router."""

import base64
import binascii
import io
import math
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageOps

MIB = 1024 * 1024
IMAGE_BYTES = 4 * MIB
VIDEO_BYTES = 100_000_000  # 100 MB on the wire, before base64
MAX_INPUT_BYTES = 256 * MIB
MAX_SOURCE_PIXELS = 178_956_970  # Existing vLLM pixel guard
DECODE_BYTES = 256 * MIB
IMAGE_TOKENS = 4096
REQUEST_TOKENS = 8192  # visual embeddings only; text and timestamps are additional
MAX_VIDEO_FRAMES = 96
FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
POLICY = "kimi31-inline-v2"


class MediaPolicyError(ValueError):
    pass


def cells(w, h):
    return math.ceil(w / 28) * math.ceil(h / 28)


def native_cells(w, h, patches):
    # Match the deployed checkpoint's 14px patches, 2x2 merge, 512-patch edge.
    scale = min(
        1, math.sqrt(patches / (max(1, w // 14) * max(1, h // 14))), 7168 / w, 7168 / h
    )
    return cells(max(1, int(w * scale)), max(1, int(h * scale)))


def fit(w, h, budget, *, video=False):
    # Grid padding matters for wide/tall inputs. A pixel area bound alone can overshoot.
    unit = 2 if video else 1

    def size(scale):
        return tuple(max(unit, int(edge * scale) // unit * unit) for edge in (w, h))

    hi = min(1, 1920 / max(w, h), 1080 / min(w, h)) if video else 1
    if cells(*size(hi)) <= budget:
        return size(hi)
    lo = 0.0
    for _ in range(32):
        mid = (lo + hi) / 2
        if cells(*size(mid)) <= budget:
            lo = mid
        else:
            hi = mid
    return size(lo)


def inspect_video(raw):
    try:
        with av.open(
            io.BytesIO(raw),
            format="mov",
            options={"enable_drefs": "0", "use_absolute_path": "0"},
        ) as movie:
            stream = movie.streams.video[0]
            width, height = stream.codec_context.width, stream.codec_context.height
            duration = (
                Fraction(stream.duration) * stream.time_base
                if stream.duration
                else Fraction(movie.duration or 0, av.time_base)
            )
            fps = stream.average_rate
            if not (
                width > 0
                and height > 0
                and width * height <= MAX_SOURCE_PIXELS
                and duration > 0
                and fps
                and fps > 0
            ):
                raise MediaPolicyError(
                    "Video has unsupported dimensions, frame rate or duration"
                )
            frames = stream.frames or max(1, round(duration * fps))
            return width, height, duration, fps, frames
    except (IndexError, av.FFmpegError) as e:
        raise MediaPolicyError("Cannot read the MP4 video stream") from e


def normalize_video(raw, budget):
    w, h, duration, fps, frames = inspect_video(raw)
    per_chunk = native_cells(w, h, 16384)
    native_n = min(
        frames,
        max(1, round(frames * min(8, fps) / fps)),
        768,
        655360 // (4 * per_chunk),
        4 * (40960 // per_chunk),
    )
    native_tokens = math.ceil(native_n / 4) * per_chunk
    if (
        len(raw) <= VIDEO_BYTES
        and native_tokens <= budget
        and native_n * w * h * 3 <= DECODE_BYTES
        and max(w, h) <= 1920
        and min(w, h) <= 1080
    ):
        return raw, "video/mp4", native_tokens
    count = min(
        frames, MAX_VIDEO_FRAMES, max(1, round(duration * min(8, fps))), 4 * budget
    )
    target = fit(w, h, budget // math.ceil(count / 4), video=True)
    rate = (Fraction(count, 1) / duration).limit_denominator(1_000_000)
    if rate <= 0:
        raise MediaPolicyError("Video duration exceeds the supported sampling range")
    ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "source.mp4", Path(tmp) / "normalized.mp4"
        src.write_bytes(raw)
        for crf in (23, 32):
            filters = (
                f"setpts=PTS-STARTPTS,fps={rate},scale={target[0]}:{target[1]}:"
                "force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"
            )
            cmd = [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-protocol_whitelist",
                "file",
                "-f",
                "mov",
                "-enable_drefs",
                "0",
                "-use_absolute_path",
                "0",
                "-threads",
                "2",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-filter_threads",
                "1",
                "-vf",
                filters,
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-fs",
                str(VIDEO_BYTES + MIB),
                str(dst),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=90,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                raise MediaPolicyError(
                    "Cannot normalize the video within the processing limit"
                ) from e
            if dst.stat().st_size > VIDEO_BYTES:
                continue
            encoded = dst.read_bytes()
            nw, nh, nd, nfps, nframes = inspect_video(encoded)
            used = math.ceil(nframes / 4) * cells(nw, nh)
            if (
                0 < nframes <= count
                and used <= budget
                and nframes * nw * nh * 3 <= DECODE_BYTES
                and abs(duration - nd) <= max(Fraction(1, 1000), 1 / nfps)
                and nframes == count
            ):
                return encoded, "video/mp4", used
    raise MediaPolicyError(
        "Could not fit the complete video into the configured budget"
    )


def normalize_image(raw, mime, budget):
    try:
        with Image.open(io.BytesIO(raw)) as pic:
            w, h = pic.size
            if pic.format != FORMATS[mime] or w * h > MAX_SOURCE_PIXELS:
                raise MediaPolicyError(
                    "Image format or pixel dimensions exceed the input contract"
                )
            if getattr(pic, "n_frames", 1) != 1:
                raise MediaPolicyError("Animated images require a video input")
            used = native_cells(w, h, 65536)
            if used <= budget and len(raw) <= IMAGE_BYTES and w * h * 4 <= DECODE_BYTES:
                return raw, mime, used
            size = fit(w, h, budget)
            if pic.format == "JPEG":
                pic.draft("RGB", size)
            oriented = ImageOps.exif_transpose(pic)
            if pic.getexif().get(274) in (5, 6, 7, 8):
                size = size[::-1]
            oriented.thumbnail(size, Image.Resampling.LANCZOS)
            has_alpha = oriented.has_transparency_data
            converted = oriented.convert("RGBA" if has_alpha else "RGB")
            # Preserve transparency and sharp edges for PNG/WebP inputs.
            output = "image/jpeg" if mime == "image/jpeg" else "image/png"
            for _ in range(8):
                buf = io.BytesIO()
                if output == "image/jpeg":
                    converted.save(
                        buf, "JPEG", quality=88, subsampling=0, optimize=True
                    )
                else:
                    converted.save(buf, "PNG", compress_level=6)
                if buf.tell() <= IMAGE_BYTES:
                    return buf.getvalue(), output, cells(*converted.size)
                converted.thumbnail(
                    tuple(max(1, int(e * 0.8)) for e in converted.size),
                    Image.Resampling.LANCZOS,
                )
    except (OSError, Image.DecompressionBombError) as e:
        raise MediaPolicyError("Cannot read the image") from e
    raise MediaPolicyError("Could not fit the image into the configured budget")


def normalize_chat(request):
    """Return a new Chat payload, preserving unchanged parts, order and tenant cache salt."""
    messages = []
    report = []
    remaining = REQUEST_TOKENS
    input_bytes = 0
    videos = 0
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        raise MediaPolicyError("Expected a Chat messages array")
    source_messages = request["messages"]
    if not all(isinstance(msg, dict) for msg in source_messages):
        raise MediaPolicyError("Expected an object for each Chat message")
    pending = sum(
        1
        for msg in source_messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if isinstance(part, dict) and part.get("type") in ("image_url", "video_url")
    )
    if pending > REQUEST_TOKENS // 64:
        raise MediaPolicyError("Too many visual items for the request token budget")
    reserve = min(1024, REQUEST_TOKENS // max(1, pending))
    for message in source_messages:
        content = message.get("content")
        if not isinstance(content, list):
            messages.append(message)
            continue
        parts = []
        for item in content:
            kind = item.get("type") if isinstance(item, dict) else None
            if kind not in ("image_url", "video_url"):
                parts.append(item)
                continue
            media = item.get(kind)
            url = media.get("url") if isinstance(media, dict) else None
            if not isinstance(url, str) or not url.startswith("data:"):
                raise MediaPolicyError(
                    "This inline-media endpoint accepts base64 data URLs"
                )
            comma = url.find(",")
            head = url[:comma] if comma >= 0 else ""
            if not head.endswith(";base64"):
                raise MediaPolicyError("Missing base64 data URL header")
            mime = head[5:-7].lower()
            if not (mime in FORMATS if kind == "image_url" else mime == "video/mp4"):
                raise MediaPolicyError("Unsupported media type")
            chars = len(url) - comma - 1
            if chars > 4 * math.ceil((MAX_INPUT_BYTES - input_bytes) / 3):
                raise MediaPolicyError("Encoded media exceed the request input limit")
            try:
                raw = base64.b64decode(url[comma + 1 :], validate=True)
            except (ValueError, binascii.Error) as e:
                raise MediaPolicyError("Invalid base64 media") from e
            input_bytes += len(raw)
            if input_bytes > MAX_INPUT_BYTES or not raw:
                raise MediaPolicyError(
                    "Empty media or request exceeds input byte limit"
                )
            pending -= 1
            # Reserve room for later items so an early large clip cannot starve the last image.
            item_budget = remaining - reserve * pending
            if kind == "video_url":
                videos += 1
                if videos > 1:
                    raise MediaPolicyError("At most one video per request")
                encoded, out_type, tokens = normalize_video(raw, item_budget)
            else:
                encoded, out_type, tokens = normalize_image(
                    raw, mime, min(item_budget, IMAGE_TOKENS)
                )
            remaining -= tokens
            assert remaining >= 0
            changed = encoded is not raw
            if changed:
                newurl = (
                    "data:" + out_type + ";base64," + base64.b64encode(encoded).decode()
                )
                item = {**item, kind: {**media, "url": newurl}}
                # A client-provided identifier for the original must not alias the reduced media.
                item.pop("uuid", None)
            parts.append(item)
            report.append(
                {
                    "type": kind,
                    "changed": changed,
                    "input_bytes": len(raw),
                    "output_bytes": len(encoded),
                    "visual_tokens": tokens,
                }
            )
        messages.append({**message, "content": parts})
    return {**request, "messages": messages}, report


def serve_router_pipes():
    """Process one ordered list of media parts per length-prefixed frame."""
    import json
    import struct
    import sys

    executable = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not executable:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    if not shutil.which(executable):
        raise RuntimeError("Configured FFmpeg executable is unavailable")
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    writer.write(b"K31\n")
    writer.flush()
    while header := reader.read(4):
        size = struct.unpack(">I", header)[0]
        if size > 4 * math.ceil(MAX_INPUT_BYTES / 3) + MIB:
            raise ValueError("Router frame exceeds limit")
        parts = json.loads(reader.read(size))
        try:
            request, report = normalize_chat({"messages": [{"content": parts}]})
            reply = {
                "replacements": [
                    item if stats["changed"] else None
                    for item, stats in zip(request["messages"][0]["content"], report)
                ],
                "visual_tokens": sum(item["visual_tokens"] for item in report),
                "rewritten": sum(item["changed"] for item in report),
            }
        except (MediaPolicyError, ValueError, TypeError, AttributeError) as exc:
            reply = {"error": str(exc)}
        data = json.dumps(reply, separators=(",", ":")).encode()
        writer.write(struct.pack(">I", len(data)))
        writer.write(data)
        writer.flush()
        # Release the last request before blocking for another. Each worker is long-lived.
        del parts, reply, data
        request = report = None


if __name__ == "__main__":
    serve_router_pipes()
