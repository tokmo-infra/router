"""DeepSeek-V4.1 inline image admission for the shared PD router."""
import base64
import binascii
import io
import json
import math
import struct
import sys

from PIL import Image, ImageOps, UnidentifiedImageError

FILE_LIMIT = 4 * 1024**2
MAX_INPUT_BYTES = 256 * 1024**2
MAX_PIXELS = 64 * 1024**2
TOKEN_LIMIT = 1024
PATCH = 14
CELL = 3 * PATCH
MIN_PIXELS = 295936
FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP", "image/gif": "GIF"}


def span(width, height):
    return math.ceil(height / CELL) * (math.ceil(width / CELL) + 1) + 2


def geometry(width, height):
    """Match mm_preprocess.load_image and solve_resize_ratio including row delimiters."""
    w, h = width, height
    if w * h < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (w * h))
        w, h = int(w * scale), int(h * scale)
    if span(w, h) <= TOKEN_LIMIT:
        return (width, height), span(w, h)
    ratio = h / w
    cols = math.sqrt((TOKEN_LIMIT - 2) / ratio + 0.25) - 0.5
    rows = cols * ratio
    if cols < 1:
        target = CELL, (TOKEN_LIMIT - 2) // 2 * CELL
    elif rows < 1:
        target = (TOKEN_LIMIT - 3) * CELL, CELL
    else:
        scale = min(math.floor(cols) * CELL / w, math.floor(rows) * CELL / h)
        target = (max(PATCH, math.floor(w * scale / PATCH) * PATCH),
                  max(PATCH, math.floor(h * scale / PATCH) * PATCH))
    return target, span(*target)


def encode(picture):
    has_alpha = "A" in picture.getbands() or "transparency" in picture.info
    pic = picture.convert("RGBA" if has_alpha else "RGB")
    pic.info.clear()
    while True:
        out = io.BytesIO()
        if has_alpha:
            pic.save(out, format="PNG", compress_level=6)
        else:
            pic.save(out, format="JPEG", quality=88, subsampling=0)
        if out.tell() <= FILE_LIMIT:
            return out.getvalue(), "image/png" if has_alpha else "image/jpeg", geometry(*pic.size)[1]
        pic = pic.resize(tuple(max(1, n * 7 // 8) for n in pic.size), Image.Resampling.LANCZOS)


def normalize(part):
    kind = part["type"]
    if kind == "image":
        source = part["source"]
        mime, data = source["media_type"], source["data"]
    else:
        url_obj = part["image_url"]
        url = url_obj if isinstance(url_obj, str) else url_obj["url"]
        header, data = url.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("Expected a base64 image data URL")
        mime = header[5:-7]
    if mime not in FORMATS:
        raise ValueError("Unsupported DS image type")
    if len(data) > 4 * math.ceil(MAX_INPUT_BYTES / 3):
        raise ValueError("Image exceeds source byte budget")
    raw = base64.b64decode(data, validate=True)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Invalid image length")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != FORMATS[mime] or image.width * image.height > MAX_PIXELS:
            raise ValueError("Image format or pixel budget mismatch")
        if getattr(image, "is_animated", False):
            raise ValueError("DS supports static images")
        transpose = image.getexif().get(274, 1)
        dims = image.size[::-1] if transpose in (5, 6, 7, 8) else image.size
        target, tokens = geometry(*dims)
        if len(raw) <= FILE_LIMIT and target == dims:
            with Image.open(io.BytesIO(raw)) as check:
                check.verify()
            return None, tokens
        pic = ImageOps.exif_transpose(image)
        if target != pic.size:
            if "transparency" in pic.info or "A" in pic.getbands():
                pic = pic.convert("RGBA")
            else:
                pic = pic.convert("RGB")
            pic = ImageOps.pad(pic, target, method=Image.Resampling.LANCZOS,
                               color=(0, 0, 0, 0) if pic.mode == "RGBA" else (127, 127, 127))
        encoded, mime, tokens = encode(pic)
    b64 = base64.b64encode(encoded).decode("ascii")
    replacement = dict(part)
    replacement.pop("uuid", None)
    if kind == "image":
        replacement["source"] = {**part["source"], "media_type": mime, "data": b64}
    else:
        url = "data:" + mime + ";base64," + b64
        replacement["image_url"] = url if isinstance(url_obj, str) else {**url_obj, "url": url}
    return replacement, tokens


def serve():
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    writer.write(b"DS4\n"); writer.flush()
    while header := reader.read(4):
        length, = struct.unpack(">I", header)
        if length > 4 * math.ceil(MAX_INPUT_BYTES / 3) + 1024**2:
            raise ValueError("Image frame exceeds budget")
        try:
            items = json.loads(reader.read(length))
            result = [normalize(item) for item in items]
            replacements = [v for v, _ in result]
            reply = {"replacements": replacements, "visual_tokens": sum(t for _, t in result),
                     "rewritten": sum(v is not None for v in replacements)}
        except (ValueError, TypeError, KeyError, AttributeError, OSError, SyntaxError,
                binascii.Error, Image.DecompressionBombError, UnidentifiedImageError) as exc:
            reply = {"error": str(exc)[:200]}
        data = json.dumps(reply, separators=(",", ":")).encode()
        writer.write(struct.pack(">I", len(data))); writer.write(data); writer.flush()
        items = result = replacements = reply = data = None


if __name__ == "__main__":
    serve()
