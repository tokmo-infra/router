import base64
import io
import json
import struct
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from normalize import IMAGE_BYTES, VIDEO_BYTES, MediaPolicyError, inspect_video, normalize_chat, normalize_video


def image_part(size, fmt, color, **save_options):
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format=fmt, **save_options)
    url = (
        f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
    )
    return {"type": "image_url", "image_url": {"url": url}}


class InlineMediaTests(unittest.TestCase):
    def test_small_history_and_tenant_salt_are_preserved(self):
        part = image_part((64, 32), "WEBP", (255, 14, 77, 190), lossless=True)
        original = {
            "messages": [
                {"role": "user", "content": [part]},
                {"role": "assistant", "content": "Seen."},
            ],
            "cache_salt": "tenant-test",
            "stream": True,
        }
        normalized, items = normalize_chat(original)
        self.assertEqual(normalized, original)
        self.assertFalse(items[0]["changed"])

    def test_bytes_only_resize_preserves_alpha_and_part_order(self):
        part = image_part((1600, 1200), "PNG", (15, 70, 220, 183), compress_level=0)
        part["uuid"] = "original"
        question = {"type": "text", "text": "What color?"}
        body = {"messages": [{"role": "user", "content": [question, part, question]}]}
        result, items = normalize_chat(body)
        first, reduced, last = result["messages"][0]["content"]
        self.assertEqual(first, last)
        self.assertIsNot(reduced, part)
        self.assertNotIn("uuid", reduced)
        self.assertIn("uuid", part)
        data = base64.b64decode(reduced["image_url"]["url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(
                (image.size, image.getpixel((0, 0))), ((1600, 1200), (15, 70, 220, 183))
            )
        self.assertLess(items[0]["output_bytes"], items[0]["input_bytes"])

    def test_video_has_fixed_duration_and_reproducible_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=blue:s=224x224:r=8:d=2",
                    "-threads",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ],
                check=True,
            )
            raw = path.read_bytes()
        encoded, mime, tokens = normalize_video(raw, 32)
        self.assertEqual(mime, "video/mp4")
        self.assertLessEqual(tokens, 32)
        self.assertEqual(inspect_video(encoded)[2], 2)
        self.assertEqual(normalize_video(raw, 32)[0], encoded)

        # A valid MP4 free-space box isolates the file-size gate from geometry and tokens.
        self.assertEqual((IMAGE_BYTES, VIDEO_BYTES), (4 * 1024**2, 100_000_000))
        box_size = 65536
        padded = raw + box_size.to_bytes(4, "big") + b"free" + bytes(box_size - 8)
        with patch("normalize.VIDEO_BYTES", len(padded)):
            self.assertIs(normalize_video(padded, 8192)[0], padded)
        with patch("normalize.VIDEO_BYTES", len(padded) - 1):
            smaller, _, _ = normalize_video(padded, 8192)
            self.assertLess(len(smaller), len(padded))
            self.assertEqual(inspect_video(smaller)[2], inspect_video(raw)[2])

    def test_router_worker_recovers_after_invalid_input_and_returns_only_replacements(self):
        small = image_part((64, 32), "PNG", "blue")
        large = image_part((1600, 1200), "PNG", (14, 97, 204, 185), compress_level=0)
        frames = [
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
            [small, large],
        ]
        payload = io.BytesIO()
        for parts in frames:
            data = json.dumps(parts).encode()
            payload.write(struct.pack(">I", len(data)) + data)
        worker = Path(__file__).with_name("normalize.py")
        response = subprocess.run(
            [sys.executable, str(worker)], input=payload.getvalue(), capture_output=True,
            timeout=30, check=True,
        )
        stream = io.BytesIO(response.stdout)
        self.assertEqual(stream.readline(), b"K31\n")

        def read_frame():
            size, = struct.unpack(">I", stream.read(4))
            return json.loads(stream.read(size))

        self.assertIn("error", read_frame())
        reply = read_frame()
        self.assertEqual(reply["rewritten"], 1)
        self.assertIsNone(reply["replacements"][0])
        self.assertTrue(reply["replacements"][1]["image_url"]["url"].startswith("data:image/png;"))
        self.assertEqual(stream.read(), b"")

    def test_endpoint_only_accepts_inline_media(self):
        with self.assertRaisesRegex(MediaPolicyError, "data URLs"):
            normalize_chat(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://example.invalid/p.png"
                                    },
                                }
                            ],
                        }
                    ]
                }
            )
