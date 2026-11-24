import base64
import io
import struct
import unittest
import zlib

from PIL import Image

from normalize import FILE_LIMIT, TOKEN_LIMIT, geometry, normalize


def fixture(w, h, alpha=False):
    stream = io.BytesIO()
    Image.new("RGBA" if alpha else "RGB", (w, h), (255, 0, 0, 127) if alpha else "red").save(stream, "PNG")
    return stream.getvalue()


def padding(raw, size):
    extra = b"paDd" + bytes(size - len(raw) - 12)
    return raw[:-12] + struct.pack(">I", len(extra) - 4) + extra + struct.pack(">I", zlib.crc32(extra)) + raw[-12:]


def blocks(raw):
    b64 = base64.b64encode(raw).decode()
    uri = "data:image/png;base64," + b64
    return [
        {"type": "image_url", "image_url": {"url": uri, "detail": "auto"}, "uuid": "old"},
        {"type": "input_image", "image_url": uri, "detail": "auto"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
    ]


class ImagePolicyTest(unittest.TestCase):
    def test_boundaries_and_envelopes(self):
        raw = fixture(640, 480, True)
        for item in blocks(padding(raw, FILE_LIMIT)):
            self.assertIsNone(normalize(item)[0])
        chat, responses, messages = [normalize(item)[0] for item in blocks(padding(raw, FILE_LIMIT + 1))]
        self.assertNotIn("uuid", chat)
        self.assertEqual(chat["image_url"]["detail"], "auto")
        self.assertEqual(responses["detail"], "auto")
        self.assertEqual(chat["image_url"]["url"], responses["image_url"])
        self.assertEqual(messages["source"]["data"], responses["image_url"].split(",")[1])
        img = Image.open(io.BytesIO(base64.b64decode(messages["source"]["data"])))
        self.assertEqual(img.getpixel((0, 0)), (255, 0, 0, 127))

    def test_source_limit_applies_before_reencoding(self):
        raw = fixture(128, 128)
        limit = 32 * 1024**2
        self.assertIsNotNone(normalize(blocks(padding(raw, limit))[0])[0])
        with self.assertRaisesRegex(ValueError, "32 MiB"):
            normalize(blocks(padding(raw, limit + 1))[0])

    def test_ds_budget(self):
        for shape in ((4032, 3024), (1, 10000), (10000, 1), (1, 1), (999, 321), (16384, 8)):
            with self.subTest(shape=shape):
                size, tokens = geometry(*shape)
                self.assertGreater(tokens, 0)
                self.assertLessEqual(tokens, TOKEN_LIMIT)
                self.assertEqual(geometry(*size)[1], tokens)
        part = blocks(fixture(4000, 3000))[0]
        self.assertEqual(normalize(part), normalize(part))
        self.assertIsNotNone(normalize(part)[0])

    def test_reject_video_and_invalid_bytes(self):
        with self.assertRaises(ValueError):
            normalize({"type": "image_url", "image_url": {"url": "data:video/mp4;base64,AAAA"}})
        with self.assertRaises(ValueError):
            normalize({"type": "input_image", "image_url": "data:image/png;base64,!"})


if __name__ == "__main__":
    unittest.main()
