"""Verify GLM media and prefill validation via a local Router and two HTTP stubs."""

import binascii
import contextlib
import concurrent.futures
import hashlib
import http.client
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def encode64(raw):
    return binascii.b2a_base64(raw, newline=False)


def decode64(encoded):
    raw = binascii.a2b_base64(encoded)
    assert encode64(raw).decode("ascii") == encoded
    return raw


CHAT = "/v1/chat/completions"
RESPONSES = "/v1/responses"
MESSAGES = "/v1/messages"
COUNT = "/v1/messages/count_tokens"
BUDGET = 65536
DEFAULT_BUDGET = 4 * 1024 * 1024
SECRET = "private-prompt-and-inline-image"
ENV_BUDGET = "VLLM_ROUTER_GLM_INLINE_IMAGE_MAX_BYTES"


def png(size):
    rng = random.Random(2026)
    rows = b"".join(b"\0" + rng.randbytes(size * 3) for _ in range(size))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def request(path, part, model="glm-test"):
    message = {
        "role": "user",
        "content": [
            {"type": "text" if path != RESPONSES else "input_text", "text": "image?"},
            part,
        ],
    }
    return {"model": model, "input" if path == RESPONSES else "messages": [message]}


def image(path, bytes_):
    data = encode64(bytes_).decode()
    url = "data:image/png;base64," + data
    if path == CHAT:
        return {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
    if path == RESPONSES:
        return {
            "type": "input_image",
            "image_url": url,
            "detail": "low",
            "uuid": "old-id",
        }
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def media(path, body):
    return body["input" if path == RESPONSES else "messages"][0]["content"][1]


def data(part):
    if "source" in part:
        return part["source"]["data"]
    url = part["image_url"]
    return (url["url"] if isinstance(url, dict) else url).split(",", 1)[1]


def fixture(tmp, fps):
    path = tmp / f"red-blue-{fps}.mp4"
    # A full second of red followed by a full second of blue; no external fixtures.
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=red:size=160x90:rate={fps}:d=1",
            "-f",
            "lavfi",
            "-i",
            f"color=blue:size=160x90:rate={fps}:d=1",
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def video(path, url, *, use_source_url=False):
    if path == CHAT:
        return {
            "type": "video_url",
            "video_url": {"url": url},
            "uuid": "original-bytes",
        }
    if path == RESPONSES:
        return {"type": "input_video", "video_url": url}
    if use_source_url or not url.startswith("data:"):
        return {"type": "video", "source": {"type": "url", "url": url}}
    mime, encoded = url[5:].split(";base64,")
    return {
        "type": "video",
        "source": {"type": "base64", "media_type": mime, "data": encoded},
    }


def video_regression(call, p, d, tmp, run_router):
    def uri(raw):
        return "data:video/mp4;base64," + encode64(raw).decode("ascii")

    def forwarded(path, part):
        before = (len(p.received), len(d.received))
        status, raw = call(path, request(path, part))
        assert status == 200, (path, status, raw)
        assert (len(p.received), len(d.received)) == (
            before[0] + 1,
            before[1] + (path != COUNT),
        )
        assert p.received[-1][0] == path
        result = media(path, p.received[-1][1])
        if path != COUNT:
            assert (
                d.received[-1][0] == path and media(path, d.received[-1][1]) == result
            )
        return result

    def rejected(path, body):
        before = (len(p.received), len(d.received))
        status, raw = call(path, body)
        assert (
            status == 400
            and b"invalid_request_error" in raw
            and SECRET.encode() not in raw
        ), (path, status, raw)
        assert before == (len(p.received), len(d.received))

    small, fast = fixture(tmp, 4), fixture(tmp, 24)
    small_url, fast_url = uri(small), uri(fast)
    common = None
    for path in (CHAT, RESPONSES, MESSAGES, COUNT):
        original = video(path, small_url)
        assert forwarded(path, original) == original
        for remote in (
            "https://media.invalid/red.mp4?sig=abc",
            "file:///only-on-worker/a.mp4",
        ):
            part = video(path, remote)
            assert forwarded(path, part) == part
        for use_url in [False, True] if path in (MESSAGES, COUNT) else [False]:
            original = video(path, fast_url, use_source_url=use_url)
            result = forwarded(path, original)
            assert result["type"] == original["type"] and "uuid" not in result
            url = (
                result["video_url"]["url"]
                if path == CHAT
                else result["video_url"]
                if path == RESPONSES
                else result["source"]["url"]
                if use_url
                else "data:"
                + result["source"]["media_type"]
                + ";base64,"
                + result["source"]["data"]
            )
            assert url.startswith("data:video/mp4;base64,")
            encoded = decode64(url.split(",")[1])
            assert encoded != fast and (common is None or encoded == common)
            common = encoded
        if path == RESPONSES:
            wrapped = {"type": "video_url", "video_url": {"url": small_url}}
            assert forwarded(path, wrapped) == wrapped
            wrong_role = request(path, original)
            wrong_role["input"][0]["role"] = "assistant"
            rejected(path, wrong_role)
        if path in (MESSAGES, COUNT):
            nested = {
                "type": "tool_result",
                "tool_use_id": "test",
                "content": [video(path, fast_url)],
            }
            assert forwarded(path, nested)["content"][0] == video(path, uri(common))
        rejected(path, request(path, video(path, "data:video/mp4;base64,%%%")))
        rejected(path, request(path, video(path, uri(SECRET.encode()))))
        rejected(path, request(path, {"type": "input_video"}))
        repeated = request(path, video(path, small_url))
        repeated["input" if path == RESPONSES else "messages"][0]["content"].append(
            video(path, fast_url)
        )
        rejected(path, repeated)
    for ext, mime, codec in (
        ("mov", "quicktime", "copy"),
        ("mkv", "x-matroska", "copy"),
        ("webm", "webm", "libvpx-vp9"),
    ):
        converted = tmp / f"source.{ext}"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(tmp / "red-blue-24.mp4"),
                "-c:v",
                codec,
                "-threads",
                "1",
                str(converted),
            ],
            check=True,
        )
        part = video(
            MESSAGES,
            f"data:video/{mime};base64," + encode64(converted.read_bytes()).decode(),
        )
        result = forwarded(MESSAGES, part)
        assert result["source"]["media_type"] == "video/mp4"
        assert b"ftyp" in decode64(result["source"]["data"])[:16]
    path = tmp / "forwarded.mp4"
    path.write_bytes(common)
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,avg_frame_rate,duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    assert (
        stream["codec_name"],
        int(stream["nb_frames"]),
        float(stream["duration"]),
        stream["avg_frame_rate"],
    ) == ("h264", 16, 2.0, "8/1"), stream
    for ts, channel in (("0.25", 0), ("1.75", 2)):
        rgb = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                ts,
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=1:1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        assert len(rgb) == 3 and rgb[channel] > 200 and rgb[2 - channel] < 30, (ts, rgb)
    print(
        f"real MP4 {len(fast)} B / 48 frames -> {len(common)} B / 16 frames, 2 s, red then blue,"
        f" P/D sha256={hashlib.sha256(common).hexdigest()}; 4 fps MP4 {len(small)} B unchanged"
    )

    # Gate both codec jobs inside ffprobe. A third inline request fails promptly,
    # while URLs/plain text still reach workers without invoking any local codec.
    probe = shutil.which("ffprobe")
    shim = tmp / "ffprobe"
    shim.write_text(
        '#!/bin/sh\ntouch "$GATE/running.$$"\nwhile [ ! -f "$GATE/open" ]; do sleep 0.05; done\nexec "$REAL_PROBE" "$@"\n'
    )
    shim.chmod(0o755)
    encoder = tmp / "ffmpeg"
    encoder.write_text('#!/bin/sh\ntouch "$GATE/codec.$$"\nexec "$REAL_FFMPEG" "$@"\n')
    encoder.chmod(0o755)
    limited = run_router(
        extra_env={
            "PATH": f"{tmp}:{os.environ['PATH']}",
            "REAL_PROBE": probe,
            "REAL_FFMPEG": shutil.which("ffmpeg"),
            "GATE": str(tmp),
        }
    )
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs = [
            pool.submit(limited, CHAT, request(CHAT, video(CHAT, small_url)))
            for _ in range(2)
        ]
        try:
            for _ in range(150):
                if len(list(tmp.glob("running.*"))) == 2:
                    break
                time.sleep(0.05)
            assert len(list(tmp.glob("running.*"))) == 2
            before = (len(p.received), len(d.received))
            status, raw = limited(CHAT, request(CHAT, video(CHAT, small_url)))
            assert (
                status == 503
                and json.loads(raw)["error"]["code"] == "video_processing_busy"
            )
            assert before == (len(p.received), len(d.received))
            remote = video(RESPONSES, "https://media.invalid/clip")
            assert limited(RESPONSES, request(RESPONSES, remote))[0] == 200
            assert media(RESPONSES, p.received[-1][1]) == remote
            assert (
                limited(RESPONSES, {"input": "input_video", "model": "glm-test"})[0]
                == 200
            )
            assert len(list(tmp.glob("running.*"))) == 2
        finally:
            (tmp / "open").touch()
        assert all(job.result()[0] == 200 for job in jobs)
    assert len(list(tmp.glob("codec.*"))) == 2
    assert limited(CHAT, request(CHAT, video(CHAT, fast_url)))[0] == 200
    assert media(CHAT, p.received[-1][1]) == media(CHAT, d.received[-1][1])
    assert len(list(tmp.glob("codec.*"))) == 3, "P and D must share one transcode"
    print(
        "admission: two slots, third inline 503, URL/text bypass; one FFmpeg invocation for P+D"
    )


class Worker(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply(
            200,
            {"0": {"engine_id": "local-test-prefill"}}
            if self.path == "/query"
            else {"status": "healthy"},
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append((self.path, body))
        self.server.trace_ids.append(self.headers.get("x-router-request-id"))
        if body.get("model") == "missing-model":
            error = {
                "type": "NotFoundError",
                "message": f"The model missing-model does not exist. {SECRET}",
            }
            if self.path not in (MESSAGES, COUNT):
                error["param"] = "model"
            self.reply(404, {"error": error})
        elif body.get("model") == "missing-route":
            self.reply(404, {"detail": f"Not Found {SECRET}"})
        elif body.get("model", "").startswith("status-"):
            status = int(body["model"].split("-")[1])
            self.reply(
                status,
                {
                    "detail": [
                        {
                            "loc": ["body", "messages", 0, SECRET],
                            "input": SECRET,
                            "msg": SECRET,
                        }
                    ]
                },
            )
        elif self.path == COUNT:
            self.reply(200, {"input_tokens": 1})
        else:
            self.reply(
                200,
                {
                    "id": "local-test-response",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
            )


def main(binary):
    large = png(200)
    small = png(4)
    assert len(large) > BUDGET > len(small)
    with contextlib.ExitStack() as stack:
        p, d = (
            stack.enter_context(ThreadingHTTPServer(("127.0.0.1", 0), Worker))
            for _ in range(2)
        )
        for backend in (p, d):
            backend.received = []
            backend.trace_ids = []
            threading.Thread(target=backend.serve_forever, daemon=True).start()
            stack.callback(backend.shutdown)

        def run_router(budget=None, extra_env=None):
            with socket.socket() as public, socket.socket() as metrics:
                public.bind(("127.0.0.1", 0))
                metrics.bind(("127.0.0.1", 0))
                port = public.getsockname()[1]
                metrics_port = metrics.getsockname()[1]
            env = {
                **os.environ,
                "NO_PROXY": "*",
                "no_proxy": "*",
                "TOKIO_WORKER_THREADS": "2",
            }
            for key in (
                ENV_BUDGET,
                "OPENAI_API_KEY",
                "VLLM_ROUTER_PD_REUSE_PROMPT_IDS",
            ):
                env.pop(key, None)
            env.update(extra_env or {})
            if budget is not None:
                env[ENV_BUDGET] = str(budget)
            proc = subprocess.Popen(
                [
                    binary,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--prometheus-host",
                    "127.0.0.1",
                    "--prometheus-port",
                    str(metrics_port),
                    "--vllm-pd-disaggregation",
                    "--kv-connector",
                    "mooncake",
                    "--prefill",
                    f"http://127.0.0.1:{p.server_port}",
                    str(p.server_port),
                    "--decode",
                    f"http://127.0.0.1:{d.server_port}",
                    "--disable-retries",
                    "--prefill-policy",
                    "round_robin",
                    "--decode-policy",
                    "round_robin",
                    "--log-level",
                    "warn",
                ],
                env=env,
            )

            def stop():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

            stack.callback(stop)

            def call(path, body=None):
                client = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                try:
                    client.request(
                        "GET" if body is None else "POST",
                        path,
                        body=None if body is None else json.dumps(body),
                        headers={
                            "Content-Type": "application/json",
                            "x-request-id": "glm-audit-trace",
                            "x-router-request-id": "not-the-trace",
                        },
                    )
                    reply = client.getresponse()
                    if path in (CHAT, RESPONSES, MESSAGES, COUNT):
                        assert reply.getheader("x-request-id") == "glm-audit-trace"
                        if path in (MESSAGES, COUNT):
                            assert reply.getheader("request-id") == "glm-audit-trace"
                    return reply.status, reply.read()
                finally:
                    client.close()

            for _ in range(150):
                assert proc.poll() is None, "local router exited"
                try:
                    status, raw = call("/workers")
                    if (
                        status == 200
                        and len(json.loads(raw)["workers"]) == 2
                        and all(
                            worker["is_healthy"]
                            for worker in json.loads(raw)["workers"]
                        )
                    ):
                        return call
                except OSError:
                    pass
                time.sleep(0.1)
            raise AssertionError("local workers not ready")

        call = run_router(BUDGET)
        for path in (CHAT, RESPONSES, MESSAGES, COUNT):
            original = image(path, large)
            status, raw = call(path, request(path, original))
            assert status == 200, (path, status, raw)
            assert p.received[-1][0] == path
            normalized = media(path, p.received[-1][1])
            jpeg_or_png = decode64(data(normalized))
            assert jpeg_or_png.startswith(b"\x89PNG") and len(jpeg_or_png) <= BUDGET
            assert data(normalized) != data(original)
            assert "uuid" not in normalized
            if path != COUNT:
                assert (
                    d.received[-1][0] == path
                    and media(path, d.received[-1][1]) == normalized
                )
        for path in (CHAT, RESPONSES, MESSAGES, COUNT):
            before = len(d.received)
            status, raw = call(
                path, request(path, image(path, small), model="missing-model")
            )
            assert status == 404 and SECRET.encode() not in raw, (path, status, raw)
            error = json.loads(raw)["error"]
            assert (
                error.get("code") == "model_not_found"
                if path in (CHAT, RESPONSES)
                else error["type"] == "not_found_error"
            )
            assert len(d.received) == before and p.trace_ids[-1] == "glm-audit-trace"
            status, raw = call(
                path, request(path, image(path, small), model="missing-route")
            )
            assert (
                status == 502
                and SECRET.encode() not in raw
                and len(d.received) == before
            )
        assert all(trace == "glm-audit-trace" for trace in d.trace_ids)
        for code in (400, 413, 415, 422):
            before = len(d.received)
            status, raw = call(
                CHAT, request(CHAT, image(CHAT, small), model=f"status-{code}")
            )
            assert status == code and json.loads(raw)["error"]["param"] == "messages", (
                status,
                raw,
            )
            assert SECRET.encode() not in raw and len(d.received) == before
        status, raw = call(CHAT, request(CHAT, {"type": "bad-content"}))
        assert (
            status == 400
            and json.loads(raw)["error"]["type"] == "invalid_request_error"
        )
        for part in (
            image(CHAT, small),
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/a.png"},
            },
        ):
            assert call(CHAT, request(CHAT, part))[0] == 200
            assert (
                media(CHAT, p.received[-1][1]) == media(CHAT, d.received[-1][1]) == part
            )
        before = (len(p.received), len(d.received))
        status, raw = call(CHAT, request(CHAT, image(CHAT, b"x" * (BUDGET + 1))))
        assert (
            status == 400 and json.loads(raw)["error"]["code"] == "invalid_inline_image"
        )
        assert before == (len(p.received), len(d.received))
        body = {"model": "glm-test", "input": "Explain the string input_video."}
        assert call(RESPONSES, body)[0] == 200
        assert p.received[-1][1]["input"] == d.received[-1][1]["input"] == body["input"]
        default = run_router()
        below, above = png(1150), png(1280)
        assert len(below) < DEFAULT_BUDGET < len(above)

        # Valid ancillary chunk makes the inclusive byte boundary exact.
        def padded_png(target_size):
            payload = bytes(target_size - len(below) - 12)
            tag = b"paDd"
            chunk = (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload))
            )
            return below[:-12] + chunk + below[-12:]

        for bytes_, should_resize in (
            (below, False),
            (padded_png(DEFAULT_BUDGET), False),
            (padded_png(DEFAULT_BUDGET + 1), True),
            (above, True),
        ):
            original = image(CHAT, bytes_)
            status, raw = default(CHAT, request(CHAT, original))
            assert status == 200, (status, raw)
            normalized = media(CHAT, p.received[-1][1])
            assert media(CHAT, d.received[-1][1]) == normalized
            if should_resize:
                resized = decode64(data(normalized))
                assert resized.startswith(b"\x89PNG") and len(resized) <= DEFAULT_BUDGET
                assert normalized != original
            else:
                assert normalized == original
        with tempfile.TemporaryDirectory() as tmp:
            video_regression(default, p, d, Path(tmp), run_router)
        print(
            "GLM media regression passed: four routes, P/D and count, 4xx diagnostics, default 4 MiB PNG"
        )


if __name__ == "__main__":
    main(sys.argv[1])
