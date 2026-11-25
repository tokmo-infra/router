"""Exercise the regular HTTP router with an isolated loopback worker and a built binary."""
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

BINARY = os.getenv("VLLM_ROUTER_TEST_BINARY") or shutil.which("vllm-router")
MODEL = "deepseek-v4.1-flash-1017"
KIMI = "kimi-k3.1-pro"
OPENER = build_opener(ProxyHandler({}))


class Worker(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply({"data": [{"id": MODEL}, {"id": KIMI}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append(body)
        self.reply({"id": f"resp_{len(self.server.requests)}", "object": "response",
                    "status": "completed", "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "red"}]}]})

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def router(enabled):
    worker = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
    worker.requests = []
    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    port, metrics = free_port(), free_port()
    while port == metrics:
        metrics = free_port()
    env = os.environ.copy()
    env.update(VLLM_ROUTER_DS41_IMAGE_WORKERS="1", VLLM_ROUTER_DS41_IMAGE_DEFAULT_MODEL=MODEL)
    env.pop("VLLM_ROUTER_K31_MEDIA_WORKERS", None)
    env.pop("VLLM_ROUTER_DS41_COUPLED_FRONTEND", None)
    if enabled:
        env["VLLM_ROUTER_DS41_COUPLED_FRONTEND"] = "1"

    def call(path, body=None):
        request = Request(f"http://127.0.0.1:{port}{path}",
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        try:
            response = OPENER.open(request, timeout=15)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, json.load(response)

    with worker, subprocess.Popen([BINARY, "--host", "127.0.0.1", "--port", str(port),
            "--prometheus-host", "127.0.0.1", "--prometheus-port", str(metrics),
            "--worker-urls", f"http://127.0.0.1:{worker.server_port}", "--api-key", "EMPTY",
            "--policy", "cache_aware"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError(f"router exited with {proc.returncode}")
                try:
                    if call("/v1/models")[0] == 200:
                        break
                except URLError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("router did not become ready")
            yield call, worker.requests
        finally:
            proc.terminate()
            try:
                proc.wait(4)
            except subprocess.TimeoutExpired:
                proc.kill()
            worker.shutdown()
            thread.join()


@unittest.skipUnless(BINARY, "set VLLM_ROUTER_TEST_BINARY to the router executable")
class CoupledFrontendTest(unittest.TestCase):
    def exercise(self, enabled, model):
        active = enabled and model == MODEL
        with router(enabled) as (call, requests):
            image = "data:image/png;base64,@@@@"
            for path, extra in [
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image}}]}]}),
                ("/v1/responses", {"input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": image}]}]}),
                ("/v1/messages", {"max_tokens": 16, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "@@@@"}}]}]}),
            ]:
                status, _, _ = call(path, {"model": model, **extra})
                self.assertEqual(status, 400 if active else 200, path)
            self.assertEqual(len(requests), 0 if active else 3)
            status, headers, first = call("/v1/responses", {"model": model, "input": "Remember red.", "store": True})
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("x-ds-visual-tokens"), "0" if active else None)
            status, _, _ = call("/v1/responses", {"model": model, "input": "What color?",
                                 "previous_response_id": first["id"], "store": False})
            self.assertEqual(status, 200)
            if active:
                self.assertNotIn("previous_response_id", requests[-1])
                self.assertEqual([v["role"] for v in requests[-1]["input"]], ["user", "assistant", "user"])
            else:
                self.assertEqual(requests[-1]["input"], "What color?")
                self.assertEqual(requests[-1]["previous_response_id"], first["id"])

    def test_deepseek_coupled(self):
        self.exercise(True, MODEL)

    def test_deepseek_default(self):
        self.exercise(False, MODEL)

    def test_other_model_on_opted_in_router(self):
        self.exercise(True, KIMI)


if __name__ == "__main__":
    unittest.main()
