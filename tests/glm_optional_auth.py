"""CPU acceptance: opt-in validator on a loopback Router with no real credentials."""

import contextlib
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from glm_media_regression import CHAT, COUNT, MESSAGES, RESPONSES, Worker


class Identity(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        key = self.headers.get("Authorization")
        self.send_response(200 if key == "Bearer test-operator" else self.server.status)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main(binary):
    with contextlib.ExitStack() as stack:

        def serve(handler):
            server = stack.enter_context(ThreadingHTTPServer(("127.0.0.1", 0), handler))
            server.received, server.trace_ids = [], []
            threading.Thread(target=server.serve_forever, daemon=True).start()
            stack.callback(server.shutdown)
            return server

        p, d, identity = serve(Worker), serve(Worker), serve(Identity)
        identity.status = 401
        with socket.socket() as s, socket.socket() as m:
            s.bind(("127.0.0.1", 0))
            m.bind(("127.0.0.1", 0))
            public, metrics = s.getsockname()[1], m.getsockname()[1]
        env = {
            **os.environ,
            "NO_PROXY": "*",
            "no_proxy": "*",
            "TOKIO_WORKER_THREADS": "2",
            "API_KEY_VALIDATION_URLS": f"http://127.0.0.1:{identity.server_port}/check",
        }
        env.pop("OPENAI_API_KEY", None)
        proc = subprocess.Popen(
            [
                binary,
                "--host",
                "127.0.0.1",
                "--port",
                str(public),
                "--prometheus-host",
                "127.0.0.1",
                "--prometheus-port",
                str(metrics),
                "--vllm-pd-disaggregation",
                "--kv-connector",
                "mooncake",
                "--prefill",
                f"http://127.0.0.1:{p.server_port}",
                str(p.server_port),
                "--decode",
                f"http://127.0.0.1:{d.server_port}",
                "--disable-retries",
                "--log-level",
                "warn",
            ],
            env=env,
        )

        def stop():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        stack.callback(stop)

        def call(path, key=None):
            headers = {"Content-Type": "application/json"}
            if key is not None:
                headers["Authorization"] = "Bearer " + key
            body = {
                "model": "glm-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
            if path == RESPONSES:
                body = {"model": "glm-test", "input": "hello"}
            with contextlib.closing(
                http.client.HTTPConnection("127.0.0.1", public, timeout=5)
            ) as c:
                if path == "/workers":
                    c.request("GET", path, headers=headers)
                else:
                    c.request("POST", path, json.dumps(body), headers)
                r = c.getresponse()
                return r.status, json.loads(r.read())

        for _ in range(150):
            assert proc.poll() is None, "router exited"
            try:
                status, reply = call("/workers", "test-operator")
                if (
                    status == 200
                    and len(reply["workers"]) == 2
                    and all(w["is_healthy"] for w in reply["workers"])
                ):
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            raise AssertionError("loopback workers not ready")
        for route in (CHAT, RESPONSES, MESSAGES, COUNT):
            for key, auth_status, want in (
                (None, 200, 401),
                ("test-client", 200, 200),
                ("test-client", 403, 401),
                ("test-client", 500, 503),
                ("test-client", 200, 200),
            ):
                before = len(p.received), len(d.received)
                identity.status = auth_status
                status, reply = call(route, key)
                assert status == want, (route, want, status, reply)
                if want != 200:
                    assert (
                        "error" in reply
                        and (len(p.received), len(d.received)) == before
                    )
        print(
            "PASS: four routes, missing key, allow, revoke, validator outage, recovery"
        )


if __name__ == "__main__":
    main(sys.argv[1])
