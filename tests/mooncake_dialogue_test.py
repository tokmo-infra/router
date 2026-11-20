"""Local HTTP regression for the Mooncake Router. Set VLLM_ROUTER_BINARY to run."""

import json
import os
import socket
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4


RESPONSES = "/v1/responses"
MESSAGES = "/v1/messages"
COUNT = MESSAGES + "/count_tokens"
USAGE = {
    "input_tokens": 5,
    "cache_read_input_tokens": 17,
    "cache_creation_input_tokens": 3,
}
MODEL = "ds41-test"


def user(text):
    return {"role": "user", "content": text}


class Worker(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/query":
            self.reply({"0": {"engine_id": str(self.server.server_port)}})
        elif self.path == "/v1/models":
            self.reply({"data": [{"id": MODEL}]})
        else:
            self.reply({"status": "healthy"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        role = self.server.role
        self.server.calls.append(
            (role, self.server.server_port, self.path, body, self.headers)
        )
        if self.path == COUNT:
            if not body.get("messages"):
                self.reply({"error": {"type": "invalid_request_error"}}, 400)
            else:
                self.reply({"input_tokens": 25})
            return
        if self.path == RESPONSES and body.get("previous_response_id"):
            self.reply({"error": {"message": "Response with this ID not found."}}, 404)
            return
        if self.path == MESSAGES:
            usage = dict(USAGE)
            fixture = body["messages"][0]["content"]
            if fixture == "missing usage":
                del usage["cache_read_input_tokens"]
            reason = (
                "max_tokens" if role == "P" and fixture != "early EOS" else "end_turn"
            )
            response = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": reason,
                "usage": usage,
            }
            events = [
                ("message_start", {"type": "message_start"}),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "OK"},
                    },
                ),
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        else:
            response = {
                "id": f"resp_{role}_{uuid4().hex}",
                "object": "response",
                "status": "completed",
                "previous_response_id": None,
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK"}],
                    },
                ],
                "usage": {
                    "input_tokens": 25,
                    "input_tokens_details": {"cached_tokens": 17},
                },
            }
            events = [
                (
                    "response.created",
                    {
                        "type": "response.created",
                        "response": dict(response, status="in_progress", output=[]),
                    },
                ),
                (
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "OK"},
                ),
                (
                    "response.completed",
                    {"type": "response.completed", "response": response},
                ),
            ]
        if not body.get("stream"):
            self.reply(response)
            return
        raw = b"".join(
            f"event: {name}\r\ndata: {json.dumps(event)}\r\n\r\n".encode()
            for name, event in events
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(raw), 19):
            self.wfile.write(raw[i : i + 19])
            self.wfile.flush()


@unittest.skipUnless(
    os.getenv("VLLM_ROUTER_BINARY"), "Set VLLM_ROUTER_BINARY to a local build"
)
class MooncakeDialogueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []
        args = [
            os.environ["VLLM_ROUTER_BINARY"],
            "--vllm-pd-disaggregation",
            "--kv-connector",
            "mooncake",
            "--prefill-policy",
            "round_robin",
            "--decode-policy",
            "round_robin",
            "--disable-retries",
            "--worker-startup-timeout-secs",
            "8",
            "--log-level",
            "error",
        ]
        for role in ("P", "P", "D", "D"):
            srv = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
            srv.role, srv.calls = role, cls.calls
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            cls.addClassCleanup(srv.server_close)
            cls.addClassCleanup(srv.shutdown)
            url = f"http://127.0.0.1:{srv.server_port}"
            args += (
                ["--prefill", url, str(srv.server_port)]
                if role == "P"
                else ["--decode", url]
            )
        with socket.socket() as sock, socket.socket() as metrics:
            sock.bind(("127.0.0.1", 0))
            metrics.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            metrics_port = metrics.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        proc = subprocess.Popen(
            args
            + [
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--prometheus-host",
                "127.0.0.1",
                "--prometheus-port",
                str(metrics_port),
            ],
            env={
                **os.environ,
                "NO_PROXY": "*",
                "no_proxy": "*",
                "VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP": "1",
            },
        )
        cls.addClassCleanup(proc.wait, 10)
        cls.addClassCleanup(proc.terminate)
        for _ in range(100):
            if proc.poll() is not None:
                raise RuntimeError("Router exited during startup")
            try:
                with cls.http.open(cls.url + "/workers", timeout=1) as r:
                    workers = json.load(r)["workers"]
                if len(workers) == 4 and all(w["is_healthy"] for w in workers):
                    return
            except OSError:
                pass
            time.sleep(0.1)
        raise RuntimeError("Local workers did not become ready")

    def post(self, path, body, key="tenant-a"):
        req = urllib.request.Request(
            self.url + path,
            json.dumps({"model": MODEL, **body}).encode(),
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with self.http.open(req, timeout=5) as r:
                if r.headers.get_content_type() == "text/event-stream":
                    return r.status, [
                        json.loads(line[5:]) for line in r if line.startswith(b"data:")
                    ]
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    def test_responses_chain_and_branch(self):
        start = len(self.calls)
        _, first = self.post(
            RESPONSES, {"input": "记住 CEDAR", "instructions": "first turn only"}
        )
        parent = first["id"]
        options = {
            "previous_response_id": parent,
            "input": [user("what code?")],
            "stream": True,
            "store": True,
        }
        status, events = self.post(RESPONSES, options)
        self.assertEqual(status, 200, events)
        self.assertEqual(
            [e["type"] for e in events],
            ["response.created", "response.output_text.delta", "response.completed"],
        )
        self.assertEqual(events[1]["delta"], "OK")
        self.assertEqual(events[0]["response"]["previous_response_id"], parent)
        second = events[-1]["response"]
        self.assertEqual(second["previous_response_id"], parent)
        history = [user("记住 CEDAR"), *first["output"], user("what code?")]
        self.assertEqual(
            self.post(
                RESPONSES, {"previous_response_id": second["id"], "input": "third"}
            )[0],
            200,
        )
        self.assertEqual(
            self.post(RESPONSES, {"previous_response_id": parent, "input": "branch"})[
                0
            ],
            200,
        )
        calls = self.calls[start:]
        self.assertEqual([c[0] for c in calls], ["P", "D"] * 4)
        self.assertNotEqual(calls[0][1], calls[2][1])
        self.assertNotEqual(calls[1][1], calls[3][1])
        for pair, expected in [
            (calls[2:4], history),
            (calls[4:6], history + second["output"] + [user("third")]),
            (calls[6:8], [user("记住 CEDAR"), *first["output"], user("branch")]),
        ]:
            for _, _, _, body, _ in pair:
                self.assertEqual(body["input"], expected)
                self.assertNotIn("previous_response_id", body)
                self.assertNotIn("instructions", body)

    def test_opt_out_and_scope(self):
        _, parent = self.post(RESPONSES, {"input": "hello"})
        query = {"previous_response_id": parent["id"], "input": "again", "store": False}
        status, child = self.post(RESPONSES, query)
        self.assertEqual(status, 200, child)
        self.assertEqual(child["previous_response_id"], parent["id"])
        start = len(self.calls)
        for previous, key in [
            (child["id"], "tenant-a"),
            (parent["id"], "tenant-b"),
            ("missing", "tenant-a"),
        ]:
            status, error = self.post(
                RESPONSES, dict(query, previous_response_id=previous), key
            )
            self.assertEqual(status, 404, error)
            self.assertEqual(
                json.loads(error)["error"]["param"], "previous_response_id"
            )
        self.assertEqual(len(self.calls), start)

    def test_stateless_replay(self):
        original = [user("remember CEDAR")]
        _, first = self.post(RESPONSES, {"input": original, "store": False})
        history = original + first["output"] + [user("again")]
        start = len(self.calls)
        status, events = self.post(
            RESPONSES, {"input": history, "stream": True, "store": False}
        )
        self.assertEqual(status, 200, events)
        self.assertEqual(events[-1]["type"], "response.completed")
        for _, _, _, body, _ in self.calls[start:]:
            self.assertEqual(body["input"], history)

    def test_messages(self):
        for stream in (False, True):
            for text in ("normal", "early EOS"):
                with self.subTest(stream=stream, text=text):
                    status, first = self.post(
                        MESSAGES, {"messages": [user(text)], "max_tokens": 128}
                    )
                    self.assertEqual(status, 200, first)
                    messages = [
                        user(text),
                        {"role": first["role"], "content": first["content"]},
                        user("next"),
                    ]
                    body = {
                        "messages": messages,
                        "max_tokens": 128,
                        "stop_sequences": ["END"],
                        "stream": stream,
                    }
                    start = len(self.calls)
                    status, data = self.post(MESSAGES, body)
                    self.assertEqual(status, 200, data)
                    if stream:
                        self.assertEqual(data[1]["delta"]["text"], "OK")
                        self.assertEqual(data[-1]["type"], "message_stop")
                    else:
                        self.assertEqual(
                            (data["content"][0]["text"], data["stop_reason"]),
                            ("OK", "end_turn"),
                        )
                    prefill, decode = self.calls[start:]
                    self.assertEqual((prefill[0], decode[0]), ("P", "D"))
                    self.assertEqual(prefill[4]["anthropic-version"], "2023-06-01")
                    self.assertEqual(decode[4]["anthropic-version"], "2023-06-01")
                    self.assertEqual(
                        (prefill[3]["max_tokens"], prefill[3]["stream"]), (1, False)
                    )
                    self.assertFalse(prefill[3].get("stop_sequences"))
                    for field in body:
                        self.assertEqual(decode[3][field], body[field])
                    params = decode[3]["kv_transfer_params"]
                    if text == "early EOS":
                        self.assertIsNone(params)
                    else:
                        self.assertTrue(params["do_remote_prefill"])
                        self.assertEqual(params["remote_prefill_cached_tokens"], 17)
        start = len(self.calls)
        self.assertEqual(
            self.post(
                MESSAGES, {"messages": [user("missing usage")], "max_tokens": 32}
            )[0],
            500,
        )
        self.assertEqual([c[0] for c in self.calls[start:]], ["P"])

    def test_count_is_prefill_only(self):
        for messages, expected in [([user("count this")], 200), ([], 400)]:
            start = len(self.calls)
            status, data = self.post(COUNT, {"messages": messages})
            self.assertEqual(status, expected, data)
            if status == 200:
                self.assertEqual(data, {"input_tokens": 25})
            [call] = self.calls[start:]
            self.assertEqual(call[0], "P")
            self.assertEqual(call[3], {"model": MODEL, "messages": messages})
            self.assertEqual(call[4]["anthropic-version"], "2023-06-01")


if __name__ == "__main__":
    unittest.main()
