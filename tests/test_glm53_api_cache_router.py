"""Validate API affinity, native forwarding and failure accounting."""

import asyncio
import json
import sys
import socket
import tempfile
import unittest
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parents[1] / "integrations/glm53flash"))
from glm_api_cache_router import ApiPool, RouterConfig, create_app, request_prefixes


CONFIG = RouterConfig("http://control", ("http://api0", "http://api1"))
BODY = {"model": "glm", "cache_salt": "session-a", "messages": [
    {"role": "system", "content": "system " * 400},
    {"role": "user", "content": "hello"},
]}


class UpstreamBytes(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        yield self.data


class ApiPoolTest(unittest.TestCase):
    def test_completed_prefix_overrides_new_cold_hash_and_load_can_override(self):
        pool = ApiPool(CONFIG)
        affinity, prefixes = request_prefixes(BODY)
        first, _ = pool.reserve(affinity, prefixes)
        pool.release(first, prefixes, 200, True)
        other_affinity = (1 - first).to_bytes(8, "little")
        selected, reason = pool.reserve(other_affinity, prefixes)
        self.assertEqual((selected, reason), (first, "prefix_hit"))
        pool.release(selected, prefixes)
        pool.active[first] = 20
        selected, reason = pool.reserve(affinity, prefixes)
        self.assertEqual((selected, reason), (1 - first, "load_override"))
        pool.release(selected, prefixes)

    def test_salt_isolation_failed_requests_and_bounded_prefixes(self):
        pool = ApiPool(RouterConfig("http://control", CONFIG.api_urls, prefix_limit=2))
        affinity, prefixes = request_prefixes(BODY)
        first, _ = pool.reserve(affinity, prefixes)
        pool.release(first, prefixes, 500, True)
        self.assertFalse(pool.prefixes)
        first, _ = pool.reserve(affinity, prefixes)
        pool.release(first, prefixes, 200, True)
        self.assertEqual(len(pool.prefixes), 2)
        _, other = request_prefixes({**BODY, "cache_salt": "session-b"})
        self.assertFalse(set(k for k, _ in other) & set(pool.prefixes))


class ForwardingTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_unix_socket_preserves_body_rank_header_and_stream(self):
        class Server(uvicorn.Server):
            @contextmanager
            def capture_signals(self):
                yield

        native = FastAPI()
        payload = json.dumps(BODY).encode()

        @native.post("/v1/chat/completions")
        async def reply(request: Request):
            self.assertEqual(await request.body(), payload)
            self.assertEqual(request.headers["x-data-parallel-rank"], "7")

            async def chunks():
                yield b'data: {"choices":[]}\n\n'
                yield b'data: [DONE]\n\n'

            return StreamingResponse(chunks(), media_type="text/event-stream")

        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            path = str(Path(temp) / "api.sock")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(path)
            server = Server(uvicorn.Config(native, log_level="critical", lifespan="off"))
            task = asyncio.create_task(server.serve(sockets=[sock]))
            config = RouterConfig("http://127.0.0.1:1", ("http://api-0.local",), api_sockets=(path,))
            gateway = create_app(config)
            try:
                for _ in range(100):
                    if server.started:
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(server.started)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(gateway), base_url="http://gateway") as client:
                    response = await client.post("/v1/chat/completions", content=payload, headers={"x-data-parallel-rank": "7"})
                    self.assertEqual(response.content, b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
                self.assertEqual(sum(gateway.state.pool.active), 0)
            finally:
                await gateway.state.client.aclose()
                for api_client in gateway.state.api_clients:
                    await api_client.aclose()
                server.should_exit = True
                await task
                sock.close()

    async def exercise(self, upstream_handler, action):
        app = create_app(CONFIG, httpx.MockTransport(upstream_handler))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                await action(client, app.state.pool)
        finally:
            await app.state.client.aclose()

    async def test_sse_and_rank_headers_are_unchanged_and_request_released(self):
        raw = json.dumps(BODY, ensure_ascii=False).encode()
        sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

        async def upstream(request):
            self.assertIn(request.url.host, {"api0", "api1"})
            self.assertEqual(request.content, raw)
            self.assertEqual(request.headers["x-data-parallel-rank"], "7")
            self.assertNotIn("x-drop", request.headers)
            return httpx.Response(200, stream=UpstreamBytes(sse), headers={"content-type": "text/event-stream"})

        async def action(client, pool):
            response = await client.post(
                "/v1/chat/completions", content=raw,
                headers={"x-data-parallel-rank": "7", "connection": "x-drop", "x-drop": "x"},
            )
            self.assertEqual(response.content, sse)
            self.assertEqual(sum(pool.active), 0)
            self.assertTrue(pool.prefixes)
            self.assertEqual(sum(pool.responses.values()), 1)

        await self.exercise(upstream, action)

    async def test_native_validation_and_control_routes_stay_on_control(self):
        async def upstream(request):
            self.assertEqual(request.url.host, "control")
            return httpx.Response(422, stream=UpstreamBytes(b"native validation"))

        async def action(client, pool):
            for raw in [b"[]", b'{"messages":null}', b"invalid"]:
                response = await client.post("/v1/chat/completions", content=raw)
                self.assertEqual(response.status_code, 422)
            self.assertEqual((await client.get("/get_server_info")).status_code, 422)
            self.assertEqual(sum(pool.active), 0)

        await self.exercise(upstream, action)

    async def test_connect_failure_and_cancellation_release_exactly_once(self):
        async def upstream(request):
            if request.headers.get("test-cancel"):
                raise asyncio.CancelledError()
            raise httpx.ConnectError("injected connection failure", request=request)

        async def action(client, pool):
            response = await client.post("/v1/chat/completions", json=BODY)
            self.assertEqual(response.status_code, 502)
            self.assertEqual(sum(pool.active), 0)
            with self.assertRaises(asyncio.CancelledError):
                await client.post("/v1/chat/completions", json=BODY, headers={"test-cancel": "1"})
            self.assertEqual(sum(pool.active), 0)

        await self.exercise(upstream, action)

    async def test_metrics_are_one_native_copy_plus_gateway_counters(self):
        async def upstream(request):
            self.assertEqual(request.url.host, "control")
            return httpx.Response(200, content=b"native_metric{rank=\"0\"} 1\n")

        async def action(client, pool):
            response = await client.get("/metrics")
            self.assertEqual(response.text.count("native_metric"), 1)
            self.assertIn("glm_api_cache_router_active_requests", response.text)

        await self.exercise(upstream, action)
