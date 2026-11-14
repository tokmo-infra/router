"""Real loopback HTTP/SSE validation; no vLLM process or GPU is required."""

import asyncio
import contextlib
import gzip
import json
import sys
import unittest
from pathlib import Path

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
from yarl import URL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_router import STATE, Config, create_app  # noqa: E402


class Backend:
    def __init__(self, name):
        self.name = name
        self.requests = []
        self.unblock = asyncio.Event()
        self.started = asyncio.Event()
        self.active = 0
        self.healthy = True
        self.app = web.Application(client_max_size=0)
        self.app.router.add_route("*", "/{path:.*}", self.handle)

    async def handle(self, request):
        if request.path == "/health":
            return web.Response(status=200 if self.healthy else 503)
        if request.path == "/compressed":
            return web.Response(
                body=gzip.compress(b"original compressed response"),
                headers={"Content-Encoding": "gzip"},
            )
        if request.path == "/redirect":
            return web.Response(status=307, headers={"Location": "/v1/models"})
        raw = await request.read()
        payload = json.loads(raw) if raw else None
        self.requests.append(
            {
                "payload": payload,
                "raw_path": request.raw_path,
                "authorization": request.headers.get("Authorization"),
                "content_encoding": request.headers.get("Content-Encoding"),
            }
        )
        self.active += 1
        self.started.set()
        try:
            if request.headers.get("X-Test-Before-Headers"):
                await self.unblock.wait()
            status = int(request.headers.get("X-Test-Status", "200"))
            if payload and payload.get("stream") and status == 200:
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(f'data: {{"backend":"{self.name}"}}\n\n'.encode())
                await self.unblock.wait()
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            return web.json_response(
                {"backend": self.name, "payload": payload}, status=status
            )
        finally:
            self.active -= 1


class HybridRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.p, self.pd = Backend("p"), Backend("pd")
        self.servers = []
        self.responses = []
        p_url = await self.serve(self.p.app)
        pd_url = await self.serve(self.pd.app)
        self.origins = {"prefill_url": p_url, "pd_router_url": pd_url}
        self.app = create_app(Config(**self.origins))
        self.url = await self.serve(self.app)
        self.client = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0), auto_decompress=False
        )

    async def serve(self, app):
        server = TestServer(app, handler_cancellation=True)
        await server.start_server()
        self.servers.append(server)
        return str(server.make_url("/")).rstrip("/")

    async def asyncTearDown(self):
        for response in self.responses:
            response.close()
        await self.client.close()
        for server in reversed(self.servers):
            await server.close()

    async def eventually(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(0.01)

    async def hold(self, **kwargs):
        response = await self.client.post(
            self.url + "/v1/chat/completions",
            json={"model": "kimi", "messages": [], "stream": True},
            **kwargs,
        )
        self.responses.append(response)
        return response

    async def test_c176_has_59_local_and_117_pd_reservations(self):
        responses = await asyncio.gather(*(self.hold() for _ in range(176)))
        self.assertEqual(self.app[STATE].active, {"p_local": 59, "pd": 117})
        self.assertEqual(
            sum(r.headers["X-Hybrid-Route"] == "p_local" for r in responses), 59
        )
        self.p.unblock.set()
        self.pd.unblock.set()
        bodies = await asyncio.gather(*(response.read() for response in responses))
        self.assertTrue(all(body.endswith(b"data: [DONE]\n\n") for body in bodies))
        await self.eventually(lambda: sum(self.app[STATE].active.values()) == 0)

    async def test_c288_custom_weights_have_103_local_and_185_pd(self):
        app = create_app(Config(**self.origins, p_weight=5, pd_weight=9))
        self.url = await self.serve(app)
        responses = await asyncio.gather(*(self.hold() for _ in range(288)))
        self.assertEqual(app[STATE].active, {"p_local": 103, "pd": 185})
        self.assertEqual(
            sum(r.headers["X-Hybrid-Route"] == "p_local" for r in responses), 103
        )
        self.p.unblock.set()
        self.pd.unblock.set()
        await asyncio.gather(*(r.read() for r in responses))
        await self.eventually(lambda: sum(app[STATE].active.values()) == 0)

    async def test_completions_change_admission_instead_of_round_robin(self):
        responses = await asyncio.gather(*(self.hold() for _ in range(18)))
        self.assertEqual(self.app[STATE].active, {"p_local": 6, "pd": 12})
        for response in responses:
            if response.headers["X-Hybrid-Route"] == "p_local":
                response.close()
        await self.eventually(lambda: self.app[STATE].active["p_local"] == 0)
        replacements = await asyncio.gather(*(self.hold() for _ in range(6)))
        self.assertTrue(
            all(r.headers["X-Hybrid-Route"] == "p_local" for r in replacements)
        )
        self.assertEqual(self.app[STATE].active, {"p_local": 6, "pd": 12})

    async def test_full_local_request_and_original_pd_request(self):
        payload = {
            "model": "kimi",
            "messages": [{"role": "user", "content": "长历史"}],
            "max_tokens": 131072,
            "max_completion_tokens": 131072,
            "min_tokens": 20,
            "stop": ["END"],
            "stream": False,
            "temperature": 0.7,
            "kv_transfer_params": {"do_remote_decode": True},
        }
        # At idle prefer PD. This branch must keep its original request intact.
        async with self.client.post(
            self.url + "/v1/chat/completions", json=payload
        ) as r:
            self.assertEqual((await r.json())["payload"], payload)
            self.assertEqual(r.headers["X-Hybrid-Route"], "pd")
        await self.hold()  # Keep one PD request active so the next request selects P.
        async with self.client.post(
            URL(self.url + "/v1/chat/completions?trace=a%2Fb", encoded=True),
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        ) as r:
            local = (await r.json())["payload"]
            self.assertEqual(r.headers["X-Hybrid-Route"], "p_local")
        self.assertEqual(
            local, {k: v for k, v in payload.items() if k != "kv_transfer_params"}
        )
        self.assertEqual(self.p.requests[-1]["authorization"], "Bearer test-key")
        self.assertEqual(
            self.p.requests[-1]["raw_path"], "/v1/chat/completions?trace=a%2Fb"
        )

    async def test_large_prompt_and_completion_endpoint_have_no_added_length_cap(self):
        payload = {
            "model": "kimi",
            "prompt": "x" * (2 * 1024 * 1024),
            "max_tokens": 65536,
        }
        async with self.client.post(self.url + "/v1/completions", json=payload) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual((await r.json())["payload"], payload)

    async def test_sse_is_forwarded_before_backend_finishes(self):
        response = await self.hold()
        async with asyncio.timeout(2):
            self.assertIn(b'"backend":"pd"', await response.content.readuntil(b"\n\n"))
        self.assertEqual(self.app[STATE].active["pd"], 1)
        response.close()
        await self.eventually(lambda: self.app[STATE].active["pd"] == 0)
        await self.eventually(lambda: self.pd.active == 0)

    async def test_cancel_before_first_header_releases_upstream_and_reservation(self):
        task = asyncio.create_task(
            self.client.post(
                self.url + "/v1/chat/completions",
                json={"messages": [], "stream": True},
                headers={"X-Test-Before-Headers": "1"},
            )
        )
        await asyncio.wait_for(self.pd.started.wait(), timeout=3)
        self.assertEqual(self.app[STATE].active["pd"], 1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await self.eventually(lambda: self.app[STATE].active["pd"] == 0)
        await self.eventually(lambda: self.pd.active == 0)
        self.assertEqual(self.app[STATE].cancelled["pd"], 1)

    async def test_errors_pass_through_without_retry(self):
        async with self.client.post(
            self.url + "/v1/chat/completions",
            json={"messages": []},
            headers={"X-Test-Status": "503"},
        ) as response:
            self.assertEqual(response.status, 503)
            self.assertEqual((await response.json())["backend"], "pd")
        self.assertEqual(len(self.pd.requests), 1)
        self.assertEqual(len(self.p.requests), 0)
        self.assertEqual(sum(self.app[STATE].active.values()), 0)

    async def test_timeout_before_headers_returns_504_and_releases(self):
        app = create_app(Config(**self.origins, request_timeout_secs=0.05))
        url = await self.serve(app)
        async with self.client.post(
            url + "/v1/chat/completions",
            json={"messages": []},
            headers={"X-Test-Before-Headers": "1"},
        ) as response:
            self.assertEqual(response.status, 504)
        self.assertEqual(sum(app[STATE].active.values()), 0)
        await self.eventually(lambda: self.pd.active == 0)

    async def test_existing_global_concurrency_limit(self):
        app = create_app(Config(**self.origins, max_concurrent_requests=1))
        url = await self.serve(app)
        held = await self.client.post(
            url + "/v1/chat/completions", json={"messages": [], "stream": True}
        )
        self.responses.append(held)
        async with self.client.post(url + "/v1/chat/completions", json={}) as response:
            self.assertEqual(response.status, 503)
        self.assertEqual(len(self.pd.requests), 1)
        self.assertEqual(len(self.p.requests), 0)

    async def test_health_state_metrics_and_other_apis(self):
        async with self.client.get(self.url + "/health") as response:
            self.assertEqual(response.status, 200)
        self.p.healthy = False
        async with self.client.get(self.url + "/health") as response:
            self.assertEqual(response.status, 503)
        payload = {"previous_response_id": "response-1", "input": "续写"}
        async with self.client.post(
            self.url + "/v1/responses", json=payload
        ) as response:
            self.assertEqual((await response.json())["payload"], payload)
            self.assertNotIn("X-Hybrid-Route", response.headers)
        self.assertEqual(sum(self.app[STATE].admitted.values()), 0)
        await self.hold()
        async with self.client.get(self.url + "/routing_state") as response:
            self.assertEqual(
                (await response.json())["inflight"], {"p_local": 0, "pd": 1}
            )
        async with self.client.get(self.url + "/metrics") as response:
            self.assertIn('hybrid_router_inflight{route="pd"} 1', await response.text())

    async def test_compressed_bodies_and_redirect_are_not_corrupted(self):
        async with self.client.get(self.url + "/compressed") as response:
            self.assertEqual(response.headers["Content-Encoding"], "gzip")
            self.assertEqual(
                gzip.decompress(await response.read()), b"original compressed response"
            )
        payload = {"messages": [], "max_tokens": 12345}
        async with self.client.post(
            self.url + "/v1/chat/completions",
            data=gzip.compress(json.dumps(payload).encode()),
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        ) as response:
            self.assertEqual((await response.json())["payload"], payload)
        self.assertIsNone(self.pd.requests[-1]["content_encoding"])
        async with self.client.get(
            self.url + "/redirect", allow_redirects=False
        ) as response:
            self.assertEqual(response.status, 307)
            self.assertEqual(response.headers["Location"], "/v1/models")

    async def test_invalid_body_does_not_take_a_reservation(self):
        for body in (b"bad-json", b"[]"):
            async with self.client.post(
                self.url + "/v1/chat/completions", data=body
            ) as response:
                self.assertEqual(response.status, 400)
        self.assertEqual(sum(self.app[STATE].admitted.values()), 0)


if __name__ == "__main__":
    unittest.main()
