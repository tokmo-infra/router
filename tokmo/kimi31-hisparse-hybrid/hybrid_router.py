"""One CPU ingress: weighted in-flight routing to local P or the existing PD router.

Run exactly one process so admission and release share the same counters. All
prefills still use the same P engine and its existing prefix cache / Store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
from multidict import CIMultiDict
from yarl import URL

LOG = logging.getLogger("hybrid_router")
INFERENCE_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
ROUTES = ("p_local", "pd")
HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass(frozen=True)
class Config:
    prefill_url: str = "http://11.5.0.92:19580"
    pd_router_url: str = "http://11.5.0.92:19500"
    p_weight: int = 1
    pd_weight: int = 2
    max_concurrent_requests: int = 1024
    request_timeout_secs: float = 7200

    def __post_init__(self) -> None:
        if min(self.p_weight, self.pd_weight, self.max_concurrent_requests) <= 0:
            raise ValueError("Weights and max_concurrent_requests must be positive")
        if self.request_timeout_secs <= 0:
            raise ValueError("request_timeout_secs must be positive")
        for value in (self.prefill_url, self.pd_router_url):
            url = urlsplit(value)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.path not in ("", "/")
                or url.query
                or url.fragment
                or url.username
            ):
                raise ValueError(
                    "Upstreams must be HTTP(S) origins without credentials"
                )


@dataclass
class Inflight:
    config: Config
    active: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROUTES, 0))
    admitted: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROUTES, 0))
    released: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROUTES, 0))
    cancelled: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROUTES, 0))

    def reserve(self) -> str | None:
        """Select + reserve without an await; atomic on this process's event loop."""
        if sum(self.active.values()) >= self.config.max_concurrent_requests:
            return None
        # Compare P/1 with PD/2 using integers. Prefer PD on ties, including idle.
        route = (
            "p_local"
            if self.active["p_local"] * self.config.pd_weight
            < self.active["pd"] * self.config.p_weight
            else "pd"
        )
        self.active[route] += 1
        self.admitted[route] += 1
        return route

    def release(self, route: str) -> None:
        if self.active[route] <= 0:
            raise RuntimeError("An in-flight reservation was released twice")
        self.active[route] -= 1
        self.released[route] += 1

    def snapshot(self) -> dict:
        return {
            "weights": {"p_local": self.config.p_weight, "pd": self.config.pd_weight},
            "inflight": self.active.copy(),
            "admitted": self.admitted.copy(),
            "released": self.released.copy(),
            "cancelled": self.cancelled.copy(),
            "max_concurrent_requests": self.config.max_concurrent_requests,
        }


CONFIG = web.AppKey("config", Config)
STATE = web.AppKey("state", Inflight)
CLIENT = web.AppKey("client", aiohttp.ClientSession)


def forward_headers(headers, *, request: bool = False) -> CIMultiDict:
    excluded = set(HOP_HEADERS)
    for value in headers.getall("Connection", []):
        excluded.update(part.strip().lower() for part in value.split(","))
    if request:
        # aiohttp recalculates framing after sanitizing a local request's JSON.
        excluded.update({"host", "content-length", "expect"})
        # aiohttp's server parser has already decoded supported content encodings.
        if headers.get("Content-Encoding", "").lower() in ("gzip", "deflate", "br"):
            excluded.add("content-encoding")
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in excluded)


async def client_context(app: web.Application):
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(
            total=app[CONFIG].request_timeout_secs, sock_connect=10
        ),
        auto_decompress=False,
        skip_auto_headers={"Accept-Encoding"},
        trust_env=False,
    ) as client:
        app[CLIENT] = client
        yield


async def health(request: web.Request) -> web.Response:
    async def check(url: str) -> bool:
        try:
            async with request.app[CLIENT].get(
                url.rstrip("/") + "/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    config = request.app[CONFIG]
    p_ok, pd_ok = await asyncio.gather(
        check(config.prefill_url), check(config.pd_router_url)
    )
    return web.json_response(
        {"prefill": p_ok, "pd_router": pd_ok}, status=200 if p_ok and pd_ok else 503
    )


async def routing_state(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATE].snapshot())


async def metrics(request: web.Request) -> web.Response:
    snapshot = request.app[STATE].snapshot()
    lines = []
    for name, key, kind in (
        ("hybrid_router_inflight", "inflight", "gauge"),
        ("hybrid_router_admitted_total", "admitted", "counter"),
        ("hybrid_router_released_total", "released", "counter"),
        ("hybrid_router_cancelled_total", "cancelled", "counter"),
    ):
        lines.append(f"# TYPE {name} {kind}")
        for route in ROUTES:
            lines.append(f'{name}{{route="{route}"}} {snapshot[key][route]}')
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def proxy(request: web.Request) -> web.StreamResponse:
    config, state = request.app[CONFIG], request.app[STATE]
    route = None
    downstream = None
    try:
        body = await request.read()
        inference = request.method == "POST" and request.path in INFERENCE_PATHS
        if inference:
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                return web.json_response({"error": "Invalid JSON request"}, status=400)
            if not isinstance(payload, dict):
                return web.json_response(
                    {"error": "Expected a JSON object"}, status=400
                )
            route = state.reserve()
            if route is None:
                return web.json_response(
                    {"error": "Router concurrency limit"}, status=503
                )
            if route == "p_local":
                # Preserve max_tokens, stop, stream, and all sampling parameters.
                # Local generation must not trigger Mooncake's P truncation/export.
                payload.pop("kv_transfer_params", None)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        origin = config.prefill_url if route == "p_local" else config.pd_router_url
        url = origin.rstrip("/") + request.raw_path
        async with request.app[CLIENT].request(
            request.method,
            URL(url, encoded=True),
            data=body if body else None,
            headers=forward_headers(request.headers, request=True),
            allow_redirects=False,
        ) as upstream:
            headers = forward_headers(upstream.headers)
            if route is not None:
                headers["X-Hybrid-Route"] = route
            downstream = web.StreamResponse(status=upstream.status, headers=headers)
            await downstream.prepare(request)
            # Forward SSE immediately, retaining the reservation until EOF/drop.
            async for chunk in upstream.content.iter_any():
                await downstream.write(chunk)
            await downstream.write_eof()
            return downstream
    except asyncio.CancelledError:
        if route is not None:
            state.cancelled[route] += 1
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as error:
        LOG.warning(
            "Upstream forwarding failed route=%s type=%s", route, type(error).__name__
        )
        if downstream is not None and downstream.prepared:
            # A partial generation cannot be retried or turned into a new response.
            if request.transport is not None:
                request.transport.close()
            return downstream
        status = 504 if isinstance(error, asyncio.TimeoutError) else 502
        return web.json_response({"error": "Upstream request failed"}, status=status)
    finally:
        if route is not None:
            state.release(route)


def create_app(config: Config) -> web.Application:
    app = web.Application(client_max_size=0)
    app[CONFIG], app[STATE] = config, Inflight(config)
    app.cleanup_ctx.append(client_context)
    app.router.add_get("/health", health)
    app.router.add_get("/routing_state", routing_state)
    app.router.add_get("/metrics", metrics)
    # Stateful Responses and other APIs retain their existing PD routing.
    app.router.add_route("*", "/{path:.*}", proxy)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19510)
    parser.add_argument("--prefill-url", default=Config.prefill_url)
    parser.add_argument("--pd-router-url", default=Config.pd_router_url)
    parser.add_argument("--p-weight", type=int, default=1)
    parser.add_argument("--pd-weight", type=int, default=2)
    parser.add_argument("--max-concurrent-requests", type=int, default=1024)
    parser.add_argument("--request-timeout-secs", type=float, default=7200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = Config(
        prefill_url=args.prefill_url,
        pd_router_url=args.pd_router_url,
        p_weight=args.p_weight,
        pd_weight=args.pd_weight,
        max_concurrent_requests=args.max_concurrent_requests,
        request_timeout_secs=args.request_timeout_secs,
    )
    # Required: cancel upstream and release admission on disconnect, even while
    # awaiting the first response byte. Do not run multiple worker processes.
    web.run_app(
        create_app(config),
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        shutdown_timeout=15,
    )


if __name__ == "__main__":
    main()
