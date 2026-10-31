"""Cache-aware API selection after the outer router selects a GPU DP rank.

This proxy preserves OpenAI bodies, DP headers and streaming bytes. Its private
API listeners do not create extra engines or alter the outer worker registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

LOGGER = logging.getLogger(__name__)
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}


def request_prefixes(body: dict) -> tuple[bytes, list[tuple[bytes, int]]]:
    """Build namespace-isolated advisory prefixes; never produce token IDs."""
    if not isinstance(body, dict):
        raise ValueError("OpenAI requests must be objects")
    namespace = json.dumps(
        [
            body.get("model"),
            body.get("cache_salt"),
            body.get("tools"),
            body.get("chat_template_kwargs"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    state = hashlib.sha256(namespace)
    affinity = state.digest()
    prefixes = []
    size = 0
    messages = body.get("messages", [body.get("prompt", "")])
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    for message in messages:
        raw = json.dumps(
            message, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        for offset in range(0, len(raw), 2048):
            chunk = raw[offset : offset + 2048]
            state.update(chunk)
            size += len(chunk)
            prefixes.append((state.copy().digest(), size))
        state.update(b"\x00message-boundary\x00")
    if body.get("cache_salt") is None:
        # Missing salts must not bind every conversation sharing a system role
        # to one API. Use the first user message as the stable cold placement.
        first_user = next(
            (x for x in messages if isinstance(x, dict) and x.get("role") == "user"),
            messages[0] if messages else "",
        )
        affinity = hashlib.sha256(
            namespace + json.dumps(first_user, sort_keys=True).encode()
        ).digest()
    if len(prefixes) > 256:
        step = max(1, len(prefixes) // 255)
        prefixes = prefixes[::step][:255] + [prefixes[-1]]
    return affinity, prefixes


@dataclass(frozen=True)
class RouterConfig:
    control_url: str
    api_urls: tuple[str, ...]
    prefix_limit: int = 65536
    prefix_ttl_seconds: float = 1800
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 8
    balance_rel_threshold: float = 1.5
    request_timeout_seconds: float = 1800
    shared_cache_path: str = ""
    api_sockets: tuple[str, ...] = ()


class ApiPool:
    def __init__(self, config: RouterConfig):
        if not config.api_urls or config.prefix_limit <= 0:
            raise ValueError(
                "At least one API and a positive prefix limit are required"
            )
        self.config = config
        self.active = [0] * len(config.api_urls)
        self.unavailable_until = [0.0] * len(config.api_urls)
        self.prefixes: OrderedDict[bytes, dict[int, float]] = OrderedDict()
        self.selections = Counter()
        self.responses = Counter()
        self.connection_errors = Counter()

    def reserve(self, affinity, prefixes):
        now = time.monotonic()
        available = [i for i, end in enumerate(self.unavailable_until) if end <= now]
        if not available:
            raise RuntimeError("No API frontend is currently available")
        scores = {i: 0 for i in available}
        for digest, size in prefixes:
            observed = self.prefixes.get(digest, {})
            for index, at in observed.items():
                if index in scores and now - at <= self.config.prefix_ttl_seconds:
                    scores[index] = max(scores[index], size)
        cold = available[int.from_bytes(affinity[:8], "little") % len(available)]
        preferred = max(available, key=lambda i: (scores[i], i == cold))
        total_size = prefixes[-1][1] if prefixes else 1
        cached = scores[preferred] / total_size >= self.config.cache_threshold
        if not cached:
            preferred = cold
        least = min(available, key=lambda i: (self.active[i], i != cold))
        overloaded = (
            self.active[preferred] - self.active[least]
            > self.config.balance_abs_threshold
            and self.active[preferred]
            > max(1, self.active[least]) * self.config.balance_rel_threshold
        )
        index = least if overloaded else preferred
        reason = (
            "load_override" if overloaded else "prefix_hit" if cached else "cold_hash"
        )
        self.active[index] += 1
        self.selections[index, reason] += 1
        return index, reason

    def release(self, index, prefixes, status=None, completed=False):
        if self.active[index] <= 0:
            raise RuntimeError("Unbalanced API request accounting")
        self.active[index] -= 1
        if status is not None:
            self.responses[index, status] += 1
        if completed and status is not None and 200 <= status < 300:
            now = time.monotonic()
            for digest, _ in prefixes:
                observed = self.prefixes.setdefault(digest, {})
                observed[index] = now
                self.prefixes.move_to_end(digest)
            while len(self.prefixes) > self.config.prefix_limit:
                self.prefixes.popitem(last=False)

    def metrics(self):
        lines = []
        for index, value in enumerate(self.active):
            lines.append(
                f'glm_api_cache_router_active_requests{{api="{index}"}} {value}'
            )
        for (index, reason), value in sorted(self.selections.items()):
            lines.append(
                f'glm_api_cache_router_selected_total{{api="{index}",'
                f'reason="{reason}"}} {value}'
            )
        for (index, status), value in sorted(self.responses.items()):
            lines.append(
                f'glm_api_cache_router_responses_total{{api="{index}",'
                f'status="{status}"}} {value}'
            )
        for index, value in sorted(self.connection_errors.items()):
            lines.append(
                f'glm_api_cache_router_connection_errors_total{{api="{index}"}} {value}'
            )
        lines.append(f"glm_api_cache_router_prefix_entries {len(self.prefixes)}")
        return "\n".join(lines) + "\n"


def forwarded_headers(headers):
    connection_tokens = {
        x.strip().lower() for x in headers.get("connection", "").split(",") if x.strip()
    }
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_HEADERS | connection_tokens
    }


def shared_cache_metrics(path):
    """Read one host pool once; API-local metrics must not multiply its size."""
    if not path:
        return ""
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.01) as db:
            db.row_factory = sqlite3.Row
            rows = dict(
                db.execute("SELECT * FROM metadata WHERE singleton=1").fetchone()
            )
        fields = {
            "bytes": "bytes",
            "budget": "max_bytes",
            "evictions": "evictions_total",
        }
        return "".join(
            f"glm_tokenizer_shared_pool_{name} {int(rows[field])}\n"
            for field, name in fields.items()
        )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        return "glm_tokenizer_shared_pool_metrics_available 0\n"


def create_app(config: RouterConfig, transport=None):
    pool = ApiPool(config)
    if config.api_sockets and len(config.api_sockets) != len(config.api_urls):
        raise ValueError("Each API must have exactly one private Unix socket")
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(config.request_timeout_seconds, connect=10),
        limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
        trust_env=False,
    )
    api_clients = [
        httpx.AsyncClient(
            transport=transport
            or httpx.AsyncHTTPTransport(
                uds=path,
                retries=0,
                limits=httpx.Limits(
                    max_connections=2048, max_keepalive_connections=128
                ),
            ),
            timeout=httpx.Timeout(config.request_timeout_seconds, connect=10),
            trust_env=False,
        )
        for path in config.api_sockets
    ]

    @asynccontextmanager
    async def lifespan(app):
        yield
        await client.aclose()
        for api_client in api_clients:
            await api_client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.pool = pool
    app.state.client = client
    app.state.api_clients = api_clients

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request):
        raw = await request.body()
        index = None
        prefixes = []
        reason = "control"
        target = config.control_url
        if request.method == "POST" and path in {
            "v1/chat/completions",
            "v1/completions",
            "tokenize",
            "v1/tokenize",
        }:
            try:
                body = json.loads(raw)
                affinity, prefixes = request_prefixes(body)
                index, reason = pool.reserve(affinity, prefixes)
                target = config.api_urls[index]
            except (ValueError, TypeError, RuntimeError):
                # Let native validation handle malformed OpenAI requests. A
                # temporarily unavailable API pool uses the existing listener.
                index = None
                target = config.control_url
        suffix = f"/{path}" + (f"?{request.url.query}" if request.url.query else "")
        upstream_client = (
            api_clients[index] if index is not None and api_clients else client
        )
        try:
            upstream = await upstream_client.send(
                upstream_client.build_request(
                    request.method,
                    target + suffix,
                    headers=forwarded_headers(request.headers),
                    content=raw,
                ),
                stream=True,
            )
        except BaseException as error:
            if index is not None:
                pool.release(index, prefixes)
                if isinstance(error, httpx.RequestError):
                    pool.unavailable_until[index] = time.monotonic() + 5
                    pool.connection_errors[index] += 1
            if not isinstance(error, httpx.RequestError):
                raise
            LOGGER.exception("API forwarding failed: target=%s", target)
            return JSONResponse(
                {"error": "API frontend connection failed"}, status_code=502
            )
        headers = forwarded_headers(upstream.headers)
        if path == "metrics" and upstream.status_code == 200:
            data = await upstream.aread()
            await upstream.aclose()
            return Response(
                data
                + pool.metrics().encode()
                + shared_cache_metrics(config.shared_cache_path).encode(),
                media_type="text/plain; version=0.0.4",
            )

        released = False

        async def cleanup(completed=False):
            nonlocal released
            if released:
                return
            released = True
            try:
                await upstream.aclose()
            finally:
                if index is not None:
                    pool.release(index, prefixes, upstream.status_code, completed)

        async def stream():
            completed = False
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
                completed = True
            finally:
                await cleanup(completed)

        headers["x-glm-api-cache-route"] = reason
        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=headers,
            background=BackgroundTask(cleanup),
        )

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--api-url", action="append")
    parser.add_argument("--api-socket", action="append")
    parser.add_argument("--shared-cache-path", default="")
    args = parser.parse_args()
    if bool(args.api_url) == bool(args.api_socket):
        parser.error("Choose either API TCP URLs or private Unix sockets")
    urls = args.api_url or [
        f"http://api-{i}.local" for i in range(len(args.api_socket))
    ]
    config = RouterConfig(
        args.control_url.rstrip("/"),
        tuple(x.rstrip("/") for x in urls),
        shared_cache_path=args.shared_cache_path,
        api_sockets=tuple(args.api_socket or []),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
