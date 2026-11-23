"""Loopback Chat endpoint for validating the inline-media admission policy."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from normalize import MAX_INPUT_BYTES, MIB, POLICY, MediaPolicyError, normalize_chat


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(
        base_url=os.environ["K31_ROUTER_URL"], timeout=480, trust_env=False
    ) as client:
        app.state.client = client
        app.state.slot = asyncio.Semaphore(1)
        yield


async def chat(request: Request):
    async with request.app.state.slot:
        limit = 4 * ((MAX_INPUT_BYTES + 2) // 3) + MIB
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > limit:
                return JSONResponse(
                    {"error": {"message": "JSON body exceeds the inline-media limit"}},
                    status_code=413,
                )
            raw.extend(chunk)
        try:
            payload = json.loads(raw)
            del raw
            payload, report = await asyncio.to_thread(normalize_chat, payload)
        except (MediaPolicyError, ValueError, TypeError, AttributeError) as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status_code=400,
            )
    client = request.app.state.client
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "x-request-id")
    }
    try:
        upstream = await client.send(
            client.build_request(
                "POST", "/v1/chat/completions", json=payload, headers=headers
            ),
            stream=True,
        )
    except httpx.RequestError:
        return JSONResponse(
            {"error": {"message": "Chat router unavailable", "type": "upstream_error"}},
            status_code=502,
        )
    metrics = {
        "x-k31-visual-tokens": str(sum(m["visual_tokens"] for m in report)),
        "x-k31-rewritten-items": str(sum(m["changed"] for m in report)),
        "x-k31-normalized-bytes": str(sum(m["output_bytes"] for m in report)),
    }
    print(json.dumps({"policy": POLICY, "media": report}), flush=True)
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=metrics,
        background=BackgroundTask(upstream.aclose),
    )


app = Starlette(
    routes=[Route("/v1/chat/completions", chat, methods=["POST"])], lifespan=lifespan
)
