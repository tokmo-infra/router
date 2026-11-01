import asyncio
import json
import os
import socket
from contextlib import suppress

import httpx
import uvicorn

from api_affinity import ApiPool, RouterConfig, create_app, request_prefixes


def test_affinity_balance_and_release():
    cfg = RouterConfig(
        "http://control",
        ("http://a", "http://b"),
        balance_abs_threshold=0,
        balance_rel_threshold=1,
    )
    pool = ApiPool(cfg)
    key, prefixes = request_prefixes(
        {"model": "m", "messages": [{"role": "user", "content": "abc"}]}
    )
    first, _ = pool.reserve(key, prefixes)
    pool.release(first, prefixes, 200, True)
    again, reason = pool.reserve(key, prefixes)
    assert (again, reason) == (first, "prefix_hit")
    other, reason = pool.reserve(key, prefixes)
    if other == first:
        third, reason = pool.reserve(key, prefixes)
        assert third != first and reason == "load_override"
        pool.release(third, prefixes)
    else:
        assert reason == "load_override"
    pool.release(other, prefixes)
    pool.release(again, prefixes)
    assert pool.active == [0, 0]


def test_live_concurrent_stream_cancel():
    async def run():
        name = f"uds-{os.getpid()}.sock"
        seen, closed = {}, {}
        gate = asyncio.Event()
        payloads = {
            "a": b'{"model":"m", "prompt":"a", "stream":true}',
            "b": b'{"model":"m", "prompt":"b", "stream":true}',
        }
        first = {"a": b'data: {"id":"a"}\r\n\r\n', "b": b'data: {"id":"b"}\r\n\r\n'}
        tail = b"data: [DONE]\n\n"

        async def upstream(reader, writer):
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                lines = header.split(b"\r\n")
                headers = dict(
                    line.split(b":", 1) for line in lines[1:] if b":" in line
                )
                headers = {k.lower(): v.strip() for k, v in headers.items()}
                body = await reader.readexactly(int(headers[b"content-length"]))
                ident = json.loads(body)["prompt"]
                seen[ident] = (body, headers)
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                writer.write(
                    ("%x\r\n" % len(first[ident])).encode() + first[ident] + b"\r\n"
                )
                await writer.drain()
                if ident == "a":
                    assert await asyncio.wait_for(reader.read(), 5) == b""
                    closed.setdefault(ident, asyncio.Event()).set()
                else:
                    await asyncio.wait_for(gate.wait(), 5)
                    writer.write(
                        ("%x\r\n" % len(tail)).encode() + tail + b"\r\n0\r\n\r\n"
                    )
                    await writer.drain()
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

        server = await asyncio.start_unix_server(upstream, name)
        app = create_app(
            RouterConfig(
                "http://127.0.0.1:1", ("http://api.local",), api_sockets=(name,)
            )
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        port = sock.getsockname()[1]
        uv = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(uv.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if uv.started:
                    break
                await asyncio.sleep(0.01)
            assert uv.started
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:

                async def start(ident):
                    req = client.build_request(
                        "POST",
                        f"http://127.0.0.1:{port}/v1/completions?x=1",
                        content=payloads[ident],
                        headers={
                            "x-data-parallel-rank": "7",
                            "content-type": "application/json",
                            "x-test-id": ident,
                        },
                    )
                    return await client.send(req, stream=True)

                a, b = await asyncio.wait_for(asyncio.gather(start("a"), start("b")), 3)
                ia, ib = a.aiter_raw(), b.aiter_raw()
                aa, bb = await asyncio.wait_for(
                    asyncio.gather(ia.__anext__(), ib.__anext__()), 3
                )
                assert aa == first["a"] and bb == first["b"]
                assert not gate.is_set() and app.state.pool.active == [2]
                for ident in ("a", "b"):
                    assert seen[ident][0] == payloads[ident]
                    assert seen[ident][1][b"x-data-parallel-rank"] == b"7"
                    assert seen[ident][1][b"x-test-id"] == ident.encode()
                await a.aclose()
                await asyncio.wait_for(
                    closed.setdefault("a", asyncio.Event()).wait(), 3
                )
                for _ in range(100):
                    if app.state.pool.active == [1]:
                        break
                    await asyncio.sleep(0.01)
                assert app.state.pool.active == [1]
                gate.set()
                rest = b"".join([part async for part in ib])
                assert bb + rest == first["b"] + tail
                await b.aclose()
                for _ in range(100):
                    if app.state.pool.active == [0]:
                        break
                    await asyncio.sleep(0.01)
                assert app.state.pool.active == [0]
        finally:
            gate.set()
            uv.should_exit = True
            await asyncio.wait_for(task, 5)
            server.close()
            await server.wait_closed()
            with suppress(FileNotFoundError):
                os.unlink(name)

    asyncio.run(run())




def test_disconnect_before_headers_does_not_cancel_neighbor():
    from tempfile import TemporaryDirectory

    async def run(path):
        entered = {key: asyncio.Event() for key in ("a", "b")}
        closed, release_b = asyncio.Event(), asyncio.Event()
        tasks = set()

        async def upstream(reader, writer):
            tasks.add(asyncio.current_task())
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                ident = json.loads(await reader.readexactly(length))["prompt"]
                entered[ident].set()
                if ident == "a":
                    assert await asyncio.wait_for(reader.read(), 5) == b""
                    closed.set()
                else:
                    await asyncio.wait_for(release_b.wait(), 5)
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                    await writer.drain()
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
                tasks.discard(asyncio.current_task())

        server = await asyncio.start_unix_server(upstream, path)
        app = create_app(
            RouterConfig(
                "http://127.0.0.1:1", ("http://api.local",), api_sockets=(path,)
            )
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        port = sock.getsockname()[1]
        uv = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(uv.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if uv.started:
                    break
                await asyncio.sleep(0.01)
            assert uv.started

            async def connect(ident):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                body = json.dumps({"prompt": ident}).encode()
                writer.write(
                    b"POST /v1/completions HTTP/1.1\r\nHost: test\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
                await writer.drain()
                return reader, writer

            _, a = await connect("a")
            br, b = await connect("b")
            await asyncio.wait_for(
                asyncio.gather(*(e.wait() for e in entered.values())), 3
            )
            assert app.state.pool.active == [2]
            a.close()
            await a.wait_closed()
            await asyncio.wait_for(closed.wait(), 3)
            for _ in range(200):
                if app.state.pool.active == [1]:
                    break
                await asyncio.sleep(0.01)
            assert app.state.pool.active == [1]
            release_b.set()
            response = await asyncio.wait_for(br.readuntil(b"\r\n\r\n"), 3)
            assert b"200 OK" in response
            assert await asyncio.wait_for(br.readexactly(2), 3) == b"ok"
            b.close()
            await b.wait_closed()
            for _ in range(200):
                if app.state.pool.active == [0]:
                    break
                await asyncio.sleep(0.01)
            assert app.state.pool.active == [0]
        finally:
            release_b.set()
            uv.should_exit = True
            await asyncio.wait_for(task, 5)
            server.close()
            await server.wait_closed()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    with TemporaryDirectory(dir="/tmp", prefix="api-test-") as tmp:
        asyncio.run(run(os.path.join(tmp, "up.sock")))


def test_slow_client_limits_upstream_read_ahead():
    from tempfile import TemporaryDirectory

    async def run(path):
        blocked, resumed, done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        state = {"sent": 0}
        chunk = b"x" * 65536
        max_chunks = 1024

        async def upstream(reader, writer):
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in head.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                await reader.readexactly(length)
                writer.transport.set_write_buffer_limits(high=65536, low=32768)
                writer.get_extra_info("socket").setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, 65536
                )
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                    b"Content-Length: "
                    + str(len(chunk) * max_chunks).encode()
                    + b"\r\n\r\n"
                )
                for _ in range(max_chunks):
                    writer.write(chunk)
                    draining = asyncio.create_task(writer.drain())
                    finished, _ = await asyncio.wait((draining,), timeout=0.5)
                    if not finished:
                        blocked.set()
                        await asyncio.wait_for(draining, 5)
                        resumed.set()
                        # Keep the incomplete upstream response open until the
                        # downstream abort is actually propagated.
                        assert await asyncio.wait_for(reader.read(), 5) == b""
                        break
                    draining.result()
                    state["sent"] += len(chunk)
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
                done.set()

        server = await asyncio.start_unix_server(upstream, path)
        app = create_app(
            RouterConfig(
                "http://127.0.0.1:1", ("http://api.local",), api_sockets=(path,)
            )
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        port = sock.getsockname()[1]
        uv = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(uv.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if uv.started:
                    break
                await asyncio.sleep(0.01)
            assert uv.started
            client = socket.socket()
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
            client.setblocking(False)
            loop = asyncio.get_running_loop()
            await loop.sock_connect(client, ("127.0.0.1", port))
            body = b'{"prompt":"slow"}'
            await loop.sock_sendall(
                client,
                b"POST /v1/completions HTTP/1.1\r\nHost: test\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body,
            )
            await asyncio.wait_for(blocked.wait(), 10)
            assert 0 < state["sent"] < len(chunk) * max_chunks
            sent_at_block = state["sent"]
            await asyncio.sleep(0.3)
            assert state["sent"] == sent_at_block
            assert app.state.pool.active == [1]
            assert b"HTTP/1.1 200" in await asyncio.wait_for(
                loop.sock_recv(client, 4096), 3
            )

            # Resuming downstream consumption must release the same blocked
            # upstream drain, not merely wake a test-side gate.
            async def consume_until_resumed():
                while not resumed.is_set():
                    assert await loop.sock_recv(client, 65536)

            await asyncio.wait_for(consume_until_resumed(), 5)
            client.close()
            await asyncio.wait_for(done.wait(), 3)
            for _ in range(200):
                if app.state.pool.active == [0]:
                    break
                await asyncio.sleep(0.01)
            assert app.state.pool.active == [0]
            print("backpressure upstream bytes before drain timeout:", sent_at_block)
        finally:
            uv.should_exit = True
            await asyncio.wait_for(task, 5)
            server.close()
            await server.wait_closed()

    with TemporaryDirectory(dir="/tmp", prefix="api-test-") as tmp:
        asyncio.run(run(os.path.join(tmp, "up.sock")))
