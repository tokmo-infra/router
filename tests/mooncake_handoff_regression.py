"""CPU-only Mooncake PD handoff regression against a built router binary.

Run with --binary /path/to/vllm-router --output /new/result.json.
Fake P2/D8 endpoints validate engine rank, token IDs and measured cache usage.
No model, GPU, external network or production endpoints are used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests', type=int, default=256)
    parser.add_argument('--concurrency', type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), output
    assert args.requests > 0 and args.concurrency > 0
    tokens = list(range(131072))
    cached = 120000
    published, handoffs, errors, ranks = {}, [], [], set()
    lock = threading.Lock()
    responses = {}
    for bad in (False, True):
        responses[bad] = json.dumps({
            'id': 'fake-prefill', 'object': 'chat.completion', 'model': 'handoff-test',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'P'},
                         'finish_reason': 'length'}],
            'prompt_token_ids': [True] if bad else tokens,
            'usage': {'prompt_tokens': len(tokens), 'completion_tokens': 1,
                      'total_tokens': len(tokens) + 1,
                      'prompt_tokens_details': {'cached_tokens': cached}},
        }, separators=(',', ':')).encode()

    class Backend(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def reply(self, body, content_type='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/query':
                body = json.dumps({str(i): {'engine_id': f'engine-{i}'}
                                   for i in range(2)}).encode()
            elif self.path == '/metrics':
                body = '\n'.join(f'vllm:num_requests_running{{engine="{i}"}} 0'
                                 for i in range(self.server.ranks)).encode()
            else:
                body = b'{"status":"healthy","data":[{"id":"handoff-test"}]}'
            self.reply(body)

        def do_POST(self):
            arrived = time.monotonic()
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                request_id = self.headers['X-Request-Id']
                if self.server.role == 'P':
                    assert body['stream'] is False
                    assert body['max_tokens'] == 1
                    assert body['min_tokens'] == 1 and body['ignore_eos'] is True
                    assert body['return_token_ids'] is True
                    assert body['stop'] is None and body['stop_token_ids'] == []
                    assert body['kv_transfer_params']['do_remote_decode'] is True
                    rank = self.headers.get('X-data-parallel-rank', '0')
                    with lock:
                        published[request_id] = (time.monotonic(), rank)
                    self.reply(responses[body['messages'][0]['content'] == 'invalid-ids'])
                    return
                params = body['kv_transfer_params']
                ids = params['prompt_token_ids']
                assert ids == tokens
                assert body['max_tokens'] == 8 and body['min_tokens'] == 2
                assert body['ignore_eos'] is False
                assert body['stop'] == ['caller-stop'] and body['stop_token_ids'] == [271]
                assert params['remote_prefill_cached_tokens'] == cached
                assert params['do_remote_prefill'] is True
                assert params['transfer_id'].startswith('xfer-')
                assert body.get('return_token_ids') is not True
                with lock:
                    start, rank = published[request_id]
                    assert params['remote_engine_id'] == f'engine-{rank}'
                    ranks.add(rank)
                    handoffs.append(arrived - start)
                result = {'id': request_id, 'object': 'chat.completion', 'model': 'handoff-test',
                          'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                                       'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': len(tokens), 'completion_tokens': 1,
                                    'total_tokens': len(tokens) + 1,
                                    'prompt_tokens_details': {'cached_tokens': cached}}}
                self.reply(json.dumps(result).encode())
            except Exception as error:
                with lock:
                    errors.append(repr(error))
                self.reply(json.dumps({'error': repr(error)}).encode(), status=500)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 1024

    servers = []
    for role, n in [('P', 2), ('D', 8)]:
        server = Server(('127.0.0.1', 0), Backend)
        server.role, server.ranks = role, n
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = {**os.environ, 'NO_PROXY': '*', 'no_proxy': '*',
           'VLLM_ROUTER_PD_REUSE_PROMPT_IDS': '1',
           'VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP': '1',
           'TOKIO_WORKER_THREADS': '16'}
    report = {'passed': False, 'scope': 'CPU fake Mooncake P2/D8; no production requests',
              'requests': args.requests, 'concurrency': args.concurrency,
              'input_chars': 262144, 'prompt_token_ids': len(tokens)}
    started = time.monotonic()
    with output.with_suffix('.log').open('x') as log:
        proc = subprocess.Popen([
            args.binary, '--host', '127.0.0.1', '--port', str(port),
            '--vllm-pd-disaggregation', '--kv-connector', 'mooncake',
            '--prefill', f'http://127.0.0.1:{servers[0].server_port}', str(servers[0].server_port),
            '--decode', f'http://127.0.0.1:{servers[1].server_port}',
            '--prefill-policy', 'round_robin', '--decode-policy', 'round_robin',
            '--max-concurrent-requests', str(max(256, args.concurrency)),
            '--disable-retries', '--api-key', 'EMPTY', '--log-level', 'warn',
            '--worker-startup-timeout-secs', '20', '--request-timeout-secs', '90',
        ], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(200):
                assert proc.poll() is None, 'Router exited during startup'
                try:
                    with op.open(f'http://127.0.0.1:{port}/workers', timeout=1) as response:
                        workers = json.load(response)['workers']
                    if len(workers) == 10 and all(w['is_healthy'] for w in workers):
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError('Expected 2 P and 8 D healthy workers')

            def call(content):
                body = {'model': 'handoff-test', 'messages': [{'role': 'user', 'content': content}],
                        'stream': False, 'max_tokens': 8, 'min_tokens': 2,
                        'ignore_eos': False, 'stop': ['caller-stop'], 'stop_token_ids': [271]}
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                             data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json',
                                                      'Authorization': 'Bearer EMPTY'})
                with op.open(req, timeout=95) as response:
                    result = json.load(response)
                assert result['choices'][0]['message']['content'] == 'OK'
                assert result['usage']['prompt_tokens_details']['cached_tokens'] == cached

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(call, (f'case-{i}:' + 'x' * 262144 for i in range(args.requests))))
            assert not errors, errors
            assert len(handoffs) == args.requests
            assert ranks == {'0', '1'}, ranks
            try:
                call('invalid-ids')
            except urllib.error.HTTPError as error:
                assert error.code == 500
            else:
                raise AssertionError('Corrupt token IDs must fail before D dispatch')
            assert len(handoffs) == args.requests
            ordered = sorted(handoffs)
            report.update(passed=True, elapsed_s=time.monotonic() - started,
                          handoff_publish_to_D_mean_s=statistics.fmean(handoffs),
                          handoff_publish_to_D_p99_s=ordered[min(len(ordered)-1, int(len(ordered)*.99))],
                          handoff_publish_to_D_max_s=ordered[-1],
                          engine_ranks=sorted(ranks), invalid_ids_rejected=True,
                          backend_errors=errors,
                          timing_note='Fake P publish start to D HTTP handler entry; includes HTTP transport, not production TTFT')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            for server in servers:
                server.shutdown()
                server.server_close()
            output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
