"""CPU contract test: 8 EP4 P + 4 EP8 D; checks routing, not GPU performance."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), output
    lock = threading.Lock()
    calls = {'P': [], 'D': []}
    barriers = {}
    errors = []
    published = {}

    class Backend(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def reply(self, result, status=200):
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/metrics':
                body = '\n'.join(f'vllm:num_requests_running{{engine="{r}"}} 0'
                                 for r in range(self.server.ranks)).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/query':
                self.reply({str(r): {'engine_id': f'engine-{self.server.instance}-{r}'}
                            for r in range(self.server.ranks)})
            else:
                self.reply({'status': 'healthy', 'data': [{'id': 'low-c'}]})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                rank = int(self.headers['X-data-parallel-rank'])
                with lock:
                    calls[self.server.role].append((self.server.instance, rank))
                    barrier = barriers[self.server.role]
                barrier.wait(timeout=30)
                result = {'id': 'contract', 'object': 'chat.completion', 'model': 'low-c',
                          'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                                       'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': 4, 'completion_tokens': 1, 'total_tokens': 5,
                                    'prompt_tokens_details': {'cached_tokens': 0}}}
                if self.server.role == 'P':
                    assert body['return_token_ids'] is True
                    with lock:
                        published[self.headers['X-Request-Id']] = (self.server.instance, rank)
                    result['prompt_token_ids'] = [3, 7, 9, 11]
                else:
                    params = body['kv_transfer_params']
                    assert params['prompt_token_ids'] == [3, 7, 9, 11]
                    with lock:
                        source, source_rank = published[self.headers['X-Request-Id']]
                    assert params['remote_engine_id'] == f'engine-{source}-{source_rank}'
                    assert params['remote_prefill_cached_tokens'] == 0
                self.reply(result)
            except Exception as error:
                with lock:
                    errors.append(repr(error))
                self.reply({'error': repr(error)}, 500)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 256

    servers = []
    for role, instances, ranks in [('P', 8, 4), ('D', 4, 8)]:
        for instance in range(instances):
            server = Server(('127.0.0.1', 0), Backend)
            server.role, server.instance, server.ranks = role, instance, ranks
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)

    def free_port():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]

    port = free_port()
    command = [args.binary, '--host', '127.0.0.1', '--port', str(port),
               '--prometheus-host', '127.0.0.1', '--prometheus-port', str(free_port()),
               '--vllm-pd-disaggregation', '--kv-connector', 'mooncake',
               '--prefill-policy', 'cache_aware', '--decode-policy', 'round_robin',
               '--max-concurrent-requests', '4096', '--disable-retries', '--api-key', 'EMPTY',
               '--log-level', 'warn', '--worker-startup-timeout-secs', '20',
               '--request-timeout-secs', '60']
    for server in servers:
        url = f'http://127.0.0.1:{server.server_port}'
        command += ['--prefill', url, str(server.server_port)] if server.role == 'P' else ['--decode', url]
    env = {**os.environ, 'NO_PROXY': '*', 'no_proxy': '*',
           'VLLM_ROUTER_PD_REUSE_PROMPT_IDS': '1',
           'VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP': '1'}
    report = {'passed': False, 'scope': 'CPU fake backends; routing contract only', 'batches': []}
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with output.with_suffix('.log').open('x') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                assert proc.poll() is None, 'Router exited'
                try:
                    with op.open(f'http://127.0.0.1:{port}/workers', timeout=1) as response:
                        workers = json.load(response)['workers']
                    if len(workers) == 64 and all(w['is_healthy'] for w in workers):
                        break
                except OSError:
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError('64 workers not ready')

            def call(_):
                body = {'model': 'low-c', 'messages': [{'role': 'user', 'content': '共同前缀 ' * 20000}],
                        'stream': False, 'max_tokens': 8}
                request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                                data=json.dumps(body).encode(),
                                                headers={'Content-Type': 'application/json',
                                                         'Authorization': 'Bearer EMPTY'})
                with op.open(request, timeout=65) as response:
                    assert json.load(response)['choices'][0]['message']['content'] == 'OK'

            for concurrency in [8, 16, 32]:
                for wave in range(3):
                    with lock:
                        calls = {'P': [], 'D': []}
                        barriers = {phase: threading.Barrier(concurrency) for phase in ['P', 'D']}
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        list(pool.map(call, range(concurrency)))
                    assert not errors, errors
                    p_instances = Counter(instance for instance, _ in calls['P'])
                    p_hosts = Counter(instance // 2 for instance, _ in calls['P'])
                    d_hosts = Counter(instance for instance, _ in calls['D'])
                    assert len(calls['P']) == len(calls['D']) == concurrency
                    assert len(set(calls['P'])) == len(set(calls['D'])) == concurrency
                    assert dict(p_hosts) == {i: concurrency // 4 for i in range(4)}, p_hosts
                    assert dict(p_instances) == {i: concurrency // 8 for i in range(8)}, p_instances
                    assert dict(d_hosts) == {i: concurrency // 4 for i in range(4)}, d_hosts
                    report['batches'].append({'concurrency': concurrency, 'wave': wave,
                        'P_requests_per_instance': dict(p_instances), 'P_requests_per_host': dict(p_hosts),
                        'D_requests_per_host': dict(d_hosts), 'P_unique_ranks': len(set(calls['P'])),
                        'D_unique_ranks': len(set(calls['D']))})
            report['passed'] = True
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
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
