"""Check a real Router using CPU-only, loopback fake P2/D8 backends.

Adapted from 08d7195149fda3fc0ef7efbdb2c92dd787fcf60e. Run separately with
--mode default/on/off; --env-value also checks values other than the exact '1'.
Requires a locally built Router; Python unit tests do NOT certify that binary.
Only direct-URL NIXL read handoff is covered, not discovery, push connectors,
real tokenizer reuse, GPU correctness, or backend CPU savings.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request


ENV_NAME = 'VLLM_ROUTER_PD_REUSE_PROMPT_IDS'
CHAT = '/v1/chat/completions'
METRIC = 'vllm_router_pd_prompt_ids_reused_total'
KV = {
    'do_remote_prefill': True, 'do_remote_decode': False,
    'remote_engine_id': 'original-engine', 'remote_block_ids': [11, 13],
    'block_ids': [17], 'remote_host': '127.0.0.1', 'remote_port': 12345,
    'contract_extra': {'nested': ['preserve', 42, None]},
}
INPUT_KEYS = {CHAT: 'messages', '/v1/completions': 'prompt',
              '/v1/responses': 'input', '/generate': 'text',
              '/inference/v1/generate': 'token_ids'}


def require(condition, message):
    # Do not silently disable contract assertions under python -O.
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, label):
    # Python equality alone considers True == 1 (also inside token-ID lists).
    require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
            f'{label}: expected {expected!r}, got {actual!r}')


def router_environment(mode='default', env_value=None, environ=None):
    env = dict(os.environ if environ is None else environ)
    env.update(NO_PROXY='*', no_proxy='*')
    # Do not let inherited production auth validation URLs contact remote services.
    env.pop('API_KEY_VALIDATION_URLS', None)
    env.pop(ENV_NAME, None)
    value = env_value if env_value is not None else {
        'default': None, 'on': '1', 'off': '0'}[mode]
    if value is not None:
        env[ENV_NAME] = value
    return env, value is None or value == '1'


def scenarios():
    def chat(name):
        return {
            'name': name, 'path': CHAT, 'ids': [3, 7, 9], 'kv': copy.deepcopy(KV),
            'reject_when_on': False,
            'request': {
                'model': 'contract', 'messages': [{'role': 'user', 'content': name}],
                'stream': False, 'max_tokens': 8, 'max_completion_tokens': 12,
                'min_tokens': 2, 'temperature': 0.25, 'top_p': 0.75, 'seed': 42,
                'stop': ['STOP'], 'n': 1, 'logprobs': True, 'top_logprobs': 2,
                'response_format': {'type': 'json_object'}, 'echo': False,
                'skip_special_tokens': False, 'ignore_eos': True,
            },
        }

    cases = [chat('correct-handoff'), chat('streaming'), chat('u32-boundaries'),
             chat('client-return-ids-true'), chat('client-return-ids-false')]
    cases[1]['request'].update(
        stream=True, stream_options={'include_usage': True})
    cases[2]['ids'] = [0, 4294967295]
    cases[3]['request']['return_token_ids'] = True
    cases[4]['request']['return_token_ids'] = False
    for name, ids in [
        ('bool-true', [True]), ('bool-false', [False]), ('negative', [-1]),
        ('overflow-u32', [4294967296]), ('empty', []), ('null', None),
        ('string-id', ['7']), ('float-id', [7.0]), ('scalar', 7),
        ('object', {'id': 7}), ('mixed-bool', [3, True]),
    ]:
        case = chat(name)
        case.update(ids=ids, reject_when_on=True)
        cases.append(case)
    case = chat('missing-ids')
    del case['ids']
    case['reject_when_on'] = True
    cases.append(case)
    case = chat('streaming-bad-ids')
    case.update(ids=[True], reject_when_on=True)
    case['request'].update(stream=True, stream_options={'include_usage': True})
    cases.append(case)
    for name, kv in [('missing-kv', None), ('null-kv', None), ('array-kv', [])]:
        case = chat(name)
        case.update(kv=kv, reject_when_on=True)
        if name == 'missing-kv':
            del case['kv']
        cases.append(case)
    for path, request in [
        ('/v1/completions', {'prompt': 'non-chat-completion', 'max_tokens': 8}),
        ('/v1/responses', {'input': 'non-chat-response', 'max_output_tokens': 8}),
        ('/generate', {'text': 'non-chat-generate', 'sampling_params': {'max_new_tokens': 8}}),
        ('/inference/v1/generate', {'token_ids': [3, 7, 9],
                                  'sampling_params': {'max_tokens': 8, 'min_tokens': 2}}),
    ]:
        cases.append({'name': path, 'path': path, 'kv': copy.deepcopy(KV),
                      'reject_when_on': False,
                      'request': {'model': 'contract', 'stream': False, **request}})
        if path == '/generate':
            del cases[-1]['request']['model']  # GenerateRequest has no model field.
    # Non-chat P responses deliberately omit IDs: these routes must still dispatch D.
    return cases


def check_prefill(case, body, reuse_ids):
    original = case['request']
    enabled = reuse_ids and case['path'] == CHAT
    if enabled or 'return_token_ids' in original:
        equal(body.get('return_token_ids'), True if enabled else original['return_token_ids'],
              'P return_token_ids')
    else:
        require('return_token_ids' not in body, 'P unexpectedly forced return_token_ids')
    require(body.get('stream') is False, 'P must be non-streaming')
    require('stream_options' not in body, 'P must remove stream_options')
    if case['path'] == '/inference/v1/generate':
        equal(body['sampling_params']['max_tokens'], 1, 'P sampling max_tokens')
        equal(body['sampling_params']['min_tokens'], 1, 'P sampling min_tokens')
    elif case['path'] == '/v1/responses':
        equal(body['max_output_tokens'], 1, 'P max_output_tokens')
    else:
        equal(body['max_tokens'], 1, 'P max_tokens')
        for field in ('max_completion_tokens', 'min_tokens'):
            if field in original:
                equal(body[field], 1, f'P {field}')


def check_decode(case, body, reuse_ids):
    for field, value in case['request'].items():
        require(field in body, f'D removed client field {field}')
        equal(body[field], value, f'D client field {field}')
    require('prompt_token_ids' not in body, 'IDs must be in KV metadata, not D top level')
    if 'return_token_ids' not in case['request']:
        require('return_token_ids' not in body, 'D unexpectedly forced return_token_ids')
    if 'kv' not in case:
        require('kv_transfer_params' not in body, 'OFF must not synthesize missing KV metadata')
        return
    expected = copy.deepcopy(case['kv'])
    if reuse_ids and case['path'] == CHAT:
        expected['prompt_token_ids'] = case['ids']
    require('kv_transfer_params' in body, 'D lost KV metadata')
    equal(body['kv_transfer_params'], expected, 'D KV metadata (including all original fields)')


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_body(self, status, data, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/metrics':
            data = '\n'.join(f'vllm:num_requests_running{{engine="{i}"}} 0'
                             for i in range(self.server.ranks)).encode()
        else:
            data = json.dumps({'status': 'healthy', 'data': [{'id': 'contract'}]}).encode()
        self.send_body(200, data)

    def do_POST(self):
        state = self.server.contract
        record = {'role': self.server.role, 'path': self.path,
                  'rank': self.headers.get('X-data-parallel-rank')}
        # Record dispatch BEFORE assertions, including malformed/unexpected D requests.
        state['calls'].append(record)
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            record['body'] = body
            key = INPUT_KEYS[self.path]
            case = next(c for c in state['cases'] if c['path'] == self.path
                        and c['request'][key] == body.get(key))
            record['case'] = case['name']
            result = {'id': 'contract', 'object': 'chat.completion', 'model': 'contract',
                      'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '10'},
                                   'text': '10', 'finish_reason': 'stop'}],
                      'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}}
            if self.server.role == 'P':
                check_prefill(case, body, state['reuse_ids'])
                for source, target in [('ids', 'prompt_token_ids'), ('kv', 'kv_transfer_params')]:
                    if source in case:
                        result[target] = case[source]
            else:
                check_decode(case, body, state['reuse_ids'])
                if body.get('stream'):
                    result['object'] = 'chat.completion.chunk'
                    result['choices'] = [{'index': 0, 'delta': {'content': '10'},
                                          'finish_reason': 'stop'}]
                    data = ('data: ' + json.dumps(result) + '\n\ndata: [DONE]\n\n').encode()
                    self.send_body(200, data, 'text/event-stream')
                    return
            self.send_body(200, json.dumps(result).encode())
        except Exception as exc:
            # A backend assertion must never masquerade as an expected Router rejection.
            error = f'{self.server.role} {self.path}: {type(exc).__name__}: {exc}'
            state['errors'].append(error)
            self.send_body(500, json.dumps({'error': error}).encode())


def exercise(opener, url, state, report):
    for case in state['cases']:
        before = len(state['calls'])
        request = urllib.request.Request(
            url + case['path'], data=json.dumps(case['request']).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
        try:
            with opener.open(request, timeout=10) as response:
                status, data = response.status, response.read()
                content_type = response.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            with exc:
                status, data = exc.code, exc.read()
            content_type = None
        calls = state['calls'][before:]
        require(not state['errors'], f'Fake backend assertion failed: {state["errors"]}')
        equal([c.get('case') for c in calls if c['role'] == 'P'], [case['name']], 'P dispatch')
        blocked = state['reuse_ids'] and case['reject_when_on']
        equal(status, 500 if blocked else 200, f'{case["name"]} HTTP status: {data!r}')
        equal([c.get('case') for c in calls if c['role'] == 'D'],
              [] if blocked else [case['name']], f'{case["name"]} D dispatch')
        if not blocked:
            if case['request']['stream']:
                equal(content_type, 'text/event-stream', 'stream content type')
                events = [line[6:] for line in data.decode().splitlines() if line.startswith('data: ')]
                require(events and events[-1] == '[DONE]', 'Missing SSE terminator')
                equal(json.loads(events[0])['choices'][0]['delta']['content'], '10', 'stream output')
            else:
                equal(json.loads(data)['choices'][0]['message']['content'], '10', 'output')
        report['scenarios'].append({'name': case['name'], 'status': status, 'blocked': blocked})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, help='Path to a real, locally built Rust Router')
    parser.add_argument('--output', required=True, help='New JSON report path (never overwritten)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mode', choices=('default', 'on', 'off'), default='default')
    group.add_argument('--env-value', help='Exact environment value, e.g. true, 01, or an empty string')
    args = parser.parse_args(argv)
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f'Router binary not found or not executable: {binary}')
    out = Path(args.output)
    log_path = out.with_suffix('.log')
    if out == log_path or out.exists() or log_path.exists():
        parser.error('Report/log must be distinct new paths; do not use a .log output suffix')
    env, reuse_ids = router_environment(args.mode, args.env_value)
    state = {'cases': scenarios(), 'reuse_ids': reuse_ids, 'calls': [], 'errors': []}
    report = {'passed': False, 'mode': args.mode if args.env_value is None else 'custom',
              'env_value': env.get(ENV_NAME), 'reuse_ids': reuse_ids,
              'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'scope': 'CPU fake P2/D8; loopback direct-URL NIXL read; no production requests',
              'scenarios': []}
    servers, proc, log = [], None, None
    try:
        log = log_path.open('x')
        for role, ranks in [('P', 2), ('D', 8)]:
            server = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
            server.role, server.ranks, server.contract = role, ranks, state
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        # Reserve distinct ports together, then release just before Router startup.
        with socket.socket() as http_socket, socket.socket() as metrics_socket:
            http_socket.bind(('127.0.0.1', 0))
            metrics_socket.bind(('127.0.0.1', 0))
            http_port, metrics_port = http_socket.getsockname()[1], metrics_socket.getsockname()[1]
        proc = subprocess.Popen([str(binary), '--host', '127.0.0.1', '--port', str(http_port),
            '--prometheus-host', '127.0.0.1', '--prometheus-port', str(metrics_port),
            '--vllm-pd-disaggregation', '--prefill', f'http://127.0.0.1:{servers[0].server_port}',
            '--decode', f'http://127.0.0.1:{servers[1].server_port}',
            '--kv-connector', 'nixl',
            '--prefill-policy', 'cache_aware', '--decode-policy', 'round_robin', '--api-key', 'EMPTY',
            '--disable-retries', '--worker-startup-timeout-secs', '20'],
            env=env, stdout=log, stderr=subprocess.STDOUT)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = f'http://127.0.0.1:{http_port}'
        for _ in range(150):
            if proc.poll() is not None:
                raise RuntimeError(f'Router exited with status {proc.returncode}; see {log_path}')
            try:
                with opener.open(url + '/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.2)
        else:
            raise RuntimeError(f'Router did not become healthy; see {log_path}')
        exercise(opener, url, state, report)
        with opener.open(f'http://127.0.0.1:{metrics_port}/metrics', timeout=5) as response:
            metrics = response.read().decode()
        samples = [float(line.split()[1]) for line in metrics.splitlines()
                   if line.startswith(METRIC + ' ')]
        expected = sum(c['path'] == CHAT and not c['reject_when_on'] for c in state['cases']) if reuse_ids else 0
        equal(samples, [float(expected)] if reuse_ids else ([0.0] if samples else []), 'reuse counter')
        require(not state['errors'], f'Fake backend assertion failed: {state["errors"]}')
        report.update(passed=True, corrupt_ids_rejected=reuse_ids, reuse_counter=expected)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for server in servers:
            server.shutdown()
            server.server_close()
        if log is not None:
            log.close()
        if state['errors']:
            report.update(passed=False, error=f'Fake backend assertion failed: {state["errors"]}')
        report.update(backend_errors=state['errors'],
                      prefill_calls=[c for c in state['calls'] if c['role'] == 'P'],
                      decode_calls=[c for c in state['calls'] if c['role'] == 'D'])
        with out.open('x') as output:
            output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('prefill_calls', 'decode_calls')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
