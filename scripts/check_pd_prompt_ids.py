"""Exercise the released Router's token-ID handoff with CPU-only fake backends."""
import argparse
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

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output)
assert not out.exists()
calls = []
class Backend(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        if self.path == '/metrics':
            body = '\n'.join(f'vllm:num_requests_running{{engine="{i}"}} 0' for i in range(self.server.ranks)).encode()
        else:
            body = json.dumps({'status': 'healthy', 'data': [{'id': 'contract'}]}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        content = body['messages'][0]['content']
        result = {'id': 'contract', 'object': 'chat.completion', 'model': 'contract',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '10'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}}
        if self.server.role == 'P':
            assert body.get('return_token_ids') is True
            assert body['stream'] is False
            result.update(prompt_token_ids=[True] if content == 'invalid-ids' else [3, 7, 9],
                kv_transfer_params={'do_remote_prefill': True, 'remote_engine_id': 'original-engine', 'remote_block_ids': [11]})
        else:
            params = body['kv_transfer_params']
            assert params['prompt_token_ids'] == [3, 7, 9]
            assert params['remote_engine_id'] == 'original-engine' and params['remote_block_ids'] == [11]
            assert body.get('return_token_ids') is not True
            calls.append({'rank': self.headers.get('X-data-parallel-rank'), 'ids': params['prompt_token_ids']})
        data = json.dumps(result).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

servers=[]
for role, ranks in [('P', 2), ('D', 8)]:
    s=ThreadingHTTPServer(('127.0.0.1',0),Backend);s.role=role;s.ranks=ranks
    threading.Thread(target=s.serve_forever,daemon=True).start();servers.append(s)
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
http_port, metrics_port = port(), port()
env={**os.environ,'VLLM_ROUTER_PD_REUSE_PROMPT_IDS':'1','NO_PROXY':'*','no_proxy':'*'}
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
log=out.with_suffix('.log').open('x')
proc=subprocess.Popen([a.binary,'--host','127.0.0.1','--port',str(http_port),
    '--prometheus-host','127.0.0.1','--prometheus-port',str(metrics_port),
    '--vllm-pd-disaggregation','--prefill',f'http://127.0.0.1:{servers[0].server_port}',
    '--decode',f'http://127.0.0.1:{servers[1].server_port}',
    '--prefill-policy','cache_aware','--decode-policy','round_robin','--api-key','EMPTY',
    '--disable-retries','--worker-startup-timeout-secs','20'],env=env,stdout=log,stderr=subprocess.STDOUT)
report={'passed':False,'scope':'CPU fake P2/D8; no production requests'}
try:
    for _ in range(100):
        try:
            with op.open(f'http://127.0.0.1:{http_port}/health',timeout=1) as r:
                if r.status==200:break
        except (OSError,urllib.error.URLError):time.sleep(.2)
    else:raise RuntimeError('Router did not become healthy')
    def call(content):
        body={'model':'contract','messages':[{'role':'user','content':content}],'stream':False,'max_tokens':8}
        req=urllib.request.Request(f'http://127.0.0.1:{http_port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
        with op.open(req,timeout=10) as r:return json.load(r)
    assert call('correct handoff')['choices'][0]['message']['content']=='10'
    count=len(calls)
    try:call('invalid-ids')
    except urllib.error.HTTPError as e:assert e.code==500
    else:raise AssertionError('Corrupt P ids must fail before decode')
    assert len(calls)==count
    with op.open(f'http://127.0.0.1:{metrics_port}/metrics',timeout=5) as r:metrics=r.read().decode()
    assert 'vllm_router_pd_prompt_ids_reused_total 1' in metrics
    report.update(passed=True,decode_calls=calls,corrupt_ids_rejected=True)
finally:
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    for s in servers:s.shutdown();s.server_close()
    log.close();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
