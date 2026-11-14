# Kimi3.1 HiSparse hybrid ingress

CPU-only aiohttp ingress in front of the existing Rust PD router. P local generation and the PD path use weighted least in-flight admission. The default deployment is5:9 on92:19511; C288 produces103local/185PD reservations. Override HYBRID_P_WEIGHT/HYBRID_PD_WEIGHT and HYBRID_ROUTER_PORT for the archived1:2 deployment.

POST /v1/chat/completions and /v1/completions are weighted. Other APIs, including Responses, use the original PD router. Local requests remove only kv_transfer_params. Streaming responses, output limits and sampling settings are preserved. Reservations include prefill and handoff wait, and release on EOF, failure or client cancellation. No automatic retry. Run exactly one process.

P and D are not restarted. This wrapper uses the existing CPU Python/aiohttp environment in the P image without requesting GPUs. Existing Rust PD router source and routing policies are unchanged by this addition.

```bash
bash router.sh --dry-run
bash router.sh --execute
curl -fsS http://11.5.0.92:19511/routing_state
```

Install requirements.txt in a CPU environment and run unittest discover for test_hybrid_router.py. Tests use actual loopback HTTP/SSE, including C2885:9, disconnect-before-headers, timeout and reservation release. They do not issue inference requests.

Model deployment parameters and benchmark YAML belong to https://github.com/tokmo-infra/vllm/tree/kimi31-dcp-hisparse-pd/tokmo . This repository owns the hybrid ingress implementation and tests; runtime binaries and model source commits are recorded separately in that deployment manifest.
