# Decode-side prompt token reuse

Long multi-turn PD requests can bottleneck on decode-side CPU tokenization even
when the GPU scheduler and KV transfer are fast. For a compatible vLLM backend,
enable the existing native pre-tokenized chat path with:

```bash
export VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1
```

Keep the existing Router deployment arguments. This opt-in applies to
`/v1/chat/completions`; other routes retain their existing behavior.

The Router requests `return_token_ids=true` from P, validates the returned
`prompt_token_ids`, and forwards them in D's `kv_transfer_params.prompt_token_ids`.
All other KV-transfer metadata is preserved. The decode request keeps the
client's original output options. Missing or malformed IDs fail the request
before dispatching D. The original behavior remains the default when the
environment variable is unset.

The backend must support `OnlineRenderer._reused_prompt_token_ids`. P and D must
use matching model/tokenizer/template settings. Native vLLM still performs its
request validation and token-vocabulary checks; `echo` and non-text media can
require re-rendering, and batched chat does not accept one shared ID list.

P must complete before D can reuse its IDs, so concurrent P/D dispatch is
disabled for this opt-in chat path. No model-engine restart is required to
enable the Router option when the backend already supports native reuse.

`vllm_router_pd_prompt_ids_reused_total` counts successfully prepared ID
handoffs in the Router, including handoffs whose later decode HTTP request may
fail. Confirm actual backend reuse using CPU profiles and frontend latency;
the counter alone is not a measurement of backend CPU time saved.

A CPU-only contract check uses fake P2/D8 backends and rejects malformed IDs
before D dispatch:

```bash
python3 scripts/check_pd_prompt_ids.py --binary target/release/vllm-router --output /tmp/pd-prompt-ids.json
```
