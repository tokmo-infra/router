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

## Mooncake producer completion policy

For a Mooncake producer that publishes transferable KV only on
`FINISHED_LENGTH_CAPPED`, enable this additional Router option:

```bash
export VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1
```

This applies to the P request copy for `/v1/chat/completions` and
`/v1/completions`. In addition to the existing one-token output limit, it sets
`min_tokens=1`, `ignore_eos=true`, `stop=null`, and `stop_token_ids=[]`.
The D request retains the caller's original output limit and stop/EOS policy.
The option is disabled when the variable is unset and is independent of
`VLLM_ROUTER_PD_REUSE_PROMPT_IDS`.

Without this option, a P request can end with EOS/stop before entering the
producer's length-capped KV handoff path. D can then wait for KV that was not
published. Verify that behavior against the deployed connector version before
enabling it; the policy is intended for connectors with this handoff contract.

The 2026-11-04 B300 validation used seven P2 instances and one D32/EP32 group.
All 40 requests succeeded, including 32 forced-stop cases with joined P/D
response IDs: P returned one token with `length`, while D returned one token
with `stop`. Eight longer inputs (about 1336 tokens each) also succeeded.
Producer metrics recorded 129,841,152 transferred bytes with no new failed
transfers, failed receives, or expired KV requests. This is a correctness
regression result; sustained benchmark performance is measured separately.

The deployed source file SHA-256 is
`74ab49345aa17a286f47b2d864623e8a169e1db8beea2e81df06bd41fba07b2f`.
The matching deployed Router binary SHA-256 is
`1b5ab72d7d00491c31bff54bbb6420cad8b9d0095a2c7268b7ef6ecf68c210ae`.
