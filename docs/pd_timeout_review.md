# Direct HTTP P/D deadline review (e3786db baseline)

Scope: CPU-only patch for the direct sequential HTTP P/D path used by Mooncake.
No deployment files, services, images, model files, Git commits or pushes changed.
The planned runtime remains our fa7ab285 small Router lineage, not a third-party
Router. This patch does not build or deploy that next image.

## Confirmed failure path

`server.rs` builds the shared reqwest 0.13.5 client with
`timeout(request_timeout_secs)` and `connect_timeout(10s)`. The original direct
P/D implementation uses that client for both POSTs. Reqwest's overall timeout
includes the entire response body; receiving chunks does not reset it. Thus a
600s setting can abort a slow P before its response and can cut an actively
generating D stream 600s after the D POST. It was per leg, not a shared P+D budget.
Loopback regression scales 600s to 80ms and reproduces that failure. This is
proof of the code path, NOT attribution of every historical EOF to that cause.

## New bounded semantics

All values below are positive integer seconds, <=86400; stage limits must be
<=total. Invalid configuration fails Router initialization. No zero/infinite mode.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_ROUTER_PD_CONNECT_TIMEOUT_SECS` | min(10,total) | Data-plane connection/TLS establishment, not queue or TTFT |
| `VLLM_ROUTER_PD_FIRST_RESPONSE_TIMEOUT_SECS` | total | Each P or D POST through response headers AND its first nonempty body chunk; headers alone do not reset it |
| `VLLM_ROUTER_PD_IDLE_TIMEOUT_SECS` | total | Pending upstream read after first data; downstream pauses between delivered chunks do not count |
| `VLLM_ROUTER_PD_TOTAL_TIMEOUT_SECS` | existing request_timeout_secs | One absolute budget from direct worker selection/token estimation through P and D body completion |

Defaults deliberately do not silently increase the old configured ceiling. The
default total now covers P+D together, so it can expire earlier than two separate
old per-leg limits. To remove the inherited 600s leg cutoff, explicitly choose a
reviewed PD total. The control-plane client/CLI timeout is unchanged.

The dedicated PD client disables reqwest retries; dispatch overrides its timeout
with the remaining shared budget. The stream is lazy: no spawned reader, channel,
SSE parser, response buffering, or chunk rewriting. Only P and the pre-existing
non-streaming logprobs merge buffer whole bodies. Idle Sleep storage is reused,
not allocated per token. Total expiry remains absolute during backpressure and
is observed on polling; an unpolled response must still be dropped by its owner.
No new detached cleanup/drain task is introduced.

P/D load guards and token-balance reservations keep their original ownership.
P releases on P completion; D spans P wait and response body EOF/error/drop.
Timeout/cancellation drops upstream futures, notifies no separate backend abort
API and does not claim real GPU KV reclamation. Backend disconnect/abort behavior
still needs real-service smoke validation. Existing public pre-header error
mapping (500, or 503 during selection) is retained; a stream already started
fails its body rather than fabricating an SSE success or retrying the POST.

Discovery/concurrent push branches are not claimed fixed by this scoped patch.

## Observability and the 60s stop rule

- `vllm_router_frontend_admission_wait_seconds{outcome}`: frontend queue/admission;
  records cancellation without changing permit ownership. This precedes PD total.
- Existing `vllm_router_pd_estimation_seconds{phase}`: tokenizer slot wait, blocking
  executor wait, compute and total. Tokenization concurrency remains four.
- `vllm_router_pd_stage_waiting{stage,phase}` and
  `vllm_router_pd_stage_wait_seconds{stage,phase,outcome}`: P/D response headers and
  first body. Outcomes include cancellation/error, not only successful requests.
- `vllm_router_pd_first_decode_body_seconds`: from PD selection to first D body.
  **Not true first-token TTFT**: an SSE role or heartbeat may arrive first.
- `vllm_router_pd_timeout_total{phase}`: connect, first_response, body_idle, total.
  Streaming timeout warnings include stage/phase/elapsed time, no prompt/key.

Histogram buckets include 30/45/60/90 seconds. Backend queue versus GPU compute
cannot be separated from Router waiting alone: correlate P/D native running,
waiting and KV metrics. Router admission is not a replacement for those metrics.

For main-thread review, a bounded diagnostic starting point is connect=5,
first_response=60, idle=60, total=1800 seconds. This is NOT a measured optimum.
Total must also be reviewed against output cap and observed TPOT. Raising total
only permits long active generation; it cannot make high TTFT acceptable.

**Any true TTFT >60s stops the benchmark.** Preserve the independent perf in-flight
watchdog; per-stage first-response limits cannot enforce end-to-end true TTFT.
Use the requested 16 CPU/64GiB limit, initial C1024 and 2048 user samples. No claim
that CPU/RSS or the 60s SLO has been verified on that production topology.

New topology is four independent DEP2 P on .92 (8 P ranks), one single-node DEP8
D on .93 (8 D ranks). Keep prompt-ID reuse explicitly enabled and token-balance
capacities measured per actual rank, not summed origin capacities. No real
capacities are supplied by this CPU-only work.

## Separate P-only stopping-contract risk (NOT changed here)

Frozen DS e0d00e4b8 source: ChatCompletionRequest accepts ignore_eos/min_tokens;
`vllm/v1/core/sched/utils.py:98` checks EOS/stop token before max length.
Mooncake only retains P blocks for FINISHED_LENGTH_CAPPED. Router max_tokens=1
therefore does not alone prove that KV survives an early stop.

`VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP` is not read by this branch. Do not use
that environment variable as a fix. This patch deliberately does not mix a
sampling-policy change into the timeout fix: ignore_eos/min_tokens acceptance
alone does not prove correct native-model behavior for EOS/custom stop paths.

Before real performance runs, independently smoke-test EOS-first/stop-token-first
P completion, retained/transferred KV, valid prompt IDs and measured cache usage;
D must retain the user's actual sampling policy. A failure is HOLD and a separate
P-only correctness patch, not a POST retry or a longer timeout.

## CPU reproduction

Standard targets (with the repository's pinned Rust dependencies available):

    cargo test --offline --locked --lib timeout
    cargo test --offline --locked --lib
    python3 -B -m unittest discover -s scripts -p test_check_pd_prompt_ids.py -v

Local isolated cache/runner and logs are under this repository's ignored
`target/pd-timeout-work/`. They use Rust 1.99.1 and CLI path overrides to the
existing archived vLLM v0.29 Rust dependency sources; Cargo.toml/Cargo.lock are
unchanged. The runner and all build/cache writes stay inside this repository.

Validated 2026-11-05:

- Full Rust lib: **794 passed**, 0 failed (31.69s test execution).
- Five integration targets: **34 passed**, 0 failed: `otel_disabled_path_test` (4),
  `policy_registry_integration` (3), `streaming_tests` (7), `test_dp_routing` (18),
  `test_pd_routing` (2).
- Python prompt-ID checker tests: **7 passed**. These test the checker itself,
  not a real DS backend; Rust loopback tests cover the actual two-stage method.
- New regressions cover legacy active-stream cutoff, inherited timeout override,
  slow P, TLS/connect versus first response, first-body wait, body idle, shared
  P+D total, lazy/no-pump backpressure, EOF/error/drop/cancellation cleanup,
  measured usage and prompt-ID fail-closed contracts, 32 concurrent mock requests,
  token-balance demand, and queue/stage metric cleanup.
- `git diff --check` passed; HEAD remains e3786dbd41bcfa659af1835bc729eb0121af031b.

Local replay:

    bash /Users/yyf/workspacke/glm5.3-flash-1103-router/target/pd-timeout-work/run-cargo.sh test --offline --locked --lib
    bash /Users/yyf/workspacke/glm5.3-flash-1103-router/target/pd-timeout-work/run-cargo.sh test --offline --locked --test test_pd_routing --test streaming_tests --test test_dp_routing --test policy_registry_integration --test otel_disabled_path_test

Evidence: `full-lib-final.log`, `integration.log`, `prompt-checker.log` under the
same isolated target directory. No real inference, remote operations or model
performance/RSS claims are part of these results.
