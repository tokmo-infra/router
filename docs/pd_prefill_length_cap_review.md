# Opt-in Mooncake P-only length-cap follow-up

Base: e11515a76fe7e97b27bebf02635cd213febe9a49. This is a separate opt-in
correctness follow-up to the timeout patch. No engine/main, deployment files,
dependencies, timeout implementation, retry logic or stream lifecycle changed.
The CPU implementation/review did not perform SSH, a Linux release/image build
or deployment. Main-thread commit/build/runtime identities are recorded in the
deployment task rather than inferred from these CPU results.

## Switch and exact scope

VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP is now actually read in
VllmPDRouter::new, once at initialization. Only the exact string 1 enables it;
unset, 0, true, 01 and whitespace-padded values leave it OFF. The default is OFF.
The earlier e11515 timeout-review statement that this env was not consumed is
historically correct for that base, but is superseded by this follow-up.

Only internal P requests in the direct sequential Mooncake route for exact
/v1/chat/completions and /v1/completions receive the extra normalization:

| Field | Internal P with opt-in1 | D |
| --- | --- | --- |
| max_tokens / present max_completion_tokens | 1, as before this patch | Original client values |
| ignore_eos | true | Original client value / absence |
| min_tokens | 1 | Original client value / absence |
| stop | Explicit empty array | Original string/list/null/absence |
| stop_token_ids | Explicit empty array | Original IDs/null/absence |
| stream / stream_options | false / removed, as before | Original values |
| prompt IDs / KV handoff / measured usage | Existing contract unchanged | Existing metadata and IDs unchanged |

The original request is cloned for P. D still clones the original, not the
normalized P body. The new helper changes no messages, prompt, tools, template
kwargs, reasoning_effort=max, thinking_token_budget, opaque vllm_xargs (including
max_thinking_tokens), seed, temperature, output formatting or D output budget.
It adds no blanket removal of unknown sampling fields.

NIXL, MoRI-IO, discovery, concurrent push, Responses, Generate and batch APIs
keep their old behavior; this switch does not certify them. With this opt-in
active on the supported Mooncake direct endpoints, use_beam_search=true is
explicitly rejected before inference dispatch: beam search does not use the
same min_tokens sampling path. Its default-off/other-connector behavior is not
changed. Existing error mapping is retained; no new HTTP status policy added.

The sole new dispatch call is in process_vllm_two_stage_request_inner. The
general prepare_prefill_request used by other paths is unchanged. P/D load
ownership, D body EOF/error/drop, token-balance reservations, async KV transfer,
backpressure, timeout budgets and the no-POST-retry contract are unchanged.

## Actual native schema / rationale

The actual deployment-specified native0.33/e0d Git object was read, not assumed
from the GLM reference branch. It exists in the read-only checkout at:

/Users/yyf/workspacke/dsv41-roce-megamoe-b30056-20261102/native-return-candidate-20261103/source

Object: e0d00e4b8eb0c89ec30921d0695cfdd409e9c211. Read with git show OBJECT:PATH:

- vllm/entrypoints/openai/chat_completion/protocol.py:256,294-297 declares
  stop, stop_token_ids, ignore_eos and min_tokens at top level. Lines772-778
  merge server-default stop IDs even when the request's list is empty.
- vllm/entrypoints/openai/completion/protocol.py:68,83-86 has the same fields;
  lines380-386 also merge server-default stop IDs. This is why clearing the
  client IDs alone is insufficient.
- vllm/sampling_params.py:785-806 suppresses primary/additional generation EOS
  stopping when ignore_eos is true, while retaining all_stop_token_ids for
  min-token processing.
- vllm/v1/sample/logits_processor/builtin.py:198-206,280-288 masks stop IDs
  while the output count is below min_tokens. min_tokens=1 protects the first
  sample against server-default stop IDs as well as the client's cleared IDs.
- vllm/v1/core/sched/utils.py:105-118 checks EOS/stop IDs BEFORE the length cap.
  Merely max_tokens=1 is not a length-cap guarantee. Repetition detection is
  checked after the length-cap branch, so this patch leaves its field alone.
  early_stopping belongs to the batch/beam schema, not these two normal APIs;
  include_stop_str_in_output is formatting, not a separate stop predicate.

Retained runtime-source evidence for the actual Mooncake finish contract:

/Users/yyf/workspacke/dsv41-roce-megamoe-b30056-20261102/c4096-p3-d8x2-ep16-20261104/mooncake/evidence/runtime-source/mooncake/mooncake_connector.py:1308

That producer path retains pending KV only for FINISHED_LENGTH_CAPPED. Other
finish statuses can immediately release blocks. This patch requests the right
P stopping behavior; the CPU mocks do NOT execute that native connector or
prove real GPU KV retention. Model/structured-output/logit constraints can
conflict with stop masking (native min-token logic can restore constrained stop
logits); their behavior remains a real-engine acceptance item, not a CPU claim.

## Reproducible CPU coverage

Commands from /Users/yyf/workspacke/glm5.3-flash-1103-router:

    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 bash target/pd-prefill-force-work/run-cargo.sh test --offline --locked --lib force_length_cap -- --test-threads=4
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 bash target/pd-prefill-force-work/run-cargo.sh test --offline --locked --lib -- --test-threads=4
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 bash target/pd-prefill-force-work/run-cargo.sh test --offline --locked --test test_pd_routing --test streaming_tests --test test_dp_routing --test policy_registry_integration --test otel_disabled_path_test -- --test-threads=4
    python3 -B -m unittest discover -s scripts -p test_check_pd_prompt_ids.py -v

The ignored target-local harness is necessary because the manifest's sibling
vllm-rust-v029 path no longer exists. It symlinks LIVE workspace source into a
test manifest, changes only the three dependency paths to the existing prep
archive, copies the unchanged Cargo.lock, and reuses the existing Cargo cache.
It does not change repository Cargo.toml/Cargo.lock or any source under engine.

The first full-suite attempt failed eight unrelated WASM tests because that
test manifest lacked examples/. The corrected harness exposes the existing
prebuilt WASM fixture. Its restricted PATH prevents rustup/global installation;
the WASM helper's existing fallback loads that fixture when rustup is absent.
The initial failure log is retained, not hidden.

New regressions extend the existing loopback CPU harness:

- Separate child test processes exercise the REAL constructor environment read
  for unset/0/1/true/01/space1; no global set_var race in parallel tests.
- Each mode runs real Router two-stage dispatch through fake P/D HTTP handlers
  for Chat and Completions, with null/string/list stop forms, custom stop IDs,
  inherited ignore_eos=false/min_tokens=0, streaming and non-streaming bodies.
- Exact P body comparison limits changes to the expected internal-P fields.
  Exact D JSON-value comparison retains every client sampling/thinking/body
  field after removing only the existing internal KV handoff envelope.
- Assert matching transfer ID, bootstrap/engine identity, measured cached-token
  count, valid Chat prompt IDs, no IDs added to Completions, one P and one D
  POST per request, and local reservations released on EOF and body drop.
- Scope matrix verifies unsupported APIs/connectors/discovery unchanged and
  enabled beam search rejected with local admission released.
- Existing timeout/slow-P/stream/cancellation/missing-usage/invalid-ID regressions
  and the Python checker suite are rerun. The checker tests alone are not a
  certification of a production Router binary or native sampler.

Final test results and logs are recorded separately under
target/pd-timeout-work/p-only-force-*.log and
target/pd-prefill-force-work/delivery.json. No new production image/ELF identity
is asserted by these CPU tests.

Final run: 797 Rust library tests passed (including all12 timeout/force-cap
module tests), 34 selected integration tests passed, and 7 Python prompt-ID
checker tests passed. The opt-in matrix covers six isolated constructor env
modes and42 mock P/D pairs; these are fake backends, not real GPU sampling.

## Minimum Linux incremental source replacement

Release compilation needs only:

1. src/routers/http/vllm_pd_router.rs

For the same reviewed source/test snapshot also transfer:

2. src/routers/http/pd_timeout_tests.rs
3. docs/pd_prefill_length_cap_review.md (review documentation, not compiled)

No dependency/Cargo.lock/vendor/toolchain change. Existing Linux dependency
targets may be reused; this task did not perform a Linux build. Do NOT overwrite
the sealed a09a4f3c package and continue claiming its old manifest/source ID.
Main thread must freeze the new inputs, refresh provenance/checksums, rebuild
and record the actual new binary/image ID. Do not skip bundle verification.

## Planner integration gate (owner must update AFTER review)

The separate planner remains intentionally unmodified at:

/Users/yyf/workspacke/dsv41-roce-megamoe-b30056-20261102/p4x2-d8-slo-20261105/router-build/router_plan.py

- SOURCE_HASHES: the ONLY existing schema-bearing entry changed by this patch
  is src/routers/http/vllm_pd_router.rs. The other eight pinned source entries
  are unchanged. The test file is not currently a SOURCE_HASHES entry.
- AUDITED_REVISION: replace e11515 only with the actual reviewed follow-up
  commit after main thread commits it. No nonexistent commit is supplied here.
- FROZEN_SOURCE_SHA256: replace a09a4f3c only with the actual newly frozen input
  manifest's identity. The old package does not contain this patch.
- Refresh the old planner/README claim that FORCE_LENGTH_CAP is not consumed;
  it is now consumed only within the above scope. Preserve default OFF and add
  explicit opt-in output only for the rebuilt, verified image. This round's
  deployment is explicitly authorized by the user to set its non-secret env1.
- Refresh the image ID/binary provenance, rather than appending env1 to the
  existing 1cfb/e11515 image and assuming behavior changed. Retain native EOS,
  stop/KV and prompt-ID smoke blockers and the true TTFT>60s watchdog.

Until those owner updates occur, the old planner's source audit SHOULD REFUSE
this modified workspace. Do not drop --router-source or alter hashes just to
make that old audit pass. Source-file SHA is not a commit, build or runtime ID.

## Required main-thread real smoke (NOT done by this task)

Using the newly rebuilt Router with explicit env1, show EOS-first and custom
stop-string/stop-token-first cases: P reaches FINISHED_LENGTH_CAPPED, pending
KV is retained/transferred, D receives valid prompt IDs and exact measured
cache usage, and D still honors the ORIGINAL EOS/stop/output/thinking policy.
Check both Chat and Completions, plus cancellation/abort reclamation. Keep the
full64path and healthy-rank gates. Any actual TTFT>60s stops performance testing.
Native sampler acceptance/retention, GPU results and latency are not certified
by mock request-body assertions.
