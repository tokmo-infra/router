# Mooncake handoff metadata lock convoy

At high PD concurrency, the router could receive a complete P response and
then spend about 64 seconds waiting to prepare the D transfer metadata.
`build_decode_kv_transfer_params` had one suspension point: acquiring the
shared async mutex in `get_mooncake_info`. All P instances used that mutex,
including requests looking up different engine IDs.

The bootstrap table is populated during router construction. It has no
request-time writers. The fix stores it as an immutable `HashMap` and makes
the metadata lookup and parameter construction synchronous. Both direct-URL
and discovery handoff call sites use the synchronous path. Engine-rank lookup,
the existing missing-rank fallback, measured cache usage validation and token-ID
handoff retain their previous behavior.

## Runtime evidence

A passive capture matched 343 complete P/D requests. A complete P response was
acknowledged by TCP after 1.2 ms on average, followed by a mean 64.64-second gap
before D HTTP dispatch. This rules out missing P response tail packets as the
source of that gap.

A separate 110-second userspace function probe recorded repeated polls of the
same metadata future about 64.2 seconds apart. The mutex acquisition was the
only `await` in that function. This identifies metadata lock queueing directly;
a CPU profile alone would not expose the time spent waiting.

## Regression check

Build the release binary and run the CPU-only protocol regression:

```sh
cargo build --locked --release --bin vllm-router
python3 tests/mooncake_handoff_regression.py \
  --binary target/release/vllm-router \
  --output /tmp/mooncake-handoff-result.json \
  --requests 256 --concurrency 128
```

The script starts fake P2/D8 servers on loopback. Each request carries 262,144
input characters and reuses 131,072 prompt token IDs. It checks both P engine
ranks, forwarding of measured cached-token usage, length-cap isolation to P,
and rejection of invalid prompt IDs before D dispatch. It uses no model or GPU.

The isolated regression passed for the fixed binary. The existing NIXL token-ID
handoff contract also passed. This workload checks protocol compatibility; it
does not reproduce the long-running production lock queue or establish an
end-to-end production TTFT improvement. Recheck P-to-D dispatch timing after
rolling out the fixed router.
