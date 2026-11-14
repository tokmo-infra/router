#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:---dry-run}
ROUTER_NAME=${HYBRID_ROUTER_NAME:-kimi31-hisparse-hybrid-92-router-5to9}
ROUTER_PORT=${HYBRID_ROUTER_PORT:-19511}
PREFILL_URL=${HYBRID_PREFILL_URL:-http://11.5.0.92:19580}
PD_ROUTER_URL=${HYBRID_PD_ROUTER_URL:-http://11.5.0.92:19500}
CPU_IMAGE=${HYBRID_CPU_IMAGE:-sha256:dc18c152f9318756058df9dfd43124c4865e1ba392452be7195a1a28d5cbb3dd}
command=(
  docker run -d
  --name "$ROUTER_NAME"
  --pull=never --runtime=runc --network host --restart no
  --label ai-infra.plan=kimi31-b300-hisparse-hybrid-5to9
  --label ai-infra.node=92 --label ai-infra.role=hybrid-router
  --log-driver local --log-opt max-size=20m --log-opt max-file=3
  --memory 16g --cpus 8 --ulimit nofile=1048576:1048576
  -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONUNBUFFERED=1
  -e 'NO_PROXY=*' -e 'no_proxy=*'
  --mount "type=bind,src=$SCRIPT_DIR,dst=/hybrid,readonly"
  --entrypoint python3 "$CPU_IMAGE"
  /hybrid/hybrid_router.py
  --host 0.0.0.0 --port "$ROUTER_PORT"
  --prefill-url "$PREFILL_URL" --pd-router-url "$PD_ROUTER_URL"
  --p-weight "${HYBRID_P_WEIGHT:-5}" --pd-weight "${HYBRID_PD_WEIGHT:-9}"
  --max-concurrent-requests "${HYBRID_MAX_CONCURRENT_REQUESTS:-1024}"
  --request-timeout-secs "${HYBRID_REQUEST_TIMEOUT_SECS:-7200}"
)
case "$MODE" in
  --dry-run) printf '%q ' "${command[@]}"; printf '\n' ;;
  --execute)
    ip -o addr show | grep -F '11.5.0.92/' >/dev/null || {
      echo 'Run on b300-92.' >&2; exit 1;
    }
    docker image inspect "$CPU_IMAGE" >/dev/null
    if docker inspect "$ROUTER_NAME" >/dev/null 2>&1; then
      echo "Container already exists: $ROUTER_NAME; inspect its state." >&2; exit 1
    fi
    if [[ -n $(ss -H -ltn "sport = :$ROUTER_PORT") ]]; then
      echo "Port already occupied: $ROUTER_PORT" >&2; exit 1
    fi
    curl -fsS --max-time 5 "$PREFILL_URL/health" >/dev/null
    curl -fsS --max-time 5 "$PD_ROUTER_URL/health" >/dev/null
    exec "${command[@]}"
    ;;
  *) echo 'Usage: [--dry-run|--execute]' >&2; exit 2 ;;
esac
