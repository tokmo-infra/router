# GLM-5.3 Flash · 3P4D

## 拓扑与参数

| 主机 | 角色 | TP / 全局DP / EP | 本地DP | API | seq/DP | batch/DP | util | CUDA Graph / 上限 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81、83、84 | 各一个P实例 | 1 / 8 / 8 | 8 | 各4 | 32 | 2048 | 0.95 | FULL_DECODE_ONLY / 256 |
| 85 | D主节点 | 1 / 32 / 32 | 8，rank 0–7 | 8 | 32 | 1024 | 0.95 | 128 |
| 86、87、88 | D headless | 1 / 32 / 32 | 各8，起始rank 8/16/24 | 0 | 32 | 1024 | 0.95 | 128 |

| 项目 | 值 |
| --- | --- |
| 推理、Mooncake镜像 | local/glm53flash-vllm:0.32.0，基于540155b18ee7叠加缓存/API修复 |
| 外层Router、API网关镜像 | local/glm53flash-router:0.1.16，基于21c9cfa19e34增加API代理 |
| 模型 / KV / MTP | GLM-5.3-Flash-NVFP4 / FP8 / 3 |
| 上下文 / 匹配粒度 | auto，实际1048576 / 256 |
| block-size / 实际物理页 | 64 / 8960 |
| Attention / SSM | FLASHINFER_MLA_SPARSE_SM120 / float32 |
| MoE / FlashInfer autotune | 原生自动选择 / 原生自动调优 |
| P all2all / D all2all | allgather_reducescatter / deepep_v2 |
| D DeepEP补丁 | 镜像内置，AI_INFRA_DEEPEP_V2_NUM_SMS=32，不再额外挂载elastic.py |
| Mooncake KV-load接收线程 | P=4，D=1 |
| Tokenizer CPU预算 | P每GPU等效2GiB，每EP8组16GiB；D主节点85共享8GiB＋8个API各64MiB，共8.5GiB |
| Tokenizer共享池 | P每API主机15.75GiB＋每API L1 64MiB；D主节点85共享8GiB＋每API L1 64MiB；headless不重复建池 |
| Tokenizer共享载体 | 同机模型/API网关共享Docker tmpfs卷 |
| L2 | 七台各750GiB，共5250GiB |
| L3 | 关闭；Master/Store enable_offload=false，无SSD目录挂载 |

## 公共变量

~~~bash
export NODE=81  # 各机分别填写81、83、84、85、86、87、88
export IP=11.5.0.$NODE
export PROJECT=glm53flash-3p4d-l2-20261031
export WORK="${GLM_WORKDIR:-$HOME/yyf/glm53flash-3p4d-l2-20261031}"
export MODEL_DIR="${GLM_MODEL_DIR:-/data/models/GLM-5.3-Flash-NVFP4}"
export IMAGE=local/glm53flash-vllm:0.32.0
export ROUTER_IMAGE=local/glm53flash-router:0.1.16
export CACHE_PREFIX=glm53flash-3p4d-l2-20261031-v1
export ALL_HCAS=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export TOKEN_VOLUME="$PROJECT-tokenizer"
mkdir -p "$WORK/config"
~~~

## Mooncake master：81

~~~bash
docker run -d --init --name "$PROJECT-master-81" --pull=never --network host \
  --log-opt max-size=128m --log-opt max-file=5 --label ai-infra.project="$PROJECT" \
  --memory 8g -e NO_PROXY='*' -e no_proxy='*' -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint mooncake_master "$IMAGE" \
  --rpc_port=54395 --enable_http_metadata_server=true \
  --http_metadata_server_host=11.5.0.81 --http_metadata_server_port=54396 \
  --metrics_port=29495 --eviction_high_watermark_ratio=0.98 \
  --enable_offload=false --offload_on_evict=false --offload_force_evict=false \
  --allocation_strategy=local_first --logtostderr=true
~~~

## Mooncake Store：七台各一个

~~~bash
case "$NODE" in
  81|83|84) L2=805306368000; MEM=798g; CPUS=0-255 ;;
  85|86|87|88) L2=805306368000; MEM=798g; CPUS=0-127 ;;
esac
TOPOLOGY="$WORK/config/mooncake-topology-store-$NODE.json"
cat > "$TOPOLOGY" <<'JSON'
{"cpu:0":[["mlx5_0","mlx5_1","mlx5_2","mlx5_3"],["mlx5_4","mlx5_5","mlx5_6","mlx5_7"]],"cpu:1":[["mlx5_4","mlx5_5","mlx5_6","mlx5_7"],["mlx5_0","mlx5_1","mlx5_2","mlx5_3"]]}
JSON
docker run -d --init --name "$PROJECT-store-$NODE" --pull=never --network host \
  --log-opt max-size=128m --log-opt max-file=5 --label ai-infra.project="$PROJECT" \
  --device /dev/infiniband --ipc host --cap-add IPC_LOCK --cap-add SYS_NICE \
  --ulimit memlock=-1 --ulimit stack=67108864 --security-opt seccomp=unconfined \
  --memory "$MEM" --cpuset-cpus "$CPUS" --cpuset-mems 0,1 \
  -e NO_PROXY='*' -e no_proxy='*' -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e NVIDIA_VISIBLE_DEVICES=void -e MC_TCP_BIND_ADDRESS="$IP" \
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$IP" -e WITH_NVIDIA_PEERMEM=false \
  -e MC_TE_FILTERS="$ALL_HCAS" -e MC_MS_AUTO_DISC=1 -e MC_MS_FILTERS="$ALL_HCAS" \
  -e MC_CUSTOM_TOPO_JSON=/etc/mooncake/topology.json \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
  -v "$TOPOLOGY:/etc/mooncake/topology.json:ro" \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/opt/host-driver/libcuda.so.1:ro \
  --entrypoint numactl "$IMAGE" --cpunodebind=0,1 --interleave=0,1 mooncake_client \
  --host="$IP:15495" --port=54495 --metadata_server=http://11.5.0.81:54396/metadata \
  --master_server_address=11.5.0.81:54395 --protocol=rdma --device_names="$ALL_HCAS" \
  --global_segment_size="$L2" --local_buffer_size=4294967296 \
  --enable_offload=false --start_offload_rpc_server=false --enable_http_server=true \
  --http_port=29496 --logtostderr=true
~~~

## P、D模型

~~~bash
EXTRA_ENV=(); EXTRA_MOUNTS=(); API_ARGS=()
case "$NODE" in
  81|83|84)
    ROLE=prefill; SUFFIX=prefill-$NODE-0; ENGINE="$PROJECT-$SUFFIX"
    DP=8; SEQS=32; BATCH_TOKENS=2048; RECV_THREADS=4; UTIL="${GLM_GPU_UTILIZATION:-0.95}"; ALL2ALL=allgather_reducescatter
    COMPILATION='{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":256}'
    PORT=31010; RPC=18360; BOOTSTRAP=27600; NET_IF=enp3s0f0np0; P2P=PXB
    KV_HCAS=mlx5_1,mlx5_3,mlx5_5,mlx5_7
    EP_HCAS='=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1'
    PD_ROLE=kv_producer; STORE_ROLE=kv_both; BUFFER=536870912
    CACHE_GIB=2; SHARED_GIB=15.75; APIS=4
    API_ARGS=(--api-server-count 4 --renderer-num-workers 1 --host 0.0.0.0 --port "$PORT")
    ;;
  85|86|87|88)
    ROLE=decode; SUFFIX=decode-$NODE; ENGINE="$PROJECT-dmem095-$SUFFIX"
    DP=32; SEQS=32; BATCH_TOKENS=1024; RECV_THREADS=1; UTIL="${GLM_GPU_UTILIZATION:-0.95}"; ALL2ALL=deepep_v2
    COMPILATION='{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}'
    PORT=31011; RPC=28360; BOOTSTRAP=28600; NET_IF=ens50np0; P2P=PIX
    KV_HCAS=mlx5_2,mlx5_3,mlx5_4,mlx5_7
    EP_HCAS='=mlx5_0:1,mlx5_1:1,mlx5_5:1,mlx5_6:1'
    PD_ROLE=kv_consumer; STORE_ROLE=kv_consumer; BUFFER=2147483648
    CACHE_GIB=0; SHARED_GIB=0; APIS=0
    EXTRA_ENV=(-e NCCL_GIN_TYPE=2 -e AI_INFRA_DEEPEP_V2_NUM_SMS=32 -e VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=0)
    if [ "$NODE" = 85 ]; then
      CACHE_GIB=4; SHARED_GIB=8; APIS=8
      API_ARGS=(--api-server-count 8 --renderer-num-workers 1 --host 0.0.0.0 --port "$PORT")
    else
      API_ARGS=(--data-parallel-start-rank "$(( (NODE-85)*8 ))" --headless)
    fi
    ;;
esac
CACHE="$WORK/cache/$SUFFIX"
MC_CONFIG="$WORK/config/mooncake-$SUFFIX.json"
mkdir -p "$CACHE"/{cuda,triton,tilelang,deep_ep}
printf '%s\n' "$(date -u +%Y%m%dT%H%M%S)-$$" > "$CACHE/tokenizer-epoch"
if [ "$APIS" -gt 0 ]; then
  docker volume create --driver local --opt type=tmpfs --opt device=tmpfs --opt o=size=34359738368,mode=0700 "$TOKEN_VOLUME" >/dev/null
  EXTRA_MOUNTS+=(-v "$TOKEN_VOLUME:/run/glm-tokenizer:rw")
  EXTRA_ENV+=(-e VLLM_GLM53_TOKENIZER_SHARED_PATH=/run/glm-tokenizer/segments.sqlite3
    -e VLLM_GLM53_TOKENIZER_SHARED_GIB="$SHARED_GIB" -e VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB=64
    -e VLLM_API_PRIVATE_SOCKET_DIR=/run/glm-tokenizer)
fi
cat > "$MC_CONFIG" <<JSON
{"metadata_server":"http://11.5.0.81:54396/metadata","master_server_address":"11.5.0.81:54395","protocol":"rdma","device_name":"$KV_HCAS","mode":"standalone-store","global_segment_size":0,"local_buffer_size":$BUFFER,"enable_offload":false,"tenant_id":"default"}
JSON
PLACEMENT='{"11.5.0.81:15495":1,"11.5.0.83:15495":1,"11.5.0.84:15495":1,"11.5.0.85:15495":3,"11.5.0.86:15495":3,"11.5.0.87:15495":3,"11.5.0.88:15495":3}'
PREFERRED=''
if [ "$ROLE" = prefill ]; then PREFERRED=",\"preferred_segment\":\"$IP:15495\""; fi
KV_CONFIG=$(cat <<JSON
{"kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$ENGINE","kv_buffer_device":"cuda","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"$PD_ROLE","kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"$KV_HCAS"}},{"kv_connector":"MooncakeStoreConnector","kv_role":"$STORE_ROLE","kv_connector_extra_config":{"cache_prefix":"$CACHE_PREFIX","load_async":true,"lookup_async":true,"store_compact_group_payloads":true,"remote_store_read_batch_bytes":67108864,"placement_segments":$PLACEMENT$PREFERRED}}]}}
JSON
)
DP_IP="$IP"; if [ "$ROLE" = decode ]; then DP_IP=11.5.0.85; fi
docker run -d --init --name "$PROJECT-$SUFFIX" --pull=never --network host \
  --log-opt max-size=128m --log-opt max-file=5 --label ai-infra.project="$PROJECT" \
  --device /dev/infiniband --ipc host --cap-add IPC_LOCK --cap-add SYS_NICE \
  --ulimit memlock=-1 --ulimit stack=67108864 --gpus all --pid host \
  --security-opt seccomp=unconfined -e NO_PROXY='*' -e no_proxy='*' \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e WITH_NVIDIA_PEERMEM=false \
  -e MC_TCP_BIND_ADDRESS="$IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$IP" \
  -e MC_TE_FILTERS="$KV_HCAS" -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_HOST_IP="$IP" -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_SERVER_DEV_MODE=1 -e VLLM_KV_CACHE_LAYOUT=BLHNC \
  -e MOONCAKE_CONFIG_PATH=/etc/vllm/mooncake.json -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_MOONCAKE_LOAD_RECV_THREADS="$RECV_THREADS" \
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 -e NCCL_IB_GID_INDEX=3 -e MC_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME="$NET_IF" -e GLOO_SOCKET_IFNAME="$NET_IF" \
  -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_IB_HCA="$EP_HCAS" \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL="$P2P" -e NCCL_SHM_DISABLE=1 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e PYTHONHASHSEED=0 -e VLLM_GLM53_TOKENIZER_CACHE_GIB="$CACHE_GIB" \
  -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/root/.cache/tokenizer-epoch \
  "${EXTRA_ENV[@]}" "${EXTRA_MOUNTS[@]}" \
  -v "$MODEL_DIR:/model:ro" -v "$CACHE:/root/.cache:rw" \
  -v "$CACHE/cuda:/root/.nv/ComputeCache:rw" -v "$CACHE/triton:/root/.triton/cache:rw" \
  -v "$CACHE/tilelang:/root/.tilelang:rw" -v "$CACHE/deep_ep:/root/.deep_ep:rw" \
  -v "$MC_CONFIG:/etc/vllm/mooncake.json:ro" --entrypoint vllm "$IMAGE" serve /model \
  --served-model-name glm-5.3-flash-nvfp4 --api-key EMPTY --language-model-only \
  --device-ids 0,1,2,3,4,5,6,7 --tensor-parallel-size 1 \
  --data-parallel-size "$DP" --data-parallel-size-local 8 --data-parallel-backend mp \
  --data-parallel-address "$DP_IP" --data-parallel-rpc-port "$RPC" \
  --enable-expert-parallel --enable-ep-weight-filter --all2all-backend "$ALL2ALL" \
  --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1 --max-model-len -1 \
  --max-num-seqs "$SEQS" --max-num-batched-tokens "$BATCH_TOKENS" --gpu-memory-utilization "$UTIL" \
  --kv-cache-dtype fp8 --block-size 64 --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mamba-cache-mode align --mamba-ssm-cache-dtype float32 --no-disable-hybrid-kv-cache-manager \
  --enable-chunked-prefill --enable-prefix-caching --prefix-match-unit 256 \
  --kv-transfer-config "$KV_CONFIG" --reasoning-parser glm47 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --enable-prompt-tokens-details --enable-force-include-usage \
  --safetensors-load-strategy lazy --enable-cumem-allocator --no-scheduler-reserve-full-isl \
  --seed 42 --compilation-config "$COMPILATION" \
  "${API_ARGS[@]}"
~~~

## API网关：81、83、84、85

~~~bash
if [ "$NODE" = 85 ]; then PORT=32111; CONTROL=31011; APIS=8
else PORT=32110; CONTROL=31010; APIS=4; fi
SOCKETS=()
for ((i=0;i<APIS;i++)); do SOCKETS+=(--api-socket "/run/glm-tokenizer/api-$i.sock"); done
docker run -d --init --name "$PROJECT-api-router-$NODE" --pull=never --network host \
  --label ai-infra.project="$PROJECT" --memory 8g --cpus 4 --pids-limit 128 \
  -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=1 \
  -v "$TOKEN_VOLUME:/run/glm-tokenizer:rw" --entrypoint python3 "$ROUTER_IMAGE" \
  /opt/glm-cache-repair/glm_api_cache_router.py --port "$PORT" \
  --control-url "http://127.0.0.1:$CONTROL" \
  --shared-cache-path /run/glm-tokenizer/segments.sqlite3 "${SOCKETS[@]}"
~~~

## 外层Router：81

kv_tokens取本次server_info实测值，max_seqs=32。

~~~bash
cat > "$WORK/config/router-token-balance.json" <<'JSON'
{"tokenizer_path":"/model/tokenizer.json","kv_fraction":0.85,"workers":{"http://11.5.0.81:32110":{"kv_tokens":4630572,"max_seqs":32},"http://11.5.0.83:32110":{"kv_tokens":4630572,"max_seqs":32},"http://11.5.0.84:32110":{"kv_tokens":4630572,"max_seqs":32},"http://11.5.0.85:32111":{"kv_tokens":5763341,"max_seqs":32}}}
JSON
docker run -d --init --name "$PROJECT-router-81" --pull=never --network host \
  --log-opt max-size=128m --log-opt max-file=5 --label ai-infra.project="$PROJECT" \
  --memory 64g -e NO_PROXY='*' -e no_proxy='*' -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e NVIDIA_VISIBLE_DEVICES=void -e OPENAI_API_KEY=EMPTY \
  -e VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG=/run/token-balance.json \
  -v "$MODEL_DIR:/model:ro" -v "$WORK/config/router-token-balance.json:/run/token-balance.json:ro" \
  --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
  --host 0.0.0.0 --port 38080 --prometheus-host 0.0.0.0 --prometheus-port 38081 \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill http://11.5.0.81:32110 27600 --prefill http://11.5.0.83:32110 27600 \
  --prefill http://11.5.0.84:32110 27600 --decode http://11.5.0.85:32111 \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --balance-abs-threshold 4 --balance-rel-threshold 1.5 --intra-node-data-parallel-size 1 \
  --disable-retries --worker-startup-check-interval 5 --api-key EMPTY \
  --request-timeout-secs 1800 --worker-startup-timeout-secs 7200
~~~

## 服务入口

http://11.5.0.81:38080/v1

## 镜像构建

vLLM仓库根目录：

~~~bash
docker build --network none -f integrations/glm53flash/Dockerfile.cache-repair -t local/glm53flash-vllm:0.32.0 .
~~~

Router仓库根目录：

~~~bash
docker build --network none -f Dockerfile.glm53flash-cache-repair -t local/glm53flash-router:0.1.16 .
~~~
