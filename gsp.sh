#!/bin/bash

echo "params: $@"
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

export NO_PROXY="127.0.0.1,localhost"

MODEL_TAG="$1"     
MODEL_SAFE="$2"    
MAX_REQ="$3"
MEM_FRAC="$4"
NGPUS="$5"
ORDER="$6"
POLICY="$7"
RLT="$8"
GRP="$9"
PG="${10}"
PRATIO="${11}"      
LENS_SAFE="${12}" 
OUTPUT_LEN="${13}"
RATE="${14}"


LENS="${LENS_SAFE//-/,}" 

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPUS-1)))
mapfile -t GPUS < <(seq 0 $((NGPUS-1)))

PORT_BASE=30000

EXTRA_MAX_REQ=()
if [[ -n "${MAX_REQ}" && "${MAX_REQ,,}" != "none" ]]; then
  EXTRA_MAX_REQ=(--max-running-requests "${MAX_REQ}")
fi

EXTRA_MEM=()
if [[ -n "${MEM_FRAC}" && "${MEM_FRAC,,}" != "none" ]]; then
  EXTRA_MEM=(--mem-fraction-static "${MEM_FRAC}")
fi

EXTRA_LRT=()
case "${RLT,,}" in
  1|true|yes|on) EXTRA_LRT=(--enable-lrt) ;;
  *) ;;
esac

File="GSP_${MODEL_SAFE}_${MAX_REQ}_${MEM_FRAC}_${NGPUS}_${ORDER}_${POLICY}_${RLT}_${GRP}_${PG}_${PRATIO}_${LENS_SAFE}_${OUTPUT_LEN}_${RATE}.jsonl"
OUTFILE="HR_GSP_${MODEL_SAFE}_${MAX_REQ}_${MEM_FRAC}_${NGPUS}_${ORDER}_${POLICY}_${RLT}_${GRP}_${PG}_${PRATIO}_${LENS_SAFE}_${OUTPUT_LEN}_${RATE}.jsonl"

pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

readarray -t PORTS < <(seq "$PORT_BASE" "$((PORT_BASE + NGPUS - 1))")
URLS=()
for i in "${!GPUS[@]}"; do
  URLS+=("http://127.0.0.1:$((PORT_BASE + i))")
done

ROUTER_PORT=$((PORT_BASE + NGPUS))


worker_pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"; port=$((PORT_BASE + i))
  echo "[BOOT] worker idx=${i} gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m sglang.launch_server \
    --model-path "${MODEL_TAG}" \
    --host 127.0.0.1 \
    --port "${port}" \
    "${EXTRA_MEM[@]}" \
    "${EXTRA_MAX_REQ[@]}" \
    "${EXTRA_LRT[@]}" \
    > >(stdbuf -oL -eL tee -a "worker_${i}.log") 2>&1 &
  worker_pids+=($!)
done


wait_health() {
  local port=$1
  for _ in {1..1000}; do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null && return 0
    sleep 1
  done
  return 1
}

health_pids=()
for i in "${!PORTS[@]}"; do
  wait_health "${PORTS[$i]}" & health_pids+=($!)
done
for i in "${!health_pids[@]}"; do
  if ! wait "${health_pids[$i]}"; then
    echo "Worker on port ${PORTS[$i]} not ready"
    tail -n 120 "worker_${i}.log" || true
    exit 1
  fi
done

python -m sglang_router.launch_router \
  --worker-urls "${URLS[@]}" \
  --host 127.0.0.1 \
  --port "${ROUTER_PORT}" \
  --policy "${POLICY}" \
  > >(stdbuf -oL -eL tee -a router.log) 2>&1 &
router_pid=$!

sleep 10

python -m sglang.bench_serving \
  --backend sglang \
  --base-url "http://127.0.0.1:${ROUTER_PORT}" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups "${GRP}" \
  --gsp-prompts-per-group "${PG}" \
  --gsp-prefix-ratio "${PRATIO}" \
  --gsp-total-lens "${LENS}" \
  --gsp-output-len "${OUTPUT_LEN}" \
  --request-rate "${RATE}" \
  --gsp-order "${ORDER}"  \
  --output-file "${File}" \
  --flush-cache

sum_c=0; sum_p=0
json_per=""

for port in "${PORTS[@]}"; do
  metrics=$(curl -s "http://127.0.0.1:${port}/metrics")
  raw_c_line=$(awk '/^sglang:cached_tokens_total\{/{print; exit}' <<<"$metrics")
  raw_p_line=$(awk '/^sglang:prompt_tokens_total\{/{print; exit}' <<<"$metrics")
  echo "----- port ${port} -----"
  echo "$raw_c_line"
  echo "$raw_p_line"

  c=$(awk '/^sglang:cached_tokens_total\{/{print $2}' <<<"$metrics")
  p=$(awk '/^sglang:prompt_tokens_total\{/{print $2}' <<<"$metrics")
  hr=$(awk -v c="$c" -v p="$p" 'BEGIN{ if(p>0) printf "%.6f", c/p; else printf "0"}')
  sum_c=$(awk -v a="$sum_c" -v b="$c" 'BEGIN{print a+b}')
  sum_p=$(awk -v a="$sum_p" -v b="$p" 'BEGIN{print a+b}')
  json_per+=$(printf '"%s": %s,' "$port" "$hr")
done

cluster=$(awk -v c="$sum_c" -v p="$sum_p" 'BEGIN{ if(p>0) printf "%.6f", c/p; else printf "0"}')
json_per=${json_per%,} 

printf '{ "cluster_hit_rate": %s, "per_worker": { %s } }\n' \
  "$cluster" "$json_per" > "$OUTFILE"

echo "hit rates saved to $OUTFILE"

kill "$router_pid" 2>/dev/null || true
pkill -f "sglang.launch_server" 2>/dev/null || true