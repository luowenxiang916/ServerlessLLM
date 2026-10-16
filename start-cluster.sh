#!/bin/bash
# ServerlessLLM 裸机集群启动脚本 (RTX 5090, uv venv 版)
# [非当前使用] 已被单机单 worker 的 start-single.sh 取代, 此脚本仅作参考保留。
# 当前请用: GPUS="4,5,6,7" MODEL_FOLDER=$HOME/models ./start-single.sh
# 单机多 worker: 每个 worker 用独立回环 IP(127.0.0.N), store 监听该 IP:8073,
# 以满足 sllm StoreManager 的 "<node_ip>:8073" 约定。
# 用法: GPUS="4 5 6 7" ./start-cluster.sh
set -e

REPO="$(cd "$(dirname "$0")"; pwd)"
VENV_HEAD="$REPO/.venv-head"
VENV_WORKER="$REPO/.venv-worker"

GPUS="${GPUS:-4 5 6 7}"                 # 要用的物理卡号
MODEL_FOLDER="${MODEL_FOLDER:-$HOME/models}"
# 本地已有模型目录: 部署时优先在这里找 <org>/<model>, 找到就不去 HF 下载
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/ssd/nfs/models}"
export LOCAL_MODEL_ROOT
RAY_PORT="${RAY_PORT:-6379}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8343}"
STORE_PORT="${STORE_PORT:-8073}"        # 每个 worker 各自 IP 上都用 8073
MEM_POOL="${MEM_POOL:-8GB}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
# 本机互联不要走代理
export no_proxy="127.0.0.0/8,localhost,$no_proxy"
export NO_PROXY="127.0.0.0/8,localhost,$NO_PROXY"

LOGDIR="$REPO/cluster-logs"
mkdir -p "$LOGDIR" "$MODEL_FOLDER"

echo "==> 清理旧 ray 进程"
"$VENV_HEAD/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 2

echo "==> 启动 head (ray head @ $HEAD_IP:$RAY_PORT, 不占 GPU)"
(
  source "$VENV_HEAD/bin/activate"
  CUDA_VISIBLE_DEVICES="" ray start --head --node-ip-address="$HEAD_IP" \
    --port="$RAY_PORT" --num-gpus=0 --resources='{"control_node": 1}' >/dev/null
)

echo "==> 各 GPU 挂 worker 节点 + 启动 sllm-store"
for i in $GPUS; do
  WIP="127.0.0.$i"                      # 该 worker 的独立回环 IP
  # 1) ray worker 节点(绑独立 IP)
  (
    source "$VENV_WORKER/bin/activate"
    CUDA_VISIBLE_DEVICES="$i" ray start --address="$HEAD_IP:$RAY_PORT" \
      --node-ip-address="$WIP" --num-gpus=1 \
      --resources='{"worker_node": 1, "worker_id_'"$i"'": 1}' >/dev/null
  )
  # 2) sllm-store 监听该 worker IP 的 8073
  (
    source "$VENV_WORKER/bin/activate"
    CUDA_VISIBLE_DEVICES="$i" nohup sllm-store start \
      --storage-path="$MODEL_FOLDER" --mem-pool-size="$MEM_POOL" \
      --host="$WIP" --port="$STORE_PORT" \
      > "$LOGDIR/store_$i.log" 2>&1 &
  )
  echo "    GPU $i -> ray node $WIP, store $WIP:$STORE_PORT (log: cluster-logs/store_$i.log)"
done

sleep 4
echo "==> 启动 sllm API (head)"
(
  source "$VENV_HEAD/bin/activate"
  export LLM_SERVER_URL="http://127.0.0.1:$API_PORT"
  CUDA_VISIBLE_DEVICES="" nohup sllm start --host="$API_HOST" --port="$API_PORT" \
    > "$LOGDIR/sllm_api.log" 2>&1 &
)
echo "    sllm API @ http://$API_HOST:$API_PORT (log: cluster-logs/sllm_api.log)"

echo
echo "集群已拉起 (GPU: $GPUS)。"
echo "部署模型 (在 head venv):"
echo "  source $VENV_HEAD/bin/activate"
echo "  export LLM_SERVER_URL=http://127.0.0.1:$API_PORT NO_PROXY=127.0.0.1,localhost"
echo "  sllm deploy --model Qwen/Qwen3-0.6B --backend transformers"
