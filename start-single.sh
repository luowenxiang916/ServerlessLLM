#!/bin/bash
# ServerlessLLM 单机多卡启动 (1 head + 1 worker 管多张卡)
# 用法: GPUS="4,5,6,7" ./start-single.sh
set -e

REPO="$(cd "$(dirname "$0")"; pwd)"
VENV_HEAD="$REPO/.venv-head"
VENV_WORKER="$REPO/.venv-worker"

GPUS="${GPUS:-4,5,6,7}"                       # 逗号分隔的物理卡号
NUM_GPUS="$(echo "$GPUS" | awk -F, '{print NF}')"
MODEL_FOLDER="${MODEL_FOLDER:-$HOME/models}"
# 本地已有模型目录: 部署时优先在这里找 <org>/<model>, 找到就不去 HF 下载
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/ssd/nfs/models}"
export LOCAL_MODEL_ROOT
RAY_PORT="${RAY_PORT:-6379}"
HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8343}"
STORE_PORT="${STORE_PORT:-8073}"
MEM_POOL="${MEM_POOL:-16GB}"
WORKER_ID="${WORKER_ID:-0}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export no_proxy="127.0.0.0/8,localhost,$HEAD_IP,10.0.0.0/8,$no_proxy"
export NO_PROXY="127.0.0.0/8,localhost,$HEAD_IP,10.0.0.0/8,$NO_PROXY"

# 模型下载走代理 + HF 镜像(worker 侧 ray actor 会继承这些)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HTTP_PROXY="${HTTP_PROXY:-http://10.33.33.21:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://10.33.33.21:7897}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

LOGDIR="$REPO/cluster-logs"
mkdir -p "$LOGDIR" "$MODEL_FOLDER"

echo "==> 清理旧进程 (sllm api / store / ray)"
for pid in $(ps -eo pid,cmd | grep -E "sllm start --host|sllm-store start" | grep -v grep | awk '{print $1}'); do
  kill "$pid" 2>/dev/null || true
done
"$VENV_HEAD/bin/ray" stop --force >/dev/null 2>&1 || true
sleep 3

echo "==> 启动 head (ray head @ $HEAD_IP:$RAY_PORT, 不占 GPU)"
(
  source "$VENV_HEAD/bin/activate"
  CUDA_VISIBLE_DEVICES="" ray start --head --node-ip-address="$HEAD_IP" \
    --port="$RAY_PORT" --num-gpus=0 --resources='{"control_node": 1}' >/dev/null
)

echo "==> 启动 1 个 worker 节点, 管 GPU [$GPUS] ($NUM_GPUS 张)"
(
  source "$VENV_WORKER/bin/activate"
  CUDA_VISIBLE_DEVICES="$GPUS" ray start --address="$HEAD_IP:$RAY_PORT" \
    --num-gpus="$NUM_GPUS" \
    --resources='{"worker_node": 1, "worker_id_'"$WORKER_ID"'": 1}' >/dev/null
)

echo "==> 启动 sllm-store (管这 $NUM_GPUS 张卡, @ $HEAD_IP:$STORE_PORT)"
(
  source "$VENV_WORKER/bin/activate"
  CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup sllm-store start \
    --storage-path="$MODEL_FOLDER" --mem-pool-size="$MEM_POOL" \
    --host="0.0.0.0" --port="$STORE_PORT" \
    > "$LOGDIR/store.log" 2>&1 < /dev/null &
)

sleep 15
echo "==> 启动 sllm API (head)"
(
  source "$VENV_HEAD/bin/activate"
  export LLM_SERVER_URL="http://127.0.0.1:$API_PORT"
  CUDA_VISIBLE_DEVICES="" setsid nohup sllm start --host="$API_HOST" --port="$API_PORT" \
    > "$LOGDIR/sllm_api.log" 2>&1 < /dev/null &
)
echo "    sllm API @ http://$API_HOST:$API_PORT"

echo
echo "单机多卡集群已拉起 (GPU: $GPUS, 共 $NUM_GPUS 张, 1 worker)。"
echo "部署模型 (在 head venv):"
echo "  source $VENV_HEAD/bin/activate"
echo "  export LLM_SERVER_URL=http://127.0.0.1:$API_PORT NO_PROXY=127.0.0.1,localhost"
echo "  sllm deploy --model Qwen/Qwen3-0.6B --backend transformers"
