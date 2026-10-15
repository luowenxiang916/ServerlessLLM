#!/bin/bash
REPO="$(cd "$(dirname "$0")"; pwd)"
echo "==> 停止 sllm-store / sllm api 进程"
pkill -f "sllm-store start" 2>/dev/null || true
pkill -f "sllm start" 2>/dev/null || true
echo "==> ray stop"
"$REPO/.venv-head/bin/ray" stop --force 2>/dev/null || true
echo "done"
