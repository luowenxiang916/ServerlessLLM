# ServerlessLLM 环境说明

本文档记录当前仓库实际使用的运行环境、依赖版本、启动方式和迁移注意事项。
仓库存在两套可用环境：

- **裸机 venv 环境**：当前实际使用，适合在这台 8x RTX 5090 机器上直接运行。
- **Docker 环境**：由 `Dockerfile` 和 `docker-compose.yml` 定义，适合复现和迁移到其他机器。

两套环境的依赖版本并不完全一致，迁移时请明确选择其中一套。

最后核对日期：2026-10-19。

---

## 1. 当前裸机环境

### 1.1 主机与 GPU

| 项目 | 当前值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 内核 | 5.15.0-94-generic |
| CPU 架构 | x86_64 |
| CPU | 2 x Intel Xeon Gold 6530，共 128 线程 |
| 内存 | 1.0 TiB |
| GPU | 8 x NVIDIA GeForce RTX 5090，32 GB |
| GPU 计算能力 | `sm_120` |
| NVIDIA 驱动 | 595.58.03 |
| CUDA Toolkit | `/usr/local/cuda-13.2`，nvcc 13.2.78 |
| 其他 CUDA | `/usr/local/cuda-13.3` 也存在 |

GPU 拓扑：`0-3` 位于 NUMA 0，`4-7` 位于 NUMA 1；GPU 之间通过 PCIe/SYS 互联，没有 NVLink。
当前启动脚本默认使用 `4,5,6,7`。

### 1.2 Python 与虚拟环境

两个虚拟环境均由 `uv 0.11.32` 创建，Python 版本为 `3.10.12`，并且都启用了
`include-system-site-packages = true`。

| 项目 | `.venv-head` | `.venv-worker` |
| --- | --- | --- |
| 用途 | API、controller、scheduler | store、Ray worker、vLLM |
| Python | 3.10.12 | 3.10.12 |
| PyTorch | 2.11.0（CUDA 13.0 runtime） | 2.11.0（CUDA 13.0 runtime） |
| torchvision | 未安装 | 0.26.0+cu130 |
| torchaudio | 未安装 | 2.11.0+cu130 |
| vLLM | 未安装 | 0.26.0 |
| Ray | 2.48.0 | 2.48.0 |
| Transformers | 5.17.0 | 5.17.0 |
| Accelerate | 1.15.0 | 1.15.0 |
| PEFT | 0.21.2 | 0.21.2 |
| bitsandbytes | 未安装 | 0.50.2 |
| datasets | 未安装 | 5.1.0 |
| FastAPI | 0.115.12 | 0.136.3 |
| Pydantic | 2.11.5 | 2.14.1 |
| OpenAI SDK | 1.52.0 | 3.29.0 |
| serverless-llm | 0.8.0 | 0.8.0 |
| serverless-llm-store | 0.8.0 | 0.8.0 |

注意：

- 两个 venv 都启用了系统 site-packages，因此系统 Python 中安装的包可能被一起导入。
  这方便了当前机器使用，但不是完全隔离的环境。
- `requirements-worker.txt` 中写的是 `vllm==0.11.2`，但当前 `.venv-worker` 实际安装的是
  `vllm 0.26.0`。Docker 镜像使用的仍是 `vllm 0.11.2`。
- `requirements.txt` 中的 `serverless-llm-store` 没有固定版本；当前 venv 中实际为 0.8.0。

### 1.3 本地代码与安装方式

当前使用的是仓库源码，而不是从 PyPI 安装的独立副本：

- `sllm` 可从仓库根目录导入。
- `sllm_store` 安装在两个 venv 的 `site-packages` 中。
- Ray worker actor 会使用仓库 checkout 中的 `sllm` 代码。

因此修改 Python 代码后，需要同步 venv 中的副本并重启相关进程。具体规则见根目录
`README.md` 的 “Two copies of every package” 一节。

`sllm_store` 包含 C++/CUDA 扩展，核心产物为：

```text
.venv-head/lib/python3.10/site-packages/sllm_store/_checkpoint_store*.so
.venv-worker/lib/python3.10/site-packages/sllm_store/_checkpoint_store*.so
```

---

## 2. 目录与运行组件

| 路径 | 说明 |
| --- | --- |
| `sllm/` | ServerlessLLM API、controller、scheduler、router、后端和 Web UI |
| `sllm_store/` | C++/CUDA checkpoint store、gRPC server、Python 客户端 |
| `start-single.sh` | 当前使用的单机启动脚本，1 个 Ray head + 1 个 Ray worker 管理多张 GPU |
| `stop-cluster.sh` | 停止 API、store 和 Ray |
| `start-cluster.sh` | 旧版启动脚本，每个 GPU 一个 worker/store，当前不再使用 |
| `cluster-logs/` | 运行日志，已被 Git 忽略 |
| `Dockerfile` | Docker 镜像定义 |
| `docker-compose.yml` | 1 head + 4 worker 的 Compose 配置 |
| `.env` | 本地模型目录配置，已被 Git 忽略 |
| `configs/` | 示例配置 |

默认端口：

| 端口 | 用途 |
| --- | --- |
| `6379` | Ray head |
| `8073` | `sllm-store` gRPC |
| `8343` | ServerlessLLM API 和 Web 控制台 |

---

## 3. 裸机启动方式

### 3.1 启动集群

```bash
cd /home/LuoWenxiang/ServerlessLLM

GPUS="4,5,6,7" \
MODEL_FOLDER="$HOME/models" \
LOCAL_MODEL_ROOT="/ssd/nfs/models" \
./start-single.sh
```

脚本会完成以下动作：

1. 启动 Ray head，不占用 GPU。
2. 启动 1 个 Ray worker，通过 `CUDA_VISIBLE_DEVICES` 管理指定 GPU。
3. 启动 `sllm-store`，监听 `8073`。
4. 启动 `sllm` API，监听 `8343`。
5. 将日志写入 `cluster-logs/`。

检查端口：

```bash
ss -ltn | grep -E '6379|8073|8343'
```

### 3.2 部署和请求模型

```bash
export LLM_SERVER_URL=http://127.0.0.1:8343
export NO_PROXY=127.0.0.1,localhost

.venv-head/bin/sllm deploy \
  --model Qwen/Qwen3-0.6B \
  --backend transformers
```

`NO_PROXY` 必须包含 `127.0.0.1,localhost`，否则本地请求可能被代理捕获并返回 `502`。

### 3.3 停止集群

```bash
./stop-cluster.sh
```

### 3.4 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GPUS` | `4,5,6,7` | 逗号分隔的物理 GPU 编号 |
| `MODEL_FOLDER` | `$HOME/models` | store 的模型目录 |
| `LOCAL_MODEL_ROOT` | `/ssd/nfs/models` | 优先查找本地模型的根目录 |
| `RAY_PORT` | `6379` | Ray head 端口 |
| `API_PORT` | `8343` | API 端口 |
| `STORE_PORT` | `8073` | store 端口 |
| `MEM_POOL` | `16GB` | store 的 pinned memory pool |
| `CUDA_HOME` | `/usr/local/cuda-13.2` | CUDA Toolkit 路径 |
| `HF_ENDPOINT` | 脚本内默认值 | Hugging Face 镜像地址 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 脚本内默认值 | 模型下载代理，迁移时请覆盖 |

如果新机器的 CUDA 版本或代理地址不同，直接通过环境变量覆盖，不要修改脚本中的默认值。

---

## 4. 从源码复现裸机环境

以下步骤用于在新机器上重建与当前机器接近的环境。实际版本请以第 1.2 节为准。

### 4.1 系统依赖

需要：

- Ubuntu 22.04 或兼容的 Linux 发行版
- NVIDIA 驱动，Blackwell 建议 570 以上
- CUDA Toolkit 13.2（或与 PyTorch wheel 匹配的版本）
- Python 3.10
- `build-essential`、`cmake>=3.20`、`ninja-build`、`pkg-config`
- `uv 0.11.32` 或兼容版本

### 4.2 创建 head 环境

```bash
cd /home/LuoWenxiang/ServerlessLLM

uv venv --python 3.10 --system-site-packages .venv-head
source .venv-head/bin/activate

uv pip install -r requirements.txt
uv pip install -e .
```

### 4.3 创建 worker 环境

```bash
uv venv --python 3.10 --system-site-packages .venv-worker
source .venv-worker/bin/activate

uv pip install -r requirements-worker.txt
uv pip install -e ".[worker]"
```

### 4.4 编译并安装 `sllm_store`

在两个 venv 中都需要安装本地编译的 `sllm_store`。CUDA 架构列表应至少包含目标 GPU：

```bash
source .venv-head/bin/activate
cd sllm_store
CUDA_HOME=/usr/local/cuda-13.2 \
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" \
uv pip install --no-build-isolation .
cd ..

source .venv-worker/bin/activate
cd sllm_store
CUDA_HOME=/usr/local/cuda-13.2 \
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" \
uv pip install --no-build-isolation .
cd ..
```

`8.9` 覆盖 RTX 6000 Ada，`12.0` 覆盖 RTX 5090 和 RTX PRO 6000 Blackwell。
如果只面向其中一种 GPU，可以缩小架构列表以缩短编译时间。

### 4.5 应用 vLLM patch

worker 环境需要应用 `sllm_store/vllm_patch`：

```bash
source .venv-worker/bin/activate
./sllm_store/vllm_patch/patch.sh
./sllm_store/vllm_patch/check_patch.sh
```

当前 patch 针对 `vllm==0.11.2` 编写。升级 vLLM 后需要重新检查 patch 是否仍然适用。
当前 `.venv-worker` 的 vLLM 0.26.0 环境中已经存在 `sllm_loader.py`，说明 patch 已应用。

---

## 5. Docker 环境

### 5.1 镜像组成

`Dockerfile` 使用两阶段构建：

| 阶段 | 基础镜像 | 作用 |
| --- | --- | --- |
| builder | `nvidia/cuda:12.8.1-devel-ubuntu22.04` | 编译 `sllm_store` 和 `serverless-llm` wheel |
| runner | `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` | 运行 head/worker |

关键配置：

- 构建阶段 Python 版本：3.10
- `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"`（`Dockerfile:49`）
- 构建阶段 PyTorch：2.9.1
- Docker 中 vLLM：0.11.2
- 创建 `head` 和 `worker` 两个 conda 环境
- `entrypoint.sh` 通过 `MODE=HEAD` 或 `MODE=WORKER` 决定启动角色

该架构列表同时覆盖：

- `sm_89`：RTX 6000 Ada
- `sm_120`：RTX 5090、RTX PRO 6000 Blackwell

### 5.2 构建和启动

```bash
export MODEL_FOLDER=/path/to/models
docker compose build
docker compose up -d
docker logs -f sllm_head
```

当前 `docker-compose.yml`：

- 镜像名为 `serverlessllm/sllm:5090`
- 启动 1 个 head 和 4 个 worker
- worker 固定使用 GPU `4,5,6,7`
- 每个 worker 的 store memory pool 为 `8GB`

迁移到其他机器时，需要修改 `device_ids`、镜像标签和 `--mem-pool-size`。

### 5.3 Docker 与裸机环境的差异

| 项目 | 裸机 venv | Docker |
| --- | --- | --- |
| PyTorch | 2.11.0 / CUDA 13.0 | 2.9.1 / CUDA 12.8 |
| vLLM | 0.26.0 | 0.11.2 |
| Python 环境 | `.venv-head` + `.venv-worker` | conda `head` + `worker` |
| 启动方式 | `start-single.sh` | `entrypoint.sh` + Compose |
| 适合场景 | 当前实机开发和调试 | 迁移、复现和部署 |

Docker 环境更接近可复现部署；裸机环境包含当前机器上较新的 PyTorch/vLLM 版本。

---

## 6. GPU 兼容性说明

### 6.1 RTX 5090 与 RTX PRO 6000 Blackwell

两者计算能力都是 `sm_120`，可以使用同一套 CUDA 12.8+、PyTorch cu128+ 环境和同一个
Docker 镜像。当前 Dockerfile 的架构列表已经包含 `12.0`。

但两者资源不同：

| GPU | 显存 | 定位 |
| --- | --- | --- |
| RTX 5090 | 32 GB | 消费级 |
| RTX PRO 6000 Blackwell | 96 GB | 工作站/服务器 |

ServerlessLLM 当前调度主要按 `num_gpus` 计数，不按显存大小区分。如果把 32 GB 和 96 GB
的卡混在同一个 worker 池中，可能出现大模型被调度到 32 GB 卡上的情况。建议：

- 将不同显存的 GPU 分成不同 worker 池；或
- 扩展调度器，使其把显存纳入资源模型。

### 6.2 RTX 6000 Ada

RTX 6000 Ada 是 `sm_89`，与 5090 的 `sm_120` 不同。当前 Dockerfile 同时编译了
`8.9` 和 `12.0`，因此同一个镜像可以覆盖两者；但如果自行编译 `sllm_store`，必须确认
`TORCH_CUDA_ARCH_LIST` 中包含 `8.9`。

### 6.3 验证 GPU

```bash
nvidia-smi

.venv-worker/bin/python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.version.cuda)
PY
```

预期：

- RTX 5090 / RTX PRO 6000 Blackwell：`(12, 0)`
- RTX 6000 Ada：`(8, 9)`

---

## 7. 存储与内存

- 模型默认目录：`$HOME/models`
- 本地模型优先目录：`/ssd/nfs/models`（NFS，约 3.5 TB）
- 根分区：约 434 GB
- 主机内存：1.0 TiB
- `start-single.sh` 的 store pinned pool 默认：`16GB`
- `docker-compose.yml` 的每个 worker pinned pool：`8GB`

pinned memory pool 会占用主机内存，不会直接占用 GPU 显存。模型较大或并发较高时，
需要根据主机内存和模型数量调整 `MEM_POOL`。

模型权重、转换后的 checkpoint 和日志都不应该提交到 Git 仓库。

---

## 8. 已知注意事项

1. `start-cluster.sh` 是旧版脚本，当前请使用 `start-single.sh`。
2. `start-single.sh` 默认把代理地址和 Hugging Face 镜像写死为环境变量默认值；
   换网络环境时请用外部环境变量覆盖。
3. `NO_PROXY` 必须包含 `127.0.0.1,localhost`，否则 API 本地请求可能失败。
4. 两个 venv 都启用了 system site-packages，依赖隔离不完整。
5. `requirements-worker.txt` 固定了 `vllm==0.11.2`，但当前 worker venv 是 0.26.0；
   迁移前应确认使用哪个版本，并重新应用 vLLM patch。
6. 修改 `sllm` 或 `sllm_store` 后，需要同步 venv 中的副本并重启进程。
7. 修改 `sllm_store` 的 C++/CUDA 代码后，需要重新编译 `_checkpoint_store` 扩展。
8. `docker-compose.yml` 中的 GPU ID `4,5,6,7`、镜像名和 memory pool 都是本机配置，
   迁移时必须修改。
9. `.env`、`.venv-head/`、`.venv-worker/`、`build/`、`cluster-logs/` 不应提交到 Git。

---

## 9. 迁移到 RTX PRO 6000 的检查清单

- [ ] 确认目标卡是 RTX PRO 6000 Blackwell（`sm_120`）还是 RTX 6000 Ada（`sm_89`）。
- [ ] 确认宿主机 NVIDIA 驱动版本和 CUDA Toolkit。
- [ ] 运行 `nvidia-smi` 和 PyTorch capability 检查。
- [ ] 选择使用 Docker 镜像，或在目标机器重新编译 `sllm_store`。
- [ ] 如果重新编译，确认 `TORCH_CUDA_ARCH_LIST` 包含目标架构。
- [ ] 修改 `docker-compose.yml` 中的 GPU ID，或设置 `start-single.sh` 的 `GPUS`。
- [ ] 根据显存调整模型、`num_gpus`、tensor parallel 和 memory pool。
- [ ] 确认 `MODEL_FOLDER`、`LOCAL_MODEL_ROOT` 和模型文件已同步。
- [ ] 启动后检查 `6379`、`8073`、`8343` 三个端口。
- [ ] 部署一个小模型做冒烟测试，例如 `Qwen/Qwen3-0.6B`。
- [ ] 检查 `cluster-logs/` 中是否有 CUDA、驱动或 kernel 相关错误。

---

## 10. 版本快照命令

迁移或排查问题时，可以先保存以下信息：

```bash
nvidia-smi
nvcc --version
uname -a
cat /etc/os-release

.venv-head/bin/python -V
.venv-worker/bin/python -V

.venv-head/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
.venv-worker/bin/python -c "import torch, vllm, ray; print(torch.__version__, torch.version.cuda, vllm.__version__, ray.__version__)"

.venv-head/bin/python -m pip freeze > requirements-head.lock.txt
.venv-worker/bin/python -m pip freeze > requirements-worker.lock.txt
```

生成的 lock 文件可以作为迁移参考，但不要直接提交包含本地路径或敏感信息的文件。
