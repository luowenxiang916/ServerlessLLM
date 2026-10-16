<p align="center">
  <picture>
    <img src="./docs/images/serverlessllm.jpg" alt="ServerlessLLM" width="30%">
  </picture>
</p>

<h1 align="center">ServerlessLLM</h1>

<p align="center">
  <strong>Load models 10x faster. Serve 10 models with 1 GPU.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/serverless-llm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/serverless-llm?logo=pypi&logoColor=white&label=PyPI&color=3775A9"></a>
  <a href="https://pypi.org/project/serverless-llm/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/serverless-llm?logo=pypi&logoColor=white&label=Downloads&color=3775A9"></a>
  <a href="https://discord.gg/AEF8Gduvm8"><img alt="Discord" src="https://img.shields.io/discord/1233345500112224279?logo=discord&logoColor=white&label=Discord&color=5865F2"></a>
  <a href="./docs/images/wechat.png"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
</p>

<p align="center">
  <a href="https://serverlessllm.github.io"><b>Docs</b></a> •
  <a href="#-quick-start-90-seconds"><b>Quick Start</b></a> •
  <a href="https://www.usenix.org/system/files/osdi24-fu.pdf"><b>OSDI'24 Paper</b></a>
</p>

---

## ⚡ Performance

<!-- <p align="center">
  <img src="./docs/images/benchmark_loading_speed.png" alt="Loading Speed Comparison" width="80%">
</p> -->

**ServerlessLLM loads models 6-10x faster than SafeTensors**, enabling true serverless deployment where multiple models efficiently share GPU resources.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Scenario</th>
      <th>SafeTensors</th>
      <th>ServerlessLLM</th>
      <th>Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen/Qwen3-32B</td>
      <td>Random</td>
      <td>20.6s</td>
      <td>3.2s</td>
      <td><strong>6.40x</strong></td>
    </tr>
    <tr>
      <td>Cached</td>
      <td>12.5s</td>
      <td>1.3s</td>
      <td><strong>9.95x</strong></td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-R1-Distill-Qwen-32B</td>
      <td>Random</td>
      <td>19.1s</td>
      <td>3.2s</td>
      <td><strong>5.93x</strong></td>
    </tr>
    <tr>
      <td>Cached</td>
      <td>10.2s</td>
      <td>1.2s</td>
      <td><strong>8.58x</strong></td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>Random</td>
      <td>4.4s</td>
      <td>0.7s</td>
      <td><strong>6.54x</strong></td>
    </tr>
  </tbody>
</table>

*Results obtained on NVIDIA H100 GPUs with NVMe SSD. "Random" simulates serverless multi-model serving; "Cached" shows repeated loading of the same model.*

## What is ServerlessLLM?

ServerlessLLM is a fast, low-cost system for deploying multiple AI models on shared GPUs, with three core innovations:

1. **⚡ Ultra-Fast Checkpoint Loading**: Custom storage format with O_DIRECT I/O loads models 6-10x faster than state-of-the-art checkpoint loaders
2. **🔄 GPU Multiplexing**: Multiple models share GPUs with fast switching and intelligent scheduling
3. **🎯 Unified Inference + Fine-Tuning**: Seamlessly integrates LLM serving with LoRA fine-tuning on shared resources

**Result:** Serve 10 models on 1 GPU, fine-tune on-demand, and serve a base model + 100s of LoRA adapters.

---

## 🚀 Quick Start (90 Seconds)

### Start ServerlessLLM Cluster

> **Don't have Docker?** Jump to [Use the Fast Loader in Your Code](#-use-the-fast-loader-in-your-code) for a Docker-free example.

```bash
# Download the docker-compose.yml file
curl -O https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/examples/docker/docker-compose.yml

# Set model storage location
export MODEL_FOLDER=/path/to/models

# Launch cluster (head node + worker with GPU)
docker compose up -d

# Wait for the cluster to be ready
docker logs -f sllm_head
```

### Deploy a Model

```bash
docker exec sllm_head /opt/conda/envs/head/bin/sllm deploy --model Qwen/Qwen3-0.6B --backend transformers
```

### Query the Model

```bash
curl http://127.0.0.1:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is ServerlessLLM?"}],
    "temperature": 0.7
  }'
```

**That's it!** Your model is now serving requests with an OpenAI-compatible API.

---

## 🖥️ Single-Machine Bare-Metal Deployment (Local Fork)

This fork runs **without Docker** on a single bare-metal server that cold-starts
multiple small models on a shared set of GPUs. The `Dockerfile` and
`docker-compose.yml` in this repo are from an earlier attempt and are **not used**
by the current workflow.

| Script | Status | Description |
| --- | --- | --- |
| `start-single.sh` | ✅ **Current** | Starts 1 Ray head (no GPU) + 1 Ray worker owning all GPUs (e.g. `4,5,6,7`), then `sllm-store` on `8073` and the SLLM API on `8343`. |
| `stop-cluster.sh` | ✅ **Current** | Stops the SLLM API, the store, and Ray. |
| `start-cluster.sh` | ⚠️ **Not currently used** | Earlier variant that launches **one Ray worker + one store per GPU**, each bound to its own loopback IP (`127.0.0.N:8073`). Kept for reference; superseded by `start-single.sh`. |

```bash
GPUS="4,5,6,7" MODEL_FOLDER=$HOME/models ./start-single.sh   # ~40s to become ready
ss -ltn | grep -E '8073|8343|6379'                           # all three must be LISTEN

export LLM_SERVER_URL=http://127.0.0.1:8343
export NO_PROXY=127.0.0.1,localhost                          # required, see notes
.venv-head/bin/sllm deploy --model Qwen/Qwen3-0.6B --backend transformers
```

**Notes**

- `NO_PROXY=127.0.0.1,localhost` is mandatory; without it local requests are captured by the proxy and fail with `502`.
- After changing any Python code (`vllm_backend.py`, `device_map_utils.py`, ...) or rebuilding the `.so`, re-run `start-single.sh` -- worker actors load code at startup.
- Only GPUs `4,5,6,7` are used; `0-3` are isolated via `CUDA_VISIBLE_DEVICES` inside `start-single.sh`.

### Web console

The API serves a small console on the same port (three columns: models, chat,
and the cold-start breakdown for the current model).

```text
http://<server>:8343/           # the console
http://<server>:8343/ui/state   # the aggregated state it polls
```

Select a model on the left, then *加载* to force (and measure) one cold start,
or simply send a message. Every `/v1/chat/completions` response carries:

| field | meaning |
| --- | --- |
| `disk_to_host_s` | SSD → pinned host memory pool (`~0` when the model is already pooled) |
| `mem_to_vram_s` | host pool → GPU |
| `disk_to_vram_s` | SSD → GPU, end to end |
| `ttft_s` | API request received → first generated token |
| `from_disk` | `true` when this request really read from storage |

All three transfer numbers are measured by the store while it pages the
weights in, so both backends report them in the same way. The store pipelines
disk -> host and host -> VRAM chunk by chunk, which shows up in the numbers:

- weights really come from SSD (`from_disk: true`): the copy to VRAM overlaps
  with the read, so `mem_to_vram_s` is close to zero and `disk_to_vram_s` is
  the number to quote;
- weights are already in the pinned pool (`from_disk: false`):
  `disk_to_host_s` is the no-op and `mem_to_vram_s` is the real host -> VRAM
  time.

Measured with Qwen3.6-35B-A3B (70 GB, TP=4) on the 4x RTX 5090 setup:

| case | `disk_to_host_s` | `mem_to_vram_s` | `disk_to_vram_s` | `ttft_s` |
| --- | --- | --- | --- | --- |
| cold start, from SSD | 79.5 | ~0.0 | 79.5 | 171.0 |
| already pooled | ~0.0 | 0.6 | 0.6 | 88.5 |

`ttft_s` contains the two transfers *plus* the per-instance cold start (Ray
actor, vLLM engine warm-up: `torch.compile`, CUDA graph capture, MoE kernel
build), which is what the remaining ~88 s goes into. Moving 70 GB to the GPUs
over PCIe takes well under a second; what actually costs time is reading it
from storage and then warming the engine up.

vLLM loads the weights inside its own EngineCore process while the metrics are
reported by the backend actor, so the store writes the stage timings of a load
to `$TMPDIR/sllm_load_timings/<model>_rank_<n>.json` and the actor reads them
back once its engine is up.

Console endpoints: `GET /ui/state`, `POST /ui/deploy`
(`{"model": "...", "backend": "transformers"}`, same defaults as `sllm deploy`),
`POST /unload` (`{"model": "..."}` -- drops the DRAM copy, keeps the SSD one)
and `GET /store/status`. The pinned pool lives inside `sllm-store` and has no
query API, so "内存驻留" is inferred by the API process.

### Two copies of every package

`sllm` and `sllm_store` are installed as **copies** in the venvs, and which copy
is imported depends on how the process was started:

| process | `sllm` | `sllm_store` |
| --- | --- | --- |
| `sllm start` (API) | `.venv-head/.../site-packages/sllm` | `.venv-head/.../site-packages/sllm_store` |
| `sllm-store start` (store) | – | `.venv-worker/.../site-packages/sllm_store` |
| Ray worker actors (backend) | repo checkout | repo checkout |

So after editing either package, sync the copies and restart:

```bash
VP=.venv-head/lib/python3.10/site-packages        # and .venv-worker/... for sllm_store
cp -rf sllm/. $VP/sllm/
cp -f sllm_store/sllm_store/{transformers,server,client}.py $VP/sllm_store/

# rebuild the C++ extension (`wait_model_in_host` etc.)
cd sllm_store/build/temp.linux-x86_64-cpython-310
CUDA_HOME=/usr/local/cuda-13.2 cmake --build . --target _checkpoint_store -j
cp -f ../lib.linux-x86_64-cpython-310/sllm_store/_checkpoint_store*.so \
      ../../../$VP/sllm_store/
```

---

## 💡 Use the Fast Loader in Your Code

Use ServerlessLLM Store standalone to speed up torch-based model loading.

### Install

```bash
pip install serverless-llm-store
```

### Convert a Model

```bash
sllm-store save --model Qwen/Qwen3-0.6B --backend transformers
```

### Start the Store Server

```bash
# Start the store server first
sllm-store start --storage-path ~/models --mem-pool-size 4GB
```

### Load it 6-10x Faster in Your Python Code

```python
from sllm_store.transformers import load_model

# Load model (6-10x faster than from_pretrained!)
model = load_model(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    torch_dtype="float16"
)

# Use as a normal PyTorch/Transformers model
output = model.generate(**inputs)
```

**How it works:**
- Custom binary format optimized for sequential reads
- O_DIRECT I/O bypassing OS page cache
- Pinned memory pool for DMA-accelerated GPU transfers
- Parallel multi-threaded loading

---

## 🎯 Key Features

### ⚡ Ultra-Fast Model Loading
- **6-10x faster** than the SafeTensors checkpoint loader
- Supports both NVIDIA and AMD GPUs
- Works with vLLM, Transformers, and custom models

**📖 Docs:** [Fast Loading Guide](https://serverlessllm.github.io/docs/store/quickstart) | [ROCm Guide](https://serverlessllm.github.io/docs/store/rocm_quickstart)

---

### 🔄 GPU Multiplexing
- **Run 10+ models on 1 GPU** with fast switching
- Storage-aware scheduling minimizes loading time
- Auto-scale instances per model (scale to zero when idle)
- Live migration for zero-downtime resource optimization

**📖 Docs:** [Deployment Guide](https://serverlessllm.github.io/docs/getting_started)

---

### 🎯 Unified Inference + LoRA Fine-Tuning
- Integrates LLM serving with serverless LoRA fine-tuning
- Deploys fine-tuned adapters for inference on-demand
- Serves a base model + 100s of LoRA adapters efficiently

**📖 Docs:** [Fine-Tuning Guide](https://serverlessllm.github.io/docs/features/peft_lora_fine_tuning)

---

### 🔍 Embedding Models for RAG
- Deploy embedding models alongside LLMs
- Provides an OpenAI-compatible `/v1/embeddings` endpoint

**💡 Example:** [RAG Example](https://github.com/ServerlessLLM/ServerlessLLM/tree/main/examples/embedding)

---

### 🚀 Production-Ready
- **OpenAI-compatible API** (drop-in replacement)
- Docker and Kubernetes deployment
- Multi-node clusters with distributed scheduling

**📖 Docs:** [Deployment Guide](https://serverlessllm.github.io/docs/developer/supporting_a_new_hardware) | [API Reference](https://serverlessllm.github.io/docs/api/intro)

---

### 💻 Supported Hardware
- **NVIDIA GPUs**: Compute capability 7.0+ (V100, A100, H100, RTX 3060+)
- **AMD GPUs**: ROCm 6.2+ (MI100, MI200 series) - Experimental

**More Examples:** [./examples/](./examples/)

---

## 🤝 Community

- **Discord**: [Join our community](https://discord.gg/AEF8Gduvm8) - Get help, share ideas
- **GitHub Issues**: [Report bugs](https://github.com/ServerlessLLM/ServerlessLLM/issues)
- **WeChat**: [QR Code](./docs/images/wechat.png) - 中文支持
- **Contributing**: See [CONTRIBUTING.md](./CONTRIBUTING.md)

Maintained by 10+ contributors worldwide. Community contributions are welcome!

---

## 📄 Citation

If you use ServerlessLLM in your research, please cite our [OSDI'24 paper](https://www.usenix.org/system/files/osdi24-fu.pdf):

```bibtex
@inproceedings{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={OSDI'24},
  year={2024}
}
```

---

## 📝 License

Apache 2.0 - See [LICENSE](./LICENSE)

---

<p align="center">
  <strong>⭐ Star this repo if ServerlessLLM helps you!</strong>
</p>
