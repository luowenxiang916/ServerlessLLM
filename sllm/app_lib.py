# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import ray
import ray.exceptions
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sllm.logger import init_logger

logger = init_logger(__name__)

UI_INDEX = Path(__file__).parent / "ui" / "index.html"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "cli" / "default_config.json"

# The pinned memory pool lives inside the store daemon and is not queryable,
# so the API keeps a best-effort view of what is cached in host memory: a
# finished generation proves the model was resident (either freshly read from
# disk or already pooled), and /unload drops it.
_host_resident = {}
_last_metrics = {}


def _default_model_config(model_name, backend="transformers"):
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        config = json.load(f)
    config["model"] = model_name
    config["backend"] = backend
    config.setdefault("backend_config", {})["pretrained_model_name_or_path"] = (
        model_name
    )
    return config


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connect to the Ray cluster
        # ray.init()
        yield
        # Shutdown the Ray cluster
        ray.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        body = await request.json()

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            await controller.register.remote(body)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Cannot register model, please contact the administrator",
            )

        return {"status": "ok"}

    @app.post("/update")
    async def update_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to update model {model_name}")
        try:
            await controller.update.remote(model_name, body)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": f"updated model {model_name}"}

    @app.post("/delete")
    async def delete_model(request: Request):
        body = await request.json()

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )
        lora_adapters = body.get("lora_adapters", None)

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        if lora_adapters is not None:
            logger.info(
                f"Received request to delete LoRA adapters {lora_adapters} on model {model_name}"
            )
            await controller.delete.remote(model_name, lora_adapters)
        else:
            logger.info(f"Received request to delete model {model_name}")
            await controller.delete.remote(model_name)

        return {"status": f"deleted model {model_name}"}

    @app.post("/unload")
    async def unload_model(request: Request):
        """Drop a model's host (DRAM) cache copy; the SSD copy is kept."""
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to unload model {model_name}")
        try:
            result = await controller.unload.remote(model_name)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        _host_resident.pop(model_name, None)
        return {
            "status": f"unloaded model {model_name} from host memory",
            "result": result,
        }

    @app.get("/store/status")
    async def store_status():
        """Host memory-pool / on-disk state per node."""
        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            return await controller.store_status.remote()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/ui/state")
    async def ui_state():
        """Everything the console needs in one round trip."""
        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        try:
            status = await controller.status.remote()
            store_info = await controller.store_status.remote()
            store_config = await controller.store_config.remote()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        on_disk = {}
        for node_id, node in (store_info or {}).items():
            if not isinstance(node, (list, tuple)) or not node:
                continue
            for name, info in (node[0] or {}).items():
                paths = info[0] if isinstance(info, (list, tuple)) else []
                size = None
                if isinstance(info, (list, tuple)) and len(info) > 1:
                    size = info[1]
                backend = None
                if paths:
                    backend = str(paths[0]).split("/")[0]
                on_disk[name] = {
                    "size": size,
                    "backend": backend,
                    "node": node_id,
                }

        models = []
        seen = set()
        for model in status.get("models", []):
            name = model.get("id")
            if not name or name in seen:
                continue
            seen.add(name)
            entry = on_disk.get(name, {})
            models.append(
                {
                    "name": name,
                    "deployed": True,
                    "on_disk": name in on_disk,
                    "backend": entry.get("backend"),
                    "size": entry.get("size"),
                    "in_host": name in _host_resident,
                }
            )
        for name, entry in on_disk.items():
            if name in seen:
                continue
            models.append(
                {
                    "name": name,
                    "deployed": False,
                    "on_disk": True,
                    "backend": entry.get("backend"),
                    "size": entry.get("size"),
                    "in_host": name in _host_resident,
                }
            )

        pool_size = (store_config or {}).get("mem_pool_size")
        host_bytes = sum(
            (on_disk.get(name, {}).get("size") or 0) for name in _host_resident
        )
        return {
            "models": models,
            "mem_pool_size": pool_size,
            "chunk_size": (store_config or {}).get("chunk_size"),
            "host_bytes": host_bytes,
            "last_metrics": _last_metrics,
        }

    @app.post("/ui/deploy")
    async def ui_deploy(request: Request):
        """Register a model with the default single-node config."""
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        config = _default_model_config(
            model_name, body.get("backend") or "transformers"
        )
        if body.get("num_gpus") is not None:
            config["num_gpus"] = int(body["num_gpus"])

        logger.info(f"Received request to deploy model {model_name}")
        try:
            await controller.register.remote(config)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        _host_resident.pop(model_name, None)
        return {"status": f"deployed model {model_name}"}

    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    async def ui_index():
        if not UI_INDEX.exists():
            raise HTTPException(status_code=404, detail="Console not found")
        return FileResponse(UI_INDEX)

    async def inference_handler(request: Request, action: str):
        body = await request.json()
        # t0 for TTFT, so the backend can measure "request received ->
        # first generated token" including any cold start.
        body["sllm_t0"] = time.time()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.inference.remote(body, action)
        result = await result

        # A finished request means the weights were in host memory (either
        # just read back from disk or served from the pinned pool).
        if isinstance(result, dict) and result.get("sllm_metrics"):
            _last_metrics[model_name] = result["sllm_metrics"]
            _host_resident[model_name] = time.time()

        return result

    async def fine_tuning_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received fine tuning request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )
        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        job_id = await controller.register_ft_job.remote(body)
        return {"job_id": job_id}

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/v1/fine-tuning/jobs")
    async def fine_tuning(request: Request):
        return await fine_tuning_handler(request)

    @app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}")
    async def get_job_status(fine_tuning_job_id: str):
        if not fine_tuning_job_id:
            raise HTTPException(
                status_code=400, detail="Missing fine_tuning_job_id parameter"
            )
        controller = ray.get_actor("controller")
        status = await controller.get_ft_job_status.remote(fine_tuning_job_id)
        return {
            "id": fine_tuning_job_id,
            "object": "fine_tuning.job",
            "status": status,
        }

    @app.post("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel")
    async def cancel_job(fine_tuning_job_id: str):
        if not fine_tuning_job_id:
            raise HTTPException(
                status_code=400, detail="Missing fine_tuning_job_id parameter"
            )
        controller = ray.get_actor("controller")
        await controller.cancel_ft_job.remote(fine_tuning_job_id)
        return {
            "id": fine_tuning_job_id,
            "object": "fine_tuning.job",
            "status": "cancelled",
        }

    @app.get("/v1/models")
    async def get_models():
        logger.info("Attempting to retrieve the controller actor")
        try:
            controller = ray.get_actor("controller")
            if not controller:
                logger.error("Controller not initialized")
                raise HTTPException(
                    status_code=500, detail="Controller not initialized"
                )
            logger.info("Controller actor found")
            result = await controller.status.remote()
            logger.info("Controller status retrieved successfully")
            return result
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve models"
            )

    return app
