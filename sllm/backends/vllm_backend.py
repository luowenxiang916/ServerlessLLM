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
import asyncio
import gc
import inspect
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt
from vllm.utils.counter import Counter

from sllm.backends.backend_utils import (
    DISK_TO_HOST_S_THRESHOLD,
    BackendStatus,
    SllmBackend,
)

try:  # `sllm_store` is only importable on a node that ships the store
    from sllm_store.utils import read_load_timings
except ImportError:  # pragma: no cover - keeps the module importable alone
    read_load_timings = None  # type: ignore[assignment]

logger = logging.getLogger("ray")


def _model_path_in_store(model_path: str) -> Optional[str]:
    """Return *model_path* the way the store names it.

    ``ServerlessLLMLoader`` strips the local storage prefix before asking the
    store for the weights, so this is the key the load timings are filed under
    (see ``sllm_store.utils.record_load_timings``).
    """
    storage_path = os.path.normpath(
        os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    )
    normalized = os.path.normpath(model_path)
    if not normalized.startswith(storage_path + os.sep):
        return None
    return normalized[len(storage_path) + 1 :]


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# Note the GPU resource will be decided when the backend is created
class VllmBackend(SllmBackend):
    # This class implements every method in vllm.entrypoints.openai.api_server
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
    # except that we use ray.remote instead of @app and we also add a few new methods:
    # - stop: stops every ongoing request and then stops the backend
    # - get_current_tokens: returns a list of all ongoing request tokens
    # - resume_kv_cache: resumes the key-value cache for the given requests
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)
        self.enforce_eager = backend_config.get("enforce_eager", False)
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", True
        )
        # NOTE: vLLM renamed the engine-level `task` arg to `runner`. Accept
        # either spelling so this works across vLLM versions.
        self.task = backend_config.get(
            "task", backend_config.get("runner", "auto")
        )

        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            model_path = backend_config.get("pretrained_model_name_or_path")
        else:
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["load_format"] = "serverless_llm"
        filtered_engine_config["model"] = model_path
        # Name the store uses for these weights; `None` when the path is not
        # below the storage root, in which case no timings can be reported.
        self.store_model_path = _model_path_in_store(model_path)
        # Cold-start timings, filled in by `init_backend`, in seconds.
        self.load_timings: Dict[str, Any] = {}

        # NOTE: Automatic enable prefix cachinging
        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        if "task" in async_engine_fields:
            filtered_engine_config["task"] = self.task
        elif "runner" in async_engine_fields:
            filtered_engine_config["runner"] = self.task

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            _engine_t0 = time.time()
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            # vLLM loads the weights inside its own EngineCore process and
            # `from_engine_args` only returns once that process is ready, so
            # the stage timings it recorded are complete by now.
            self._read_load_timings(newer_than=_engine_t0)
            self.status = BackendStatus.RUNNING

    def _read_load_timings(self, newer_than: float) -> None:
        if read_load_timings is None or not self.store_model_path:
            return
        self.load_timings = read_load_timings(
            self.store_model_path, newer_than=newer_than
        )
        logger.info(
            f"Load timings for {self.store_model_path}: {self.load_timings}"
        )

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        # t0 for TTFT, stamped by the API layer when the request arrived. It is
        # not a SamplingParams field, so drop it before building them.
        request_start = request_data.pop("sllm_t0", None)

        model_name: str = request_data.pop("model", "vllm-model")
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # If prompt is not provided, construct it from messages
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # TODO stream results

        # Non-stream case
        final_output = None
        first_token_time = None
        async for response_output in results_generator:
            if first_token_time is None:
                first_token_time = time.time()
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)

        assert final_output is not None

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)

        response = process_output(final_output, model_name)
        # Same shape the transformers backend reports, so the console can treat
        # both backends alike.
        response["sllm_metrics"] = self._collect_metrics(
            request_start, first_token_time
        )
        return response

    def _collect_metrics(
        self,
        request_start: Optional[float],
        first_token_time: Optional[float],
    ) -> Dict[str, Any]:
        """Cold-start / latency metrics for the frontend (seconds).

        `disk_to_host_s` / `mem_to_vram_s` / `disk_to_vram_s` are measured by
        the store while the weights are paged in, i.e. inside vLLM's
        EngineCore process, and handed over to this actor by `init_backend`.
        They are therefore constant for the lifetime of the instance, which is
        exactly what the console wants: it shows how long the model that is
        currently serving took to arrive.
        """
        disk_to_host_s = self.load_timings.get("disk_to_host_s")
        return {
            "disk_to_host_s": disk_to_host_s,
            "disk_to_vram_s": self.load_timings.get("disk_to_vram_s"),
            "mem_to_vram_s": self.load_timings.get("mem_to_vram_s"),
            "ttft_s": (
                None
                if (first_token_time is None or request_start is None)
                else first_token_time - request_start
            ),
            "from_disk": bool(
                disk_to_host_s is not None
                and disk_to_host_s > DISK_TO_HOST_S_THRESHOLD
            ),
        }

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Abort all requests
        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)
