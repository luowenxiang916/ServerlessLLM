# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import json
import os
import re
import tempfile
import time
from functools import reduce
from contextlib import suppress
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import BitsAndBytesConfig
from accelerate.utils import find_tied_parameters, set_module_tensor_to_device
from transformers.quantizers.auto import AutoHfQuantizer
from transformers.utils.quantization_config import (
    QuantizationConfigMixin,
)

from sllm_store.client import SllmStoreClient
from sllm_store.logger import init_logger

logger = init_logger(__name__)


def set_module_buffer_to_device(
    module: nn.Module,
    target: str,
    device: torch.device,
):
    module_path, _, buffer_name = target.rpartition(".")

    mod: torch.nn.Module = module.get_submodule(module_path)

    if not hasattr(mod, buffer_name):
        raise AttributeError(
            mod._get_name() + " has no attribute `" + buffer_name + "`"
        )

    buffer = mod._buffers[buffer_name]
    mod._buffers[buffer_name] = buffer.to(device)


def send_module_buffers_to_device(
    module: nn.Module,
    device_map: dict,
):
    if "" in device_map and len(device_map) != 1:
        raise RuntimeError(
            f"Device map {device_map} is invalid. If you want to specify the default device, use key ''."  # noqa: E501
        )

    buffer_names = [name for name, _ in module.named_buffers()]
    for tensor_or_module, device_id in device_map.items():
        if tensor_or_module == "":
            for buffer_name in buffer_names:
                set_module_buffer_to_device(module, buffer_name, device_id)
        else:
            for buffer_name in buffer_names:
                if buffer_name.startswith(tensor_or_module):
                    set_module_buffer_to_device(module, buffer_name, device_id)


# CUDA kernels (CUTLASS/FlashInfer, inductor-generated Triton) issue vectorized
# loads that require 16/128/256-byte aligned base pointers. Upstream packs the
# per-rank tensors back to back in one big device buffer, which leaves almost
# every tensor after the first at a 2-byte-aligned offset and makes such kernels
# fail with `CUDA error: misaligned address`. Pad every slot to this boundary.
DEVICE_TENSOR_ALIGNMENT = 256


def _align_up(value, alignment=DEVICE_TENSOR_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def calculate_device_memory(device_map, tensor_index):
    device_memory = {}
    tensor_record = {}
    for tensor_name, device in device_map.items():
        if tensor_name in tensor_index:
            if device not in device_memory:
                device_memory[device] = 0
            offset, size = tensor_index[tensor_name]
            if (offset, size) in tensor_record:
                continue  # Skip duplicate tensors
            tensor_record[(offset, size)] = True
            # Keep the running total in sync with
            # `calculate_tensor_device_offsets`, which also aligns each slot.
            device_memory[device] = _align_up(device_memory[device]) + size
        else:
            raise ValueError(f"Tensor {tensor_name} not found in tensor_index.")

    return device_memory


def calculate_tensor_device_offsets(device_map, tensor_index):
    tensor_device_offsets = {}
    tensor_copy_chunks = {}
    device_offset = {}
    tensor_record = {}
    for tensor_name, device in device_map.items():
        if device not in tensor_device_offsets:
            tensor_device_offsets[device] = {}
            tensor_copy_chunks[device] = []
            device_offset[device] = 0
        if tensor_name in tensor_index:
            offset, size = tensor_index[tensor_name]
            if (offset, size) in tensor_record:
                tensor_device_offsets[device][tensor_name] = tensor_record[
                    (offset, size)
                ]
            else:
                device_offset[device] = _align_up(device_offset[device])
                tensor_record[(offset, size)] = device_offset[device]
                tensor_device_offsets[device][tensor_name] = device_offset[
                    device
                ]
                tensor_copy_chunks[device].append(
                    (offset, size, device_offset[device], 0)
                )
                device_offset[device] += size
        else:
            raise ValueError(f"Tensor {tensor_name} not found in tensor_index.")

    return tensor_device_offsets, tensor_copy_chunks


def get_total_parameter_size(module):
    total_param_size = 0
    for param in module.parameters():
        total_param_size += param.numel() * dtype_byte_size(param.dtype)
    return total_param_size


def get_parameter_size(model: nn.Module, param_path: str):
    # Split the parameter path by dots
    attributes = param_path.split(".")

    # Use reduce to traverse the model's attributes
    param = reduce(getattr, attributes, model)

    # Return the size of the parameter
    return param.numel() * dtype_byte_size(param.dtype)


def get_no_split_modules(model, no_split_modules_list, parent_name=""):
    no_split_modules = {}
    for name, submodule in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        module_class_name = submodule.__class__.__name__
        # If the module is a leaf module or in the no_split_modules_list, we don't split it # noqa: E501
        if (
            not list(submodule.children())
            or module_class_name in no_split_modules_list
        ):
            no_split_modules[full_name] = get_total_parameter_size(submodule)
            continue
        no_split_modules.update(
            get_no_split_modules(submodule, no_split_modules_list, full_name)
        )

    return no_split_modules


def get_tied_no_split_modules(model, no_split_modules):
    tied_parameters = find_tied_parameters(model)
    tied_modules = []
    for tied_param_group in tied_parameters:
        tied_module_group = []
        shared_size = None
        for tied_param in tied_param_group:
            param_size = get_parameter_size(model, tied_param)
            if shared_size is None:
                shared_size = param_size
            else:
                assert (
                    shared_size == param_size
                ), f"Parameter {tied_param} does not have the same size as the other parameters in the group"  # noqa: E501
            tied_module = None
            while "." in tied_param:
                tied_param = tied_param.rsplit(".", 1)[0]
                if tied_param in no_split_modules:
                    tied_module = tied_param
                    break
            if tied_module is None:
                raise ValueError(
                    f"Parameter {tied_param} is not in the no_split_modules list"  # noqa: E501
                )
            tied_module_group.append(tied_module)
        tied_modules.append((tied_module_group, shared_size))

    return tied_modules


def dtype_byte_size(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8


def to_num_bytes(value: str) -> int:
    """
    Convert a string representing a data size to its equivalent number of bytes.

    The input must strictly follow the format:
        <number><unit>

    - <number>: A positive integer.
    - <unit>: One of the following case-sensitive units:
        B, KB, MB, GB, TB, PB, EB, ZB, YB

    No leading, trailing, or middle spaces or other characters are allowed.

    Examples:
        "1GB"  -> 1073741824
        "512MB" -> 536870912

    Args:
        value (str): The string to convert.

    Returns:
        int: The equivalent number of bytes.

    Raises:
        ValueError: If the input format is incorrect.
    """
    # Define the regular expression pattern for validation
    pattern = r"^(\d+)(B|KB|MB|GB|TB|PB|EB|ZB|YB)$"
    match = re.fullmatch(pattern, value)

    if not match:
        error_message = (
            "Invalid format. The input must be a positive integer "
            "followed immediately by a unit "
            "(B, KB, MB, GB, TB, PB, EB, ZB, YB), case sensitive, "
            "with no spaces or other characters."
        )
        raise ValueError(error_message)

    number_str, unit = match.groups()
    number = int(number_str)

    # Define the multiplier for each unit
    unit_multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "EB": 1024**6,
        "ZB": 1024**7,
        "YB": 1024**8,
    }

    bytes_value = number * unit_multipliers[unit]
    return bytes_value


def quantize(
    model,
    state_dict,
    quantization_config,
    torch_dtype,
    device_map,
    model_path,
    replica_uuid,
    logger,
):
    if isinstance(quantization_config, dict):
        try:
            quantization_config = BitsAndBytesConfig.from_dict(
                quantization_config
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid quantization_config dictionary: {e}")
            raise ValueError(
                f"Invalid quantization_config dictionary: {e}"
            ) from e

    if not isinstance(quantization_config, QuantizationConfigMixin):
        raise ValueError(f"Invalid config type: {type(quantization_config)}")

    quant_method = quantization_config.quant_method
    if quant_method in [
        "aqlm",
        "awq",
        "bitnet",
        "quanto",
        "vptq",
        "quark",
        "higgs",
        "hqq",
        "eetq",
        "torchao",
        "spqr",
        "fp8",
        "auto-round",
        "fbgemm_fp8",
        "compressed_tensors",
    ]:
        raise ValueError(
            f"{quant_method} (requires pre-quantization) " "is not supported."
        )

    logger.info(f"Using quantization method: {quant_method}")
    if quantization_config.quant_method == "bitsandbytes":
        precision = quantization_config.quantization_method()
        logger.info(f"Using precision: {precision}")

        if quantization_config.llm_int8_enable_fp32_cpu_offload:
            logger.debug("Offloading is not supported yet")
            quantization_config.llm_int8_enable_fp32_cpu_offload = False
    else:
        with suppress(Exception):
            logger.info(f"Using precision: {quantization_config.bits}")

    torch_dtype = torch_dtype or torch.float16
    hf_quantizer = AutoHfQuantizer.from_config(
        quantization_config, pre_quantized=False
    )
    model.hf_device_map = device_map
    model.hf_quantizer = hf_quantizer
    hf_quantizer.validate_environment(device_map=device_map)
    hf_quantizer.preprocess_model(model, device_map=device_map)

    # synchronize
    client = SllmStoreClient("127.0.0.1:8073")
    client.confirm_model_loaded(model_path, replica_uuid)

    for name, param in state_dict.items():
        if param.is_floating_point():
            param = param.to(torch_dtype)
        # Use the new param_needs_quantization API (transformers >= 4.52)
        if hf_quantizer.param_needs_quantization(model, name):
            final_device = param.device
            hf_quantizer.create_quantized_param(
                model,
                param,
                name,
                final_device,
            )

        else:
            set_module_tensor_to_device(model, name, param.device, param)

    # converting new biases
    for module in model.modules():
        b = getattr(module, "bias", None)
        if b is not None and b.dtype != torch_dtype:
            b.data = b.data.to(torch_dtype)

    if quant_method == "gptq":
        model = model.to("cuda")

    hf_quantizer.postprocess_model(model)
    return model


# --------------------------------------------------------------------------- #
# Load stage timings
#
# The vLLM backend loads the weights inside vLLM's EngineCore process, while
# the metrics are reported by the backend actor, which lives in another
# process. The EngineCore side therefore records the timings of the load it
# just ran as a small JSON file and the actor picks them up once its engine is
# up (see `sllm.backends.vllm_backend`).
#
# One file is written per shard (`<model>/rank_0` -> `<key>.json`); the reader
# merges all shards of a model and keeps the slowest one, because the model is
# only usable once every shard of the tensor-parallel group is resident.
# A missing or unreadable file is not an error: the timings are then simply
# reported as unknown.
# --------------------------------------------------------------------------- #
_LOAD_TIMINGS_DIR = os.path.join(tempfile.gettempdir(), "sllm_load_timings")


def _load_timings_key(model_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(model_path).strip("/"))


def record_load_timings(model_path: str, timings: Dict[str, Any]) -> None:
    """Persist the stage timings of one shard load (caller: EngineCore)."""
    try:
        os.makedirs(_LOAD_TIMINGS_DIR, exist_ok=True)
        path = os.path.join(
            _LOAD_TIMINGS_DIR, _load_timings_key(model_path) + ".json"
        )
        with open(path, "w") as f:
            json.dump({**timings, "written_at": time.time()}, f)
    except OSError as e:
        logger.warning(f"Failed to record load timings for {model_path}: {e}")


def read_load_timings(
    model_path: str, newer_than: Optional[float] = None
) -> Dict[str, Any]:
    """Read back the timings of the shards that make up *model_path*.

    ``model_path`` is the model as the store knows it, i.e. without the local
    storage prefix and without the ``rank_<n>`` suffix. Entries written before
    ``newer_than`` (epoch seconds) are ignored, so a stale file left behind by
    an earlier instance is never reported. Returns the slowest shard's
    timings, or an empty dict when nothing fresh is available.
    """
    key = _load_timings_key(model_path)
    try:
        names = os.listdir(_LOAD_TIMINGS_DIR)
    except OSError:
        return {}

    best: Dict[str, Any] = {}
    for name in names:
        if not name.endswith(".json"):
            continue
        shard_key = name[: -len(".json")]
        if shard_key != key and not shard_key.startswith(key + "__"):
            continue
        try:
            with open(os.path.join(_LOAD_TIMINGS_DIR, name)) as f:
                entry = json.load(f)
        except (OSError, ValueError):
            continue
        if newer_than is not None and entry.get("written_at", 0) < newer_than:
            continue
        if entry.get("disk_to_vram_s") is None:
            continue
        if (
            best.get("disk_to_vram_s") is None
            or entry["disk_to_vram_s"] > best["disk_to_vram_s"]
        ):
            best = entry
    return best
