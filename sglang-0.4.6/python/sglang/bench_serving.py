# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

"""
Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
"""

import argparse
import asyncio
import json
import os
import pickle
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
ASSISTANT_SUFFIX = "Assistant:"

global args


CTX_LIMIT = 32768
CTX_SAFETY = 64

def _truncate_prompt_to_limit(tokenizer, prompt: str, out_len: int) -> tuple[str, int]:
    budget = CTX_LIMIT - CTX_SAFETY - max(int(out_len), 0)
    if budget <= 0:
        return "", 0
    ids = tokenizer.encode(prompt)  
    if len(ids) <= budget:
        return prompt, len(ids)
    new_ids = ids[:budget]
    return tokenizer.decode(new_ids), len(new_ids)


# don't want to import sglang package here
def _get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("true", "1")


def _parse_total_lens(args, num_groups: int) -> List[int]:
    if args.gsp_total_lens:
        arr = [int(x.strip()) for x in args.gsp_total_lens.split(",") if x.strip()]
        if not arr:
            raise ValueError("--gsp-total-lens is empty")
  
        return [arr[i % len(arr)] for i in range(num_groups)]

    if args.gsp_total_len_min is None or args.gsp_total_len_max is None:
        raise ValueError("Provide --gsp-total-lens or both --gsp-total-len-min and --gsp-total-len-max")
    lo, hi = args.gsp_total_len_min, args.gsp_total_len_max
    if lo <= 0 or hi < lo:
        raise ValueError("Invalid total len range")
    if args.gsp_total_len_sampler == "uniform":
        return list(np.random.randint(lo, hi + 1, size=num_groups))
    else:  
        a, b = np.log(max(lo,1)), np.log(hi)
        return [int(np.exp(np.random.uniform(a, b))) for _ in range(num_groups)]


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    # new 
    queue_ms: float = 0.0     
    service_ms: float = 0.0   
    # predict
    est_wait_ms: float = 0.0
    est_service_ms: float = 0.0
    est_total_ms: float = 0.0


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def get_auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        return {}


# trt llm does not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.000001,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "min_length": request_func_input.output_len,
            "end_id": 1048576,
            **request_func_input.extra_request_body,
        }
        if args.disable_ignore_eos:
            del payload["min_length"]
            del payload["end_id"]
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                                output_len = (data.get("usage") or {}).get(
                                    "completion_tokens", output_len
                                )

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_truss(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_sglang_generate(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": request_func_input.output_len,
                "ignore_eos": not args.disable_ignore_eos,
            },
            "stream": not args.disable_stream,
            "lora_path": request_func_input.lora_name,
            "return_logprob": args.return_logprob,
            "logprob_start_len": -1,
            **request_func_input.extra_request_body,
        }
        headers = get_auth_headers()
        if not args.disable_stream:
            headers["Accept"] = "text/event-stream"

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        last_output_len = 0
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                def _fh(name: str) -> float:
                    v = response.headers.get(name)
                    try:
                        return float(v) if v not in (None, "") else 0.0
                    except Exception:
                        return 0.0
                try:
                    output.est_wait_ms    = _fh("X-SGLang-Predicted-Wait-Ms")
                    output.est_service_ms = _fh("X-SGLang-Predicted-Service-Ms")
                    output.est_total_ms   = _fh("X-SGLang-Predicted-Total-Ms") or _fh("X-SGLang-Predicted-E2E-Ms")
                    output.queue_ms       = _fh("X-SGLang-Queue-Time-Ms")
                    output.service_ms     = _fh("X-SGLang-Service-Time-Ms")
                except Exception:
                    pass

                if response.status == 200:
                    if args.disable_stream:
                        try:
                            data = await response.json(content_type=None)
                        except Exception:
                            text = await response.text()
                            data = {}
                            try:
                                data = json.loads(text)
                            except Exception:
                                pass

                        generated_text = data.get("text") or ""
                        meta = data.get("meta_info") or {}
                        output_len = int(meta.get("completion_tokens") or 0)
                        latency = time.perf_counter() - st

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len


                        output.queue_ms   = float(meta.get("queue_time_ms")    or output.queue_ms or 0.0)
                        output.service_ms = float(meta.get("service_time_ms")  or output.service_ms or 0.0)

                    else:
                        buf = ""
                        done = False
                        latency = 0.0
                        output_len = 0

                        q_ms_seen: Optional[float] = None
                        svc_ms_seen: Optional[float] = None

                        async for raw in response.content.iter_any():
                            if not raw:
                                continue

                            try:
                                buf += raw.decode("utf-8", errors="ignore")
                            except Exception:
                                continue
                            if "\r\n" in buf:
                                buf = buf.replace("\r\n", "\n")

                            while True:
                                sep = buf.find("\n\n")
                                if sep == -1:
                                    break
                                event = buf[:sep]
                                buf = buf[sep + 2 :]

                                ev_type = "message"
                                data_lines = []
                                for line in event.split("\n"):
                                    if not line:
                                        continue
                                    if line.startswith(":"):  
                                        continue
                                    if line.startswith("event:"):
                                        ev_type = line[len("event:") :].strip()
                                        continue
                                    if line.startswith("data:"):
                                        data_lines.append(line[len("data:") :].lstrip())

                                if not data_lines:
                                    continue

                                data_str = "\n".join(data_lines).strip()

                                if data_str == "[DONE]":
                                    done = True
                                    latency = time.perf_counter() - st
                                    break

                                if ev_type == "metrics":
                                    try:
                                        m = json.loads(data_str)
                                        if "queue_time_ms" in m:
                                            q_ms_seen = float(m["queue_time_ms"])
                                        if "service_time_ms" in m:
                                            svc_ms_seen = float(m["service_time_ms"])
                                    except Exception:
                                        pass

                                    continue

                                try:
                                    data = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                metrics_obj = (data.get("metrics")
                                            or (data.get("x_sglang") or {}).get("metrics")
                                            or {})
                                try:
                                    if "queue_time_ms" in metrics_obj:
                                        q_ms_seen = float(metrics_obj["queue_time_ms"])
                                    if "service_time_ms" in metrics_obj:
                                        svc_ms_seen = float(metrics_obj["service_time_ms"])
                                except Exception:
                                    pass

                                text_piece = data.get("text")
                                if text_piece is None:
                                    try:
                                        ch = data.get("choices") or []
                                        if ch:
                                            delta = ch[0].get("delta") or {}
                                            text_piece = delta.get("content")
                                    except Exception:
                                        text_piece = None

                                if text_piece:
                                    timestamp = time.perf_counter()
                                    generated_text = text_piece

                                    meta = data.get("meta_info") or {}
                                    ct = meta.get("completion_tokens")
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft
                                    else:
                                        if isinstance(ct, int):
                                            num_new_tokens = ct - last_output_len
                                            output_len = ct
                                        else:
                                            num_new_tokens = 1
                                            output_len += 1

                                        if num_new_tokens <= 0:
                                            continue

                                        adjust_itl = (timestamp - most_recent_timestamp) / num_new_tokens
                                        output.itl.extend([adjust_itl] * num_new_tokens)

                                    most_recent_timestamp = timestamp
                                    last_output_len = output_len

                 
                        if latency == 0.0:
                            latency = time.perf_counter() - st

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len

                        if q_ms_seen is not None:
                            output.queue_ms = q_ms_seen
                        if svc_ms_seen is not None:
                            output.service_ms = svc_ms_seen
                        if not output.service_ms and output.latency:
                            output.service_ms = max(output.latency * 1000.0 - output.queue_ms, 0.0)
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(f"{output.error=}")

    if pbar:
        pbar.update(1)
    return output


async def async_request_gserver(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    raise NotImplementedError()


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() == "true":
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def get_dataset(args, tokenizer):
    if args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )
    elif args.dataset_name.startswith("random"):
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            random_sample=args.dataset_name == "random",
        )
    elif args.dataset_name == "generated-shared-prefix":
        input_requests = sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            tokenizer=tokenizer,
            args=args,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return input_requests


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_sglang_generate,
    "sglang-native": async_request_sglang_generate,
    "sglang-oai": async_request_openai_completions,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "trt": async_request_trt_llm,
    "gserver": async_request_gserver,
    "truss": async_request_truss,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p95_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append((prompt, prompt_len, output_len))

    print(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    return filtered_dataset


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
) -> List[Tuple[str, int, int]]:
    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    if random_sample:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path):
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[Tuple[str, int, int]] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            prompt = tokenizer.decode(input_ids)
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % tokenizer.vocab_size
                    for j in range(input_lens[i])
                ]
            )
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def get_gen_prefix_cache_path(args, tokenizer):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / "benchmark"

    # Create a unique cache filename based on the generation parameters
    cache_key = (
        f"gen_shared_prefix_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
        f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_{args.gsp_output_len}_"
        f"{tokenizer.__class__.__name__}_{args.gsp_prefix_ratio}_{args.gsp_order}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> List[Tuple[str, int, int]]:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(args, tokenizer)

    # Try to load from cache first
    if cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    total_lens = _parse_total_lens(args, num_groups)
    ratio = float(args.gsp_prefix_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError("--gsp-prefix-ratio must be in (0,1)")

    print("\nGenerating new input data...")

    # Generate system prompts for each group
    system_prompts: List[str] = []
    sys_lens: List[int] = []
    for g in range(num_groups):
        Lg = int(total_lens[g])
        Sg = max(1, int(round(ratio * Lg)))
        Sg = min(Sg, max(Lg - 1, 1))  
        sp = gen_prompt(tokenizer, Sg)
        real_Sg = len(tokenizer.encode(sp))
        if real_Sg != Sg:
            if real_Sg > Sg:
                ids = tokenizer.encode(sp)[:Sg]
            else:
                base = tokenizer.encode(sp)
                need = Sg - real_Sg
                ids = base + base[:need]
            sp = tokenizer.decode(ids)
        system_prompts.append(sp)
        sys_lens.append(len(tokenizer.encode(sp)))

    group_reqs: List[List[Tuple[str, int, int]]] = [[] for _ in range(num_groups)]
    total_input_tokens = 0
    total_output_tokens = 0

    for g in range(num_groups):
        Lg = int(total_lens[g])
        Sg = sys_lens[g]
        Qg = max(1, Lg - Sg)
        for _ in range(prompts_per_group):
            q = gen_prompt(tokenizer, Qg)
            real_Qg = len(tokenizer.encode(q))
            if real_Qg != Qg:
                if real_Qg > Qg:
                    ids = tokenizer.encode(q)[:Qg]
                else:
                    base = tokenizer.encode(q)
                    need = Qg - real_Qg
                    ids = base + base[:need]
                q = tokenizer.decode(ids)
            full_prompt = f"{system_prompts[g]}\n\n{q}"
            prompt_len = len(tokenizer.encode(full_prompt))
            group_reqs[g].append((full_prompt, prompt_len, output_len))
            total_input_tokens += prompt_len
            total_output_tokens += output_len

    order = (args.gsp_order or "random").lower()
    if order in ("grouped", "best"):
        input_requests = [item for g in range(num_groups) for item in group_reqs[g]]
    elif order in ("round_robin", "rr", "worst"):
        input_requests = []
        for i in range(prompts_per_group):
            for g in range(num_groups):
                input_requests.append(group_reqs[g][i])
    else:
        input_requests = [item for g in range(num_groups) for item in group_reqs[g]]
        random.shuffle(input_requests)

    print(f"\nGenerated v2 stats:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Total prompts: {len(input_requests)}")
    print(f"Per-group total lens (first 10): {total_lens[:10]}")
    print(f"Prefix ratio: {ratio}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}\n")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching generated input data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(input_requests, f)

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p95_ttft_ms=np.percentile(ttfts or 0, 95) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p90_e2e_latency_ms=np.percentile(e2e_latencies, 90) * 1000,
        p95_e2e_latency_ms=np.percentile(e2e_latencies, 95) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    lora_names: List[str],
    extra_request_body: Dict[str, Any],
    profile: bool,
    pd_seperated: bool = False,
    flush_cache: bool = False,
    warmup_requests: int = 1,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    # Warmup
    print(f"Starting warmup with {warmup_requests} sequences...")

    # Use the first request for all warmup iterations
    test_prompt, test_prompt_len, test_output_len = input_requests[0]
    if lora_names is not None and len(lora_names) != 0:
        lora_name = lora_names[0]
    else:
        lora_name = None

    # Create the test input once
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=min(test_output_len, 32),
        lora_name=lora_name,
        extra_request_body=extra_request_body,
    )

    # Run warmup requests
    warmup_tasks = []
    for _ in range(warmup_requests):
        warmup_tasks.append(
            asyncio.create_task(request_func(request_func_input=test_input))
        )

    warmup_outputs = await asyncio.gather(*warmup_tasks)

    # Check if at least one warmup request succeeded
    if warmup_requests > 0 and not any(output.success for output in warmup_outputs):
        raise ValueError(
            "Warmup failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {warmup_outputs[0].error}"
        )
    else:
        print(
            f"Warmup completed with {args.warmup_requests} sequences. Starting main benchmark run..."
        )

    # Flush cache
    if ("sglang" in backend and _get_bool_env_var("SGLANG_IS_IN_CI")) or flush_cache:
        requests.post(base_url + "/flush_cache", headers=get_auth_headers())

    time.sleep(1.0)

    # Start profiler
    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        if lora_names is not None and len(lora_names) != 0:
            idx = random.randint(0, len(lora_names) - 1)
            lora_name = lora_names[idx]
        else:
            lora_name = None

        # for mixtral
        # prompt, prompt_len = _truncate_prompt_to_limit(tokenizer, prompt, output_len)

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name=lora_name,
            extra_request_body=extra_request_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    # Stop profiler
    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    if "sglang" in backend:
        server_info = requests.get(base_url + "/get_server_info")
        if pd_seperated:
            accept_length = server_info.json()["decode"][0].get(
                "avg_spec_accept_length", None
            )
        else:
            accept_length = server_info.json().get("avg_spec_accept_length", None)
    else:
        accept_length = None

    # Compute metrics and print results
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max reqeuest concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    if accept_length:
        print("{:<40} {:<10.2f}".format("Accept length:", accept_length))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{:<40} {:<10.2f}".format("P90 E2E Latency (ms):", metrics.p90_e2e_latency_ms)) 
    print("{:<40} {:<10.2f}".format("P95 E2E Latency (ms):", metrics.p95_e2e_latency_ms))   
    print("{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms))
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))
    print("{:<40} {:<10.2f}".format("P95 TTFT (ms):", metrics.p95_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("Max ITL (ms):", metrics.max_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # Arguments
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            # Results
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
            "p90_e2e_latency_ms": metrics.p90_e2e_latency_ms,
            "p95_e2e_latency_ms": metrics.p95_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p90_ttft_ms": metrics.p90_ttft_ms,
            "p95_ttft_ms": metrics.p95_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p95_itl_ms": metrics.p95_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "concurrency": metrics.concurrency,
            "accept_length": accept_length,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name.startswith("random"):
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")


    per_req_file = (args.output_file or output_file_name).replace(".jsonl", "_per_request.jsonl")
    with open(per_req_file, "a") as f:
        for i, o in enumerate(outputs):
            f.write(json.dumps({
                "idx": i,
                "success": o.success,
                "prompt_len": o.prompt_len,
                "output_len": o.output_len,
                "queue_time_ms": float(getattr(o, "queue_ms", 0.0) or 0.0),
                "service_time_ms": float(getattr(o, "service_ms", 0.0) or 0.0),
                "e2e_latency_ms": float(o.latency * 1000.0),
   
                "est_wait_ms":     float(getattr(o, "est_wait_ms", 0.0) or 0.0),
                "est_service_ms":  float(getattr(o, "est_service_ms", 0.0) or 0.0),
                "est_total_ms":    float(getattr(o, "est_total_ms", 0.0) or 0.0),
                "error": o.error,
            }) + "\n")

    result.update(
        {
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
            "queue_times_ms":   [getattr(o, "queue_ms", 0.0)   for o in outputs],
            "service_times_ms": [getattr(o, "service_ms", 0.0) for o in outputs],
            "e2e_latencies_ms": [o.latency * 1000.0 for o in outputs],
            "est_wait_times_ms":    [getattr(o, "est_wait_ms", 0.0)    for o in outputs],
            "est_service_times_ms": [getattr(o, "est_service_ms", 0.0) for o in outputs],
            "est_total_times_ms":   [getattr(o, "est_total_ms", 0.0)   for o in outputs],
        }
    )

    return result


def check_chat_template(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False


def set_global_args(args_: argparse.Namespace):
    """Set the global args."""
    global args
    args = args_


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set default value for max_concurrency if not present
    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    # Set default value for warmup_requests if not present
    if not hasattr(args, "warmup_requests"):
        args.warmup_requests = 1

    print(f"benchmark_args={args}")

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Set url
    if args.port is None:
        args.port = {
            "sglang": 30000,
            "sglang-native": 30000,
            "sglang-oai": 30000,
            "lmdeploy": 23333,
            "vllm": 8000,
            "trt": 8000,
            "gserver": 9988,
            "truss": 8080,
        }.get(args.backend, 30000)

    model_url = (
        f"{args.base_url}/v1/models"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/models"
    )

    if args.backend in ["sglang", "sglang-native"]:
        api_url = (
            f"{args.base_url}/generate"
            if args.base_url
            else f"http://{args.host}:{args.port}/generate"
        )
    elif args.backend in ["sglang-oai", "vllm", "lmdeploy"]:
        api_url = (
            f"{args.base_url}/v1/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/completions"
        )
    elif args.backend == "trt":
        api_url = (
            f"{args.base_url}/v2/models/ensemble/generate_stream"
            if args.base_url
            else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
        )
        if args.model is None:
            print("Please provide a model using `--model` when using `trt` backend.")
            sys.exit(1)
    elif args.backend == "gserver":
        api_url = args.base_url if args.base_url else f"{args.host}:{args.port}"
        args.model = args.model or "default"
    elif args.backend == "truss":
        api_url = (
            f"{args.base_url}/v1/models/model:predict"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/models/model:predict"
        )
    base_url = (
        f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url
    )

    # Get model name
    if args.model is None:
        if args.backend == "truss":
            print(
                "Please provide a model with `--model` when using truss backend. e.g. --model meta-llama/Llama-3.1-8B-Instruct"
            )
            sys.exit(1)
        try:
            response = requests.get(model_url, headers=get_auth_headers())
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    if not check_chat_template(args.model):
        print(
            "\nWARNING It is recommended to use the `Chat` or `Instruct` model for benchmarking.\n"
            "Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly.\n"
        )

    print(f"{args}\n")

    # Read dataset
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id)
    input_requests = get_dataset(args, tokenizer)

    # compatible with SimpleNamespace
    if not hasattr(args, "flush_cache"):
        args.flush_cache = False

    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            lora_names=args.lora_name,
            extra_request_body=extra_request_body,
            profile=args.profile,
            pd_seperated=args.pd_seperated,
            flush_cache=args.flush_cache,
        )
    )


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


class LoRAPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [])
        for lora_name in values:
            getattr(namespace, self.dest).append(lora_name)


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "random", "random-ids", "generated-shared-prefix"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        nargs="*",
        default=None,
        action=LoRAPathAction,
        help="The names of LoRA adapters. You can provide a list of names in the format {name} {name} {name}...",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="",
        help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
    )
    parser.add_argument(
        "--pd-seperated",
        action="store_true",
        help="Benchmark PD disaggregation server",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Flush the cache before running the benchmark",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of warmup requests to run before the benchmark",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )

    parser.add_argument("--gsp-prefix-ratio", type=float, default=0.8,
                    help="Shared prefix ratio per group, 0~1.")
    parser.add_argument("--gsp-total-lens", type=str, default=None,
                        help="Comma list of total input lens per group, e.g. '1024,1536,2048'. "
                            "If provided, it overrides gsp-system-prompt-len / gsp-question-len.")
    parser.add_argument("--gsp-total-len-min", type=int, default=None)
    parser.add_argument("--gsp-total-len-max", type=int, default=None)
    parser.add_argument("--gsp-total-len-sampler", type=str, choices=["uniform","loguniform"],
                        default="uniform")

    parser.add_argument(
        "--gsp-order",
        type=str,
        choices=["random", "grouped", "round_robin", "rr", "best", "worst"],
        default="random",
        help="Order of sending GSP samples: "
            "'grouped'/'best' = send group by group; "
            "'round_robin'/'rr'/'worst' = interleave groups in round-robin; "
            "'random' = global shuffle."
    )

    args = parser.parse_args()
    run_benchmark(args)
