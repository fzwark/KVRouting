import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def set_ulimit():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 65535), hard), hard))
    except Exception:
        pass

def remove_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s

def get_tokenizer(identifier: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(identifier, use_fast=True, trust_remote_code=True)

def _find_single_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for s in [" a", " x", " 0", " z", "\n", ".", ",", " b"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode("x", add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    if tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    return tokenizer.encode(" ", add_special_tokens=False)[0]

def force_len_tokens(text: str, tokenizer: PreTrainedTokenizerBase,
                     target_len: int, max_trials: int = 3) -> str:
    """Force re-encoded length to exactly target_len by trunc/padding with a benign single-token."""
    pad_id = _find_single_token_id(tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= target_len:
        out = tokenizer.decode(ids[:target_len])
    else:
        out = tokenizer.decode(ids + [pad_id] * (target_len - len(ids)))
    for _ in range(max_trials):
        check = tokenizer.encode(out, add_special_tokens=False)
        if len(check) == target_len:
            return out
        if len(check) > target_len:
            out = tokenizer.decode(check[:target_len])
        else:
            out = tokenizer.decode(check + [pad_id] * (target_len - len(check)))
    return out


MsgContent = str
SampleOutput = List[List[Tuple[MsgContent, int, int]]] 

def _iter_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def _load_sharegpt(path: str) -> List[List[Dict[str, str]]]:
    convs = []
    if path.endswith(".jsonl"):
        for line in _iter_lines(path):
            obj = json.loads(line)
            raw = obj.get("conversations") or obj.get("messages") or []
            conv = []
            for m in raw:
                role = m.get("role")
                content = m.get("content")
                if role is None:
                    frm = m.get("from", "")
                    role = "user" if frm in ("human", "user") else ("assistant" if frm in ("gpt", "assistant") else frm)
                    content = m.get("value")
                if role and content is not None:
                    conv.append({"role": role, "content": content})
            if conv:
                convs.append(conv)
    else:
        data = json.load(open(path, "r", encoding="utf-8"))
        # if it's a dict with conversations
        if isinstance(data, dict) and "conversations" in data:
            data = [data]
        for obj in data:
            raw = obj.get("conversations") or obj.get("messages") or []
            conv = []
            for m in raw:
                role = m.get("role")
                content = m.get("content")
                if role is None:
                    frm = m.get("from", "")
                    role = "user" if frm in ("human", "user") else ("assistant" if frm in ("gpt", "assistant") else frm)
                    content = m.get("value")
                if role and content is not None:
                    conv.append({"role": role, "content": content})
            if conv:
                convs.append(conv)
    return convs

def _load_ultrachat(path: str) -> List[List[Dict[str, str]]]:
    convs = []
    if path.endswith(".jsonl"):
        for line in _iter_lines(path):
            obj = json.loads(line)
            raw = obj.get("conversations") or obj.get("messages") or []
            conv = []
            for m in raw:
                role = m.get("role")
                content = m.get("content")
                if role is None:
                    frm = m.get("from", "")
                    role = "user" if frm in ("human","user") else ("assistant" if frm in ("gpt","assistant") else frm)
                    content = m.get("value")
                if role and content is not None:
                    conv.append({"role": role, "content": content})
            if conv:
                convs.append(conv)
    else:
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict) and "conversations" in data:
            data = [data]
        for obj in data:
            raw = obj.get("conversations") or obj.get("messages") or []
            conv = []
            for m in raw:
                role = m.get("role")
                content = m.get("content")
                if role is None:
                    frm = m.get("from", "")
                    role = "user" if frm in ("human","user") else ("assistant" if frm in ("gpt","assistant") else frm)
                    content = m.get("value")
                if role and content is not None:
                    conv.append({"role": role, "content": content})
            if conv:
                convs.append(conv)
    return convs

def build_sample_output(
    convs: List[List[Dict[str, str]]],
    turns_per_client: int,
    per_turn_user_len: int,
    fixed_output_len: int,
    per_turn_user_len_list: Optional[List[int]] = None,
) -> SampleOutput:
    result: SampleOutput = []
    client_idx = 0
    for conv in convs:
        user_msgs = [m["content"] for m in conv if m.get("role") == "user"]
        if len(user_msgs) < turns_per_client:
            continue
        turns = user_msgs[:turns_per_client]
        if per_turn_user_len_list:
            length = per_turn_user_len_list[client_idx % len(per_turn_user_len_list)]
        else:
            length = per_turn_user_len
        conv_tuples = [(t, length, fixed_output_len) for t in turns]
        result.append(conv_tuples)
        client_idx += 1
    return result


def build_sample_output_stitch(
    convs: List[List[Dict[str, str]]],
    num_clients: int,
    turns_per_client: int,
    per_turn_user_len: int,
    fixed_output_len: int,
    per_turn_user_len_list: Optional[List[int]] = None,
) -> SampleOutput:
    all_user_msgs: List[str] = []
    for conv in convs:
        for m in conv:
            if (m.get("role") or "").lower() == "user" and m.get("content") is not None:
                all_user_msgs.append(m["content"])

    need = num_clients * turns_per_client
    if len(all_user_msgs) < need:
        raise ValueError(f"error")

    result: SampleOutput = []
    for i in range(num_clients):
        start = i * turns_per_client
        turns = all_user_msgs[start:start + turns_per_client]

        if per_turn_user_len_list and len(per_turn_user_len_list) > 0:
            length = per_turn_user_len_list[i % len(per_turn_user_len_list)]
        else:
            length = per_turn_user_len

        conv_tuples = [(t, length, fixed_output_len) for t in turns]
        result.append(conv_tuples)
    return result


def get_dataset(dataset_name: str, dataset_path: str,
                num_clients: int, turns_per_client: int,
                per_turn_user_len: int, fixed_output_len: int,
                per_turn_user_len_list: Optional[List[int]] = None,
                stitch_if_short: bool = True) -> SampleOutput:
    if not dataset_path:
        raise ValueError("Please provide --dataset-path")
    if dataset_name == "sharegpt":
        convs = _load_sharegpt(dataset_path)
    elif dataset_name == "ultrachat":
        convs = _load_ultrachat(dataset_path)
    else:
        raise ValueError("dataset-name must be 'sharegpt' or 'ultrachat'")
    sample = build_sample_output(
        convs, turns_per_client, per_turn_user_len, fixed_output_len,
        per_turn_user_len_list=per_turn_user_len_list
    )
    if len(sample) < num_clients and stitch_if_short:
        sample = build_sample_output_stitch(
            convs, num_clients, turns_per_client,
            per_turn_user_len, fixed_output_len,
            per_turn_user_len_list=per_turn_user_len_list
        )

    if len(sample) < num_clients:
        raise ValueError(f"Dataset has only {len(sample)} conversations with >= {turns_per_client} user turns; need {num_clients}.")
    return sample[:num_clients]


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

@dataclass
class RequestFuncInput:
    prompts: List[Tuple[MsgContent, int, int]] 
    api_url: str
    model: str
    extra_request_body: Dict[str, Any]
    messages: List[Dict[str, str]] = field(default_factory=list) 
    finished_prompts: int = 0

@dataclass
class RequestFuncOutput:
    generated_text: List[str] = field(default_factory=list)
    prompt_len: List[int] = field(default_factory=list)             
    prompt_len_with_history: List[int] = field(default_factory=list) 
    output_len: List[int] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    ttft: List[float] = field(default_factory=list)
    itl: List[float] = field(default_factory=list)  
    success: bool = False
    error: str = ""

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
    p90_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p95_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    total_prompt_with_history: int


async def async_request_sglang(
    request_func_input: RequestFuncInput,
    tokenizer: PreTrainedTokenizerBase,
    disable_stream: bool,
    disable_ignore_eos: bool,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    if not api_url.endswith("completions"):
        raise ValueError("API URL must end with '/v1/chat/completions'.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
    }

    output = RequestFuncOutput()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        prompt_idx = request_func_input.finished_prompts
        prompt, input_len_placeholder, max_tokens = request_func_input.prompts[prompt_idx]

        # Force the *new* user message to exactly per_turn_user_len tokens.
        new_user_len = int(input_len_placeholder)
        user_text_raw = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        user_text_fixed = force_len_tokens(user_text_raw, tokenizer, new_user_len)

        # Prepare messages with history
        messages = request_func_input.messages + [{"role": "user", "content": user_text_fixed}]
        payload = {
            "model": request_func_input.model,
            "temperature": 0.0,
            "best_of": 1,
            "stream": not disable_stream,
            "stream_options": {"include_usage": True},
            "ignore_eos": not disable_ignore_eos,
            "messages": messages,
            "max_tokens": max_tokens,
            **request_func_input.extra_request_body,
        }

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st

        # Track server-reported prompt tokens with history (optional)
        prompt_tokens_with_history = None
        completion_tokens = 0

        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
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
                            timestamp = time.perf_counter()

                            # Some servers stream a usage-only frame
                            if "usage" in data and data["usage"]:
                                prompt_tokens_with_history = data["usage"].get("prompt_tokens", prompt_tokens_with_history)
                                completion_tokens = data["usage"].get("completion_tokens", completion_tokens)
                                continue

                            choices = data.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content"):
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft.append(ttft)
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    generated_text += delta["content"]
                                most_recent_timestamp = timestamp

                    output.prompt_len.append(new_user_len)
                    if prompt_tokens_with_history is not None:
                        output.prompt_len_with_history.append(prompt_tokens_with_history)
                    else:
                        output.prompt_len_with_history.append(0)
                    output.output_len.append(completion_tokens)
                    output.generated_text.append(generated_text)
                    output.latency.append(latency)
                    output.success = True

     
                    request_func_input.prompts[prompt_idx] = (
                        prompt, new_user_len, completion_tokens
                    )
                    request_func_input.messages = messages + [{"role": "assistant", "content": generated_text}]
                    request_func_input.finished_prompts = prompt_idx + 1
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            output.error = "".join(traceback.format_exception(*sys.exc_info()))

    if pbar:
        pbar.update(1)
    return output



async def get_requests(
    input_requests_queue: asyncio.Queue,
    request_rate: float,
    num_actual_requests: int,
) -> AsyncGenerator[RequestFuncInput, None]:
    for _ in range(num_actual_requests):
        try:
            request = await asyncio.wait_for(input_requests_queue.get(), timeout=300)
        except Exception as e:
            print(f"exception: {e}")
            break
        yield request
        if request_rate == float("inf"):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)

def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    total_prompt_with_history = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []

    for o in outputs:
        if o.success:
            # assert len(o.generated_text) == len(o.latency) == len(o.ttft)
            assert len(o.generated_text) == len(o.latency)
            if len(o.ttft) < len(o.latency):
                o.ttft += [o.latency[k] for k in range(len(o.ttft), len(o.latency))]
            for j in range(len(o.generated_text)):
                out_len = o.output_len[j]
                output_lens.append(out_len)
                retok_out_len = len(tokenizer.encode(o.generated_text[j], add_special_tokens=False))
                retokenized_output_lens.append(retok_out_len)

                # our "new user" perspective
                total_input += o.prompt_len[j]
                # server perspective if available
                if o.prompt_len_with_history[j] > 0:
                    total_prompt_with_history += o.prompt_len_with_history[j]

                if out_len > 1:
                    tpots.append((o.latency[j] - o.ttft[j]) / (out_len - 1))
                completed += 1
            itls += o.itl
            ttfts += o.ttft
            e2e_latencies += o.latency
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn("All requests failed. Check server and arguments.", stacklevel=2)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s if dur_s > 0 else 0.0,
        input_throughput=total_input / dur_s if dur_s > 0 else 0.0,
        output_throughput=sum(output_lens) / dur_s if dur_s > 0 else 0.0,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s if dur_s > 0 else 0.0,
        total_throughput=(total_input + sum(output_lens)) / dur_s if dur_s > 0 else 0.0,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens)) / dur_s if dur_s > 0 else 0.0,
        mean_ttft_ms=(np.mean(ttfts or 0) * 1000),
        median_ttft_ms=(np.median(ttfts or 0) * 1000),
        std_ttft_ms=(np.std(ttfts or 0) * 1000),
        p90_ttft_ms=(np.percentile(ttfts or 0, 90) * 1000 if ttfts else 0.0),
        p95_ttft_ms=(np.percentile(ttfts or 0, 95) * 1000 if ttfts else 0.0),
        p99_ttft_ms=(np.percentile(ttfts or 0, 99) * 1000 if ttfts else 0.0),
        mean_tpot_ms=(np.mean(tpots or 0) * 1000),
        median_tpot_ms=(np.median(tpots or 0) * 1000),
        std_tpot_ms=(np.std(tpots or 0) * 1000),
        p90_tpot_ms=(np.percentile(tpots or 0, 90) * 1000 if tpots else 0.0),
        p99_tpot_ms=(np.percentile(tpots or 0, 99) * 1000 if tpots else 0.0),
        mean_itl_ms=(np.mean(itls or 0) * 1000),
        median_itl_ms=(np.median(itls or 0) * 1000),
        std_itl_ms=(np.std(itls or 0) * 1000),
        p90_itl_ms=(np.percentile(itls or 0, 90) * 1000 if itls else 0.0),
        p99_itl_ms=(np.percentile(itls or 0, 99) * 1000 if itls else 0.0),
        mean_e2e_latency_ms=(np.mean(e2e_latencies or 0) * 1000),
        median_e2e_latency_ms=(np.median(e2e_latencies or 0) * 1000),
        std_e2e_latency_ms=(np.std(e2e_latencies or 0) * 1000),
        p95_e2e_latency_ms=(np.percentile(e2e_latencies or 0, 95) * 1000 if e2e_latencies else 0.0),
        p99_e2e_latency_ms=(np.percentile(e2e_latencies or 0, 99) * 1000 if e2e_latencies else 0.0),
        concurrency=(np.sum(e2e_latencies or 0) / dur_s if dur_s > 0 else 0.0),
        total_prompt_with_history=total_prompt_with_history,
    )
    return metrics, output_lens

async def benchmark(
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: SampleOutput,
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    extra_request_body: Dict[str, Any],
    disable_stream: bool,
    disable_ignore_eos: bool,
):
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_call(rfi, tok, pbar):
        if semaphore is None:
            return await async_request_sglang(rfi, tok, disable_stream, disable_ignore_eos, pbar)
        async with semaphore:
            return await async_request_sglang(rfi, tok, disable_stream, disable_ignore_eos, pbar)

    num_actual_requests = sum(len(conv) for conv in input_requests)
    print(f"Num of conversations (clients): {len(input_requests)}")
    print(f"Num of total turns: {num_actual_requests}")

    warm = RequestFuncInput(
        model=model_id,
        prompts=input_requests[0][:1],
        api_url=api_url,
        extra_request_body=extra_request_body,
    )
    test_out = await async_request_sglang(
        request_func_input=warm,
        tokenizer=tokenizer,
        disable_stream=disable_stream,
        disable_ignore_eos=disable_ignore_eos,
        pbar=None,
    )
    if not test_out.success:
        raise RuntimeError(f"Initial test failed: {test_out.error}")
    print("Warmup ok. Flushing SGLang cache...")
    try:
        requests.post(base_url + "/flush_cache", timeout=5)
    except Exception:
        pass
    time.sleep(1.0)

    states: List[RequestFuncInput] = [
        RequestFuncInput(
            model=model_id,
            prompts=conv,
            api_url=api_url,
            extra_request_body=extra_request_body,
        )
        for conv in input_requests
    ]

    outputs: List[RequestFuncOutput] = []
    pbar = None if disable_tqdm else tqdm(total=num_actual_requests)

    def available_indices() -> List[int]:
        return [i for i, s in enumerate(states) if s.finished_prompts < len(s.prompts)]

    start_t = time.perf_counter()
    in_flight: List[asyncio.Task] = []

    async def maybe_launch_one() -> bool:
        idxs = available_indices()
        if not idxs:
            return False
        i = random.choice(idxs)
        task = asyncio.create_task(limited_call(states[i], tokenizer, pbar))
        in_flight.append(task)
        return True

    target_conc = max_concurrency if max_concurrency else len(states)
    while len(in_flight) < target_conc and await maybe_launch_one():
        pass

    while in_flight or available_indices():
        if request_rate != float("inf"):
            await asyncio.sleep(np.random.exponential(1.0 / max(request_rate, 1e-9)))
            if len(in_flight) < target_conc:
                await maybe_launch_one()
        else:
            while len(in_flight) < target_conc and await maybe_launch_one():
                pass

        if in_flight:
            done, pending = await asyncio.wait(in_flight, timeout=0.0, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    out = await t
                    outputs.append(out)
                except Exception:
                    pass
            in_flight = list(pending)

        while len(in_flight) < target_conc and await maybe_launch_one():
            pass

        await asyncio.sleep(0)

    if pbar:
        pbar.close()

    dur_s = time.perf_counter() - start_t
    metrics, output_lens = calculate_metrics(outputs, dur_s, tokenizer)

    print("\n" + "=" * 50)
    print(f"{' Serving Benchmark Result ':=^50}")
    print(f"{'Backend:':<40} {'sglang':<10}")
    print(f"{'Traffic request rate:':<40} {request_rate}")
    print(f"{'Max request concurrency:':<40} {max_concurrency if max_concurrency else 'not set'}")
    print(f"{'Successful requests:':<40} {metrics.completed}")
    print(f"{'Benchmark duration (s):':<40} {dur_s:<10.2f}")
    print(f"{'Total input tokens (new user only):':<40} {metrics.total_input}")
    print(f"{'Total prompt tokens w/ history (srv)':<40} {metrics.total_prompt_with_history}")
    print(f"{'Total generated tokens:':<40} {metrics.total_output}")
    print(f"{'Total generated tokens (retokenized):':<40} {metrics.total_output_retokenized}")
    print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:<10.2f}")
    print(f"{'Input token throughput (tok/s):':<40} {metrics.input_throughput:<10.2f}")
    print(f"{'Output token throughput (tok/s):':<40} {metrics.output_throughput:<10.2f}")
    print(f"{'Total token throughput (tok/s):':<40} {metrics.total_throughput:<10.2f}")
    print(f"{'Concurrency:':<40} {metrics.concurrency:<10.2f}")

    print(f"{'End-to-End Latency':-^50}")
    print(f"{'Mean E2E Latency (ms):':<40} {metrics.mean_e2e_latency_ms:<10.2f}")
    print(f"{'Median E2E Latency (ms):':<40} {metrics.median_e2e_latency_ms:<10.2f}")
    print(f"{'P95 E2E Latency (ms):':<40} {metrics.p95_e2e_latency_ms:<10.2f}")
    print(f"{'P99 E2E Latency (ms):':<40} {metrics.p99_e2e_latency_ms:<10.2f}")

    print(f"{'Time to First Token':-^50}")
    print(f"{'Mean TTFT (ms):':<40} {metrics.mean_ttft_ms:<10.2f}")
    print(f"{'Median TTFT (ms):':<40} {metrics.median_ttft_ms:<10.2f}")
    print(f"{'P90 TTFT (ms):':<40} {metrics.p90_ttft_ms:<10.2f}")
    print(f"{'P95 TTFT (ms):':<40} {metrics.p95_ttft_ms:<10.2f}")
    print(f"{'P99 TTFT (ms):':<40} {metrics.p99_ttft_ms:<10.2f}")

    print(f"{'Time per Output Token (excl. 1st)':-^50}")
    print(f"{'Mean TPOT (ms):':<40} {metrics.mean_tpot_ms:<10.2f}")
    print(f"{'Median TPOT (ms):':<40} {metrics.median_tpot_ms:<10.2f}")
    print(f"{'P90 TPOT (ms):':<40} {metrics.p90_tpot_ms:<10.2f}")
    print(f"{'P99 TPOT (ms):':<40} {metrics.p99_tpot_ms:<10.2f}")

    print(f"{'Inter-token Latency':-^50}")
    print(f"{'Mean ITL (ms):':<40} {metrics.mean_itl_ms:<10.2f}")
    print(f"{'Median ITL (ms):':<40} {metrics.median_itl_ms:<10.2f}")
    print(f"{'P90 ITL (ms):':<40} {metrics.p90_itl_ms:<10.2f}")
    print(f"{'P99 ITL (ms):':<40} {metrics.p99_itl_ms:<10.2f}")
    print("=" * 50)

    result = {
        "dataset_name": args.dataset_name,
        "num_clients": args.num_clients,
        "turns_per_client": args.turns_per_client,
        "fixed_output_len": args.fixed_output_len,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        "duration": dur_s,
        "completed": metrics.completed,
        "total_input_tokens_new_user": metrics.total_input,
        "total_prompt_tokens_with_history": metrics.total_prompt_with_history,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "total_throughput": metrics.total_throughput,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
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
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p90_itl_ms": metrics.p90_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
    }
    out_name = args.output_file
    with open(out_name, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    details = {
        "input_lens_new_user": [o.prompt_len for o in outputs],
        "prompt_lens_with_history": [o.prompt_len_with_history for o in outputs],
        "output_lens": [o.output_len for o in outputs],
        "ttfts": [o.ttft for o in outputs],
        "itls": [o.itl for o in outputs],
        "errors": [o.error for o in outputs],
    }
    return details



def run(args):
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    base_url = f"http://{args.host}:{args.port}"


    # Detect model id if not given
    model_id = args.model
    if not model_id:
        try:
            resp = requests.get(f"http://{args.host}:{args.port}/v1/models", timeout=5)
            model_list = resp.json().get("data", [])
            model_id = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch /v1/models: {e}")
    if not model_id:
        print("No model specified or found via /v1/models. Use --model.")
        sys.exit(1)

    # Load dataset: shape -> List[conversation] with each conv a list of (prompt_text, placeholder_len, max_tokens)

    per_len_list = None
    if args.per_turn_user_len_list:
        per_len_list = [int(x) for x in args.per_turn_user_len_list.split(",") if x.strip()]
        if not per_len_list:
            print("Invalid --per-turn-user-len-list.")
            sys.exit(1)

    fixed_out = args.fixed_output_len
    sample: SampleOutput = get_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        num_clients=args.num_clients,
        turns_per_client=args.turns_per_client,
        per_turn_user_len=args.per_turn_user_len,   
        fixed_output_len=fixed_out,
        per_turn_user_len_list=per_len_list,        
    )

    turns_list = None
    if getattr(args, "turns_per_client_list", None):
        turns_list = [int(x) for x in args.turns_per_client_list.split(",") if x.strip()]

    if turns_list:
        rng = random.Random(args.seed)
        cursor = 0
        picked: SampleOutput = []
        for conv in sample:
            if args.turns_pick == "rr":
                T = turns_list[cursor % len(turns_list)]
                cursor += 1
            else:
                T = rng.choice(turns_list)
            T = max(1, min(T, len(conv)))
            picked.append(conv[:T])
        sample = picked


    tokenizer = get_tokenizer(args.tokenizer or model_id)

    # Default max_concurrency to num_clients if not set
    max_conc = args.max_concurrency if args.max_concurrency is not None else args.num_clients

    # Run
    return asyncio.run(
        benchmark(
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=sample,
            request_rate=args.request_rate,
            max_concurrency=max_conc,
            disable_tqdm=args.disable_tqdm,
            extra_request_body=json.loads(args.extra_request_body) if args.extra_request_body else {},
            disable_stream=args.disable_stream,
            disable_ignore_eos=args.disable_ignore_eos,
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multi-turn SGLang Benchmark (preserve history, per-turn new user = fixed tokens)")
    # server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)

    # dataset
    parser.add_argument("--dataset-name", type=str, choices=["sharegpt", "ultrachat"], required=True)
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset file (json/jsonl)")
    parser.add_argument("--num-clients", type=int, required=True, help="Number of conversations (clients)")
    parser.add_argument("--turns-per-client", type=int, default=8, help="User turns per conversation")

    # per-turn controls
    parser.add_argument("--per-turn-user-len", type=int, default=4096, help="New user tokens per turn (forced under local tokenizer)")
    parser.add_argument("--fixed-output-len", type=int, default=256, help="Max tokens to generate per turn")
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Disable ignoring EOS in server")
    parser.add_argument("--disable-stream", action="store_true", help="Disable streaming (TTFT/ITL unavailable)")
    parser.add_argument("--extra-request-body", type=str, help='JSON string for extra gen params, e.g. \'{"top_p":1,"temperature":0}\'')

    # load & traffic
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--output-file", type=str)

    parser.add_argument(
        "--per-turn-user-len-list",
        type=str
    )

    parser.add_argument(
        "--turns-per-client-list",
        type=str
    )
    parser.add_argument(
        "--turns-pick",
        choices=["rr","random"],
        default="rr"
    )

    args = parser.parse_args()
    run(args)
