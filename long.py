import argparse
import ast
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, AsyncGenerator
import re
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

@dataclass
class Req:
    prompt: str
    prompt_len: int
    max_new_tokens: int

@dataclass
class Res:
    ok: bool
    ttft_s: float
    e2e_s: float
    out_len: int
    err: str = ""


def _parse_qa_pairs(obj):
    if obj in (None, "none", []):
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            v = ast.literal_eval(obj)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []

CTX_LIMIT = 8192      
CTX_SAFETY = 64       

def _truncate_by_tokens(tokenizer, text: str, budget: int) -> str:
    ids = tokenizer.encode(text)
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:max(budget, 0)], skip_special_tokens=True)
    
    
def load_loogle_shared_groups(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int,
    num_groups: Optional[int],
    max_q_per_group: Optional[int],
    drop_no_qa: bool = True,
    rr_doc_lens: Optional[List[int]] = None,
    rr_start_index: int = 0,
) -> List[List[Req]]:
    records = []  
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc = data.get("input", "")
            qa_pairs = _parse_qa_pairs(data.get("qa_pairs"))
            if (not qa_pairs) and drop_no_qa:
                continue
            records.append((doc, qa_pairs))

    if not records:
        return []

    if num_groups is None or num_groups >= len(records):
        chosen = records
    else:
        chosen = random.sample(records, num_groups)

    rr_idx = rr_start_index if rr_doc_lens else 0

    groups: List[List[Req]] = []
    for doc, qa_pairs in chosen:
        group: List[Req] = []
        if not qa_pairs:
            questions = ["Please summarize the input."]
        else:
            K = max_q_per_group if max_q_per_group is not None else len(qa_pairs)
            questions = [qa.get("Q", "") for qa in qa_pairs[:K]]

        def _tmpl(q: str) -> str:
            return f"Input:\n{{DOC}}\n\nQuestion: {q}"

        tmpl_max = 0
        for q in questions:
            tmpl_max = max(
                tmpl_max,
                len(tokenizer.encode(_tmpl(q), add_special_tokens=False))
            )

        budget = CTX_LIMIT - CTX_SAFETY - fixed_output_len - tmpl_max
        budget = max(budget, 0)

        if rr_doc_lens:
            target = rr_doc_lens[rr_idx % len(rr_doc_lens)]
            rr_idx += 1
            doc_budget = min(target, budget)
        else:
            doc_budget = budget
        doc_trunc = _truncate_by_tokens(tokenizer, doc, doc_budget)


        for q in questions:
            prompt = f"Input:\n{doc_trunc}\n\nQuestion: {q}"
            plen = len(tokenizer.encode(prompt, add_special_tokens=False))
            group.append(Req(prompt, plen, fixed_output_len))

        if group:
            groups.append(group)

    return groups


def linearize_groups(
    groups: List[List[Req]],
    order: str = "random",
) -> List[Req]:
    order = order.lower()
    if order in ("grouped", "best"):
        return [r for g in groups for r in g]
    elif order in ("round_robin", "rr", "worst"):
        L = max(len(g) for g in groups)
        seq = []
        for i in range(L):
            for g in groups:
                if i < len(g):
                    seq.append(g[i])
        return seq
    else:
        flat = [r for g in groups for r in g]
        random.shuffle(flat)
        return flat


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

async def send_sglang_generate(
    req: Req, url: str, stream: bool = True
) -> Res:
    payload = {
        "text": req.prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": req.max_new_tokens,
            "ignore_eos": True,
        },
        "stream": stream,
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    headers = {"Accept": "text/event-stream"} if stream else {}

    st = time.perf_counter()
    ttft = 0.0
    out_len = 0

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as sess:
        try:
            async with sess.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    return Res(False, 0.0, 0.0, 0, f"HTTP {resp.status} {resp.reason}")

                if not stream:
                    data = await resp.json(content_type=None)
                    text = data.get("text", "") or ""
                    meta = data.get("meta_info") or {}
                    out_len = int(meta.get("completion_tokens") or 0)
                    e2e = time.perf_counter() - st
                    return Res(True, e2e, e2e, out_len) 

                buf = ""
                done = False
                async for raw in resp.content.iter_any():
                    if not raw:
                        continue
                    buf += raw.decode("utf-8", errors="ignore")
                    if "\r\n" in buf:
                        buf = buf.replace("\r\n", "\n")

                    while True:
                        sep = buf.find("\n\n")
                        if sep == -1:
                            break
                        event = buf[:sep]
                        buf = buf[sep + 2 :]

                        data_lines = []
                        for line in event.split("\n"):
                            if not line:
                                continue
                            if line.startswith(":"):
                                continue
                            if line.startswith("data:"):
                                data_lines.append(line[len("data:"):].lstrip())
                        if not data_lines:
                            continue

                        data_str = "\n".join(data_lines).strip()
                        if data_str == "[DONE]":
                            done = True
                            e2e = time.perf_counter() - st
                            return Res(True, ttft or e2e, e2e, out_len)

                        try:
                            data = json.loads(data_str)
                        except Exception:
                            continue

                        txt = data.get("text")
                        if txt:
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                            meta = data.get("meta_info") or {}
                            if "completion_tokens" in meta:
                                out_len = int(meta["completion_tokens"])
                            else:
                                out_len = max(out_len, 1)

                e2e = time.perf_counter() - st
                # return Res(True, ttft or e2e, e2e, out_len)
                return Res(False, ttft or 0.0, e2e, out_len, "incomplete stream (no [DONE])")

        except Exception as e:
            return Res(False, 0.0, 0.0, 0, f"exc: {e}")


async def poisson_requests(
    seq: List[Req], lam: float
) -> AsyncGenerator[Req, None]:
    for r in seq:
        yield r
        if lam == float("inf"):
            continue
        await asyncio.sleep(np.random.exponential(1.0 / max(lam, 1e-9)))

def pct(v: List[float], p: float) -> float:
    if not v:
        return 0.0
    return float(np.percentile(np.array(v), p))

def summarize(results: List[Res], total_input_tokens: int, dur_s: float):
    oks = [r for r in results if r.ok]
    ttfts = [r.ttft_s * 1000 for r in oks]
    e2es = [r.e2e_s * 1000 for r in oks]
    out_tokens = sum(r.out_len for r in oks)

    req_tput = len(oks) / dur_s if dur_s > 0 else 0.0
    in_tok_tput = total_input_tokens / dur_s if dur_s > 0 else 0.0
    out_tok_tput = out_tokens / dur_s if dur_s > 0 else 0.0
    total_tok_tput = (total_input_tokens + out_tokens) / dur_s if dur_s > 0 else 0.0
    concurrency = sum(r.e2e_s for r in oks) / dur_s if dur_s > 0 else 0.0

    print("\n========== Loogle Shared-Prefix Benchmark ==========")
    print(f"Successful requests:            {len(oks)}")
    print(f"Duration (s):                   {dur_s:.2f}")
    print(f"Total input tokens:             {total_input_tokens}")
    print(f"Total output tokens (approx):   {out_tokens}")
    print(f"Request throughput (req/s):     {req_tput:.2f}")
    print(f"Input token throughput (tok/s): {in_tok_tput:.2f}")
    print(f"Output token throughput (tok/s):{out_tok_tput:.2f}")
    print(f"Total token throughput (tok/s): {total_tok_tput:.2f}")
    print(f"Concurrency (Little's law est): {concurrency:.2f}")
    print("---------------- Time To First Token ---------------")
    print(f"Mean TTFT (ms):                 {np.mean(ttfts) if ttfts else 0.0:.2f}")
    print(f"Median TTFT (p50, ms):          {pct(ttfts,50):.2f}")
    print(f"P90 TTFT (ms):                  {pct(ttfts,90):.2f}")
    print(f"P95 TTFT (ms):                  {pct(ttfts,95):.2f}")
    print(f"P99 TTFT (ms):                  {pct(ttfts,99):.2f}")
    print("------------------- E2E Latency --------------------")
    print(f"Mean E2E (ms):                  {np.mean(e2es) if e2es else 0.0:.2f}")
    print(f"Median E2E (p50, ms):           {pct(e2es,50):.2f}")
    print(f"P90 E2E (ms):                   {pct(e2es,90):.2f}")
    print(f"P95 E2E (ms):                   {pct(e2es,95):.2f}")
    print(f"P99 E2E (ms):                   {pct(e2es,99):.2f}")
    print("====================================================\n")


async def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    doc_len_range = None
    if args.doc_len_range:
        parts = [int(x.strip()) for x in args.doc_len_range.split(",")]
        if len(parts) == 2:
            lo, hi = parts
            if lo > hi:
                lo, hi = hi, lo
            doc_len_range = (max(lo, 0), max(hi, 0))
    
    doc_len_list = None
    if args.doc_lens:
        doc_len_list = [int(x) for x in re.split(r"[,\s]+", args.doc_lens.strip()) if x]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    groups = load_loogle_shared_groups(
        path=args.dataset_path,
        tokenizer=tokenizer,
        fixed_output_len=args.fixed_output_len,
        num_groups=args.num_groups,
        max_q_per_group=args.max_q_per_group,
        drop_no_qa=not args.keep_no_qa,
        rr_doc_lens=doc_len_list,
        rr_start_index=0,
    )
    if not groups:
        print("No usable groups from dataset. Check your dataset path/content.")
        sys.exit(1)

    seq = linearize_groups(groups, order=args.order)
    total_input_tokens = sum(r.prompt_len for r in seq)

    if args.backend == "sglang":
        url = f"http://{args.host}:{args.port}/generate"
        sender = lambda r: send_sglang_generate(r, url, stream=not args.disable_stream)


    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    async def limited_send(r: Req):
        if sem is None:
            return await sender(r)
        async with sem:
            return await sender(r)

    pbar = tqdm(total=len(seq), disable=args.no_tqdm)
    tasks: List[asyncio.Task] = []

    t0 = time.perf_counter()
    async for req in poisson_requests(seq, args.request_rate):
        tasks.append(asyncio.create_task(limited_send(req)))
        pbar.update(1)
    results: List[Res] = await asyncio.gather(*tasks)
    pbar.close()
    dur_s = time.perf_counter() - t0

    summarize(results, total_input_tokens, dur_s)


    oks = [r for r in results if r.ok]
    ttfts = [r.ttft_s * 1000 for r in oks]
    e2es = [r.e2e_s * 1000 for r in oks]
    out_tokens = sum(r.out_len for r in oks)

    req_tput = len(oks) / dur_s if dur_s > 0 else 0.0
    in_tok_tput = total_input_tokens / dur_s if dur_s > 0 else 0.0
    out_tok_tput = out_tokens / dur_s if dur_s > 0 else 0.0
    total_tok_tput = (total_input_tokens + out_tokens) / dur_s if dur_s > 0 else 0.0
    

    if args.output_file:
        out = {
            "backend": args.backend,
            "host": args.host,
            "port": args.port,
            "model": args.model,
            "order": args.order,
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
            "num_groups": args.num_groups,
            "max_q_per_group": args.max_q_per_group,
            "fixed_output_len": args.fixed_output_len,
            "duration_s": dur_s,
            "n_success": sum(1 for r in results if r.ok),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens_approx": sum(r.out_len for r in results if r.ok),
            "request_throughput": round(req_tput, 3),
            "input_token_throughput_s": round(in_tok_tput, 2),
            "output_token_throughput_s": round(out_tok_tput, 2),
            "total_token_throughput_s": round(total_tok_tput, 2),
            "ttft_ms_mean": float(np.mean(ttfts)) if ttfts else 0.0,
            "ttft_ms_p50":  pct(ttfts, 50),
            "p95_ttft_ms":  pct(ttfts, 95),
            "ttft_ms_p99":  pct(ttfts, 99),

            "e2e_ms_mean": float(np.mean(e2es)) if e2es else 0.0,
            "e2e_ms_p50":  pct(e2es, 50),
            "e2e_ms_p95":  pct(e2es, 95),
            "e2e_ms_p99":  pct(e2es, 99),
            
            # "ttft_ms": [r.ttft_s * 1000 for r in results if r.ok],
            # "e2e_ms": [r.e2e_s * 1000 for r in results if r.ok],
        }
        with open(args.output_file, "w") as f:
            json.dump(out, f, ensure_ascii=False)


def parse_args():
    ap = argparse.ArgumentParser("Loogle shared-prefix benchmark")
    ap.add_argument("--backend", type=str, choices=["sglang", "vllm", "sglang-oai"], default="sglang")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset-path", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--fixed-output-len", type=int, default=4)

    ap.add_argument("--num-groups", type=int, default=None)
    ap.add_argument("--max-q-per-group", type=int, default=None)
    ap.add_argument("--keep-no-qa", action="store_true")

    ap.add_argument("--order", type=str, default="random",
                    choices=["random", "grouped", "best", "round_robin", "rr", "worst"])
    ap.add_argument("--request-rate", type=float, default=float("inf"))
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--disable-stream", action="store_true")

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-tqdm", action="store_true")
    ap.add_argument("--doc-len-range", type=str, default=None)
    ap.add_argument("--output-file", type=str, default=None)
    ap.add_argument(
        "--doc-lens",
        type=str,
        default=None
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Interrupted")
