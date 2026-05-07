"""Visual-information-decay ablation: API-model runner.

Runs one OpenAI/Anthropic/Together/Google model on the 300 sigma-blurred
original probes and writes per-probe responses to
results/blur_sweep/<safe_model>__sigma<sigma>.jsonl.

Sigma=inf is the no-image baseline (runner uses --no-image flag).

Imports the call_one*/parse_letter helpers from bench_run_baseline.py so all
four providers are supported with no code duplication.

Run:
    python3 scripts/bench_blur_sweep_run.py --model gpt-4o --sigma 8
    python3 scripts/bench_blur_sweep_run.py --model claude-sonnet-4-6 --sigma 0
    python3 scripts/bench_blur_sweep_run.py --model 'Qwen/Qwen3.5-397B-A17B' --sigma inf
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from api_models import _load_dotenv  # noqa: E402
from bench_run_baseline import (  # noqa: E402
    REASONING_MODELS, REASONING_TOKENS, DEFAULT_TOKENS,
    detect_provider, build_mcq_prompt, parse_letter, call_one,
)

_load_dotenv()

BENCH = ROOT / "data/medvigil_v1"
SWEEP_ROOT = BENCH / "_blur_sweep"
RESULTS = ROOT / "results/blur_sweep"


def load_sigma_probes(sigma: str) -> list[dict]:
    if sigma == "inf":
        # Use sigma0 CSV; runner will skip the image at request time.
        path = SWEEP_ROOT / "probes_sigma0.csv"
    else:
        path = SWEEP_ROOT / f"probes_sigma{sigma}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run bench_blur_sweep_make.py first")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(probe: dict) -> Path | None:
    img_rel = probe.get("image_file", "").strip()
    if not img_rel:
        return None
    # CSV stores e.g. images/MVB-0001.jpg or images_blur/sigma8/MVB-0001.jpg
    return BENCH / img_rel


def load_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["probe_id"])
            except Exception:
                continue
    return done


async def process_one(client, sem, model, probe, max_tokens, no_image):
    pid = probe["probe_id"]
    img = None if no_image else resolve_image_path(probe)
    prompt = build_mcq_prompt(probe)
    async with sem:
        for attempt in range(4):
            try:
                t0 = time.time()
                raw, usage = await call_one(client, model, img, prompt, max_tokens)
                row = {
                    "probe_id": pid,
                    "case_id": probe["case_id"],
                    "model_id": model,
                    "image_file": probe.get("image_file", ""),
                    "model_answer": raw,
                    "model_letter": parse_letter(raw),
                    "correct_letter": probe.get("correct_letter", ""),
                    "latency_ms": int((time.time() - t0) * 1000),
                    "usage": usage,
                    "status": "ok",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                return row
            except Exception as e:
                msg = str(e).lower()
                if attempt < 3 and any(k in msg for k in ("rate", "429", "500", "502", "503", "overloaded", "timeout", "connection")):
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    continue
                return {
                    "probe_id": pid,
                    "case_id": probe.get("case_id", ""),
                    "model_id": model,
                    "model_answer": "",
                    "status": f"error: {str(e)[:200]}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }


async def main_async(args):
    provider = detect_provider(args.model)
    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    elif provider == "deepseek":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    elif provider == "together":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
    else:
        raise ValueError(provider)

    no_image = (args.sigma == "inf")
    probes = load_sigma_probes(args.sigma)
    print(f"[init] provider={provider} model={args.model} sigma={args.sigma} no_image={no_image} probes={len(probes)}", flush=True)

    safe = args.model.replace("/", "--")
    out_path = RESULTS / f"{safe}__sigma{args.sigma}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path) if args.resume else set()
    todo = [p for p in probes if p["probe_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[init] resume={len(done)} todo={len(todo)}", flush=True)

    if args.model in REASONING_MODELS:
        max_tokens = REASONING_TOKENS
    elif provider == "together":
        max_tokens = 2000
    else:
        max_tokens = DEFAULT_TOKENS

    sem = asyncio.Semaphore(args.concurrency)
    out_f = open(out_path, "a" if args.resume else "w")
    tasks = [asyncio.create_task(process_one(client, sem, args.model, p, max_tokens, no_image)) for p in todo]
    n_ok = n_err = n_match = 0
    t0 = time.time()
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        row = await fut
        if row["status"] == "ok":
            n_ok += 1
            if row.get("model_letter") == row.get("correct_letter"):
                n_match += 1
        else:
            n_err += 1
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        if i % 30 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(todo) - i) / max(rate, 1e-3)
            extra = f"  rough_acc={n_match/max(n_ok,1):.1%}"
            print(f"[{args.model} sigma={args.sigma}] [{i}/{len(todo)}] ok={n_ok} err={n_err} {rate*60:.0f}/min ETA {eta/60:.1f}min{extra}", flush=True)
    out_f.close()
    print(f"[{args.model} sigma={args.sigma}] done. ok={n_ok} err={n_err} rough_acc={n_match/max(n_ok,1):.2%} -> {out_path.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sigma", required=True, help="0, 2, 4, 8, 16, 32, 64, or inf")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
