"""Single-model baseline runner for MedVLM-Bench v1.

Runs one OpenAI model on the entire probes_mcq.csv (or probes_open.csv)
and writes per-probe responses to results/baselines/<model>__<format>.jsonl.

Async + semaphore for concurrency, resumable (per-probe-id skip),
per-call latency tracking, retry-with-backoff on rate-limit errors.

Run:
    python3 scripts/bench_run_baseline.py --model <MODEL> --format mcq
    python3 scripts/bench_run_baseline.py --model gpt-4o --format open --concurrency 8
    python3 scripts/bench_run_baseline.py --model gpt-5.4-nano --format mcq --limit 10  # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_models import _load_dotenv  # noqa: E402

_load_dotenv()

BENCH = ROOT / "data/medvlm_bench_v1"
PROBES_MCQ = BENCH / "probes_mcq.csv"
PROBES_OPEN = BENCH / "probes_open.csv"
RESULTS = ROOT / "results/baselines"

REASONING_MODELS = {"<MODEL>"}      # use larger token budget for reasoning models
DEFAULT_TOKENS = 600
REASONING_TOKENS = 1200


def detect_provider(model: str) -> str:
    if model.startswith("gpt-"): return "openai"
    if model.startswith("claude-"): return "anthropic"
    if model.startswith("gemini-"): return "google"
    if model.startswith("deepseek-") and "/" not in model: return "deepseek"
    if "/" in model: return "together"   # meta-llama/..., Qwen/..., moonshotai/..., google/...
    raise ValueError(f"unknown provider for model_id: {model}")


def encode_image(p: Path) -> tuple[str, str]:
    ext = p.suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/jpeg")
    return base64.standard_b64encode(p.read_bytes()).decode(), media


def build_mcq_prompt(probe: dict) -> str:
    parts = [probe["question"], ""]
    for L in "ABCDE":
        c = probe.get(f"choice_{L}", "").strip()
        if c:
            parts.append(f"{L}) {c}")
    parts.append("")
    parts.append("Reply with ONLY the single letter of the correct option (A, B, C, D, or E). No explanation.")
    return "\n".join(parts)


def build_open_prompt(probe: dict) -> str:
    return probe["question"]


def resolve_image_path(probe: dict) -> Path:
    img_rel = probe.get("image_file", "").strip()
    if not img_rel:
        return None
    if img_rel.startswith("images_perturbed/"):
        return BENCH / img_rel
    return BENCH / "images" / img_rel


def parse_letter(text: str) -> str:
    """Extract the model's chosen letter from its response. Permissive."""
    if not text:
        return ""
    s = text.strip().upper()
    m = re.match(r"^\s*[\(\[\*]*\s*([ABCDE])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCDE])[\.\)]", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCDE])\b", s)
    if m:
        return m.group(1)
    return ""


async def call_one_openai(client, model: str, image_path: Path, prompt: str,
                           max_tokens: int) -> tuple[str, dict]:
    msgs_content = []
    if image_path is not None and image_path.exists():
        b64, media = encode_image(image_path)
        msgs_content.append({"type": "image_url",
                             "image_url": {"url": f"data:{media};base64,{b64}"}})
    msgs_content.append({"type": "text", "text": prompt})
    response = await client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": msgs_content}],
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    return raw, usage


async def call_one_anthropic(client, model: str, image_path: Path, prompt: str,
                              max_tokens: int) -> tuple[str, dict]:
    if image_path is not None and image_path.exists():
        b64, media = encode_image(image_path)
        content = [
            {"type": "image",
             "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    r = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(b.text for b in r.content if hasattr(b, "text"))
    usage = {
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
    }
    return raw, usage


async def call_one_google(client, model: str, image_path: Path, prompt: str,
                           max_tokens: int) -> tuple[str, dict]:
    from google.genai import types as genai_types
    from PIL import Image as PILImage
    contents = []
    if image_path is not None and image_path.exists():
        contents.append(PILImage.open(image_path).convert("RGB"))
    contents.append(prompt)
    cfg = genai_types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=max_tokens,
    )
    r = await client.aio.models.generate_content(
        model=model, contents=contents, config=cfg,
    )
    raw = (r.text or "")
    um = getattr(r, "usage_metadata", None)
    usage = {}
    if um is not None:
        usage = {
            "prompt_token_count": getattr(um, "prompt_token_count", None),
            "candidates_token_count": getattr(um, "candidates_token_count", None),
            "total_token_count": getattr(um, "total_token_count", None),
        }
    return raw, usage


async def call_one_together(client, model: str, image_path: Path, prompt: str,
                             max_tokens: int) -> tuple[str, dict]:
    """Together AI API: OpenAI-compatible, supports vision via image_url
    block, uses `max_tokens` rather than `max_completion_tokens`.
    """
    msgs_content = []
    if image_path is not None and image_path.exists():
        b64, media = encode_image(image_path)
        msgs_content.append({"type": "image_url",
                             "image_url": {"url": f"data:{media};base64,{b64}"}})
    msgs_content.append({"type": "text", "text": prompt})
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": msgs_content}],
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    return raw, usage


async def call_one_deepseek(client, model: str, image_path: Path, prompt: str,
                             max_tokens: int) -> tuple[str, dict]:
    """DeepSeek API is OpenAI-compatible but does NOT accept image inputs.
    The image_path argument is ignored; the model is queried text-only.
    This makes DeepSeek a useful 'blind LM' baseline for measuring
    language-prior ceiling on visual probes.
    """
    response = await client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    usage["image_used"] = False
    return raw, usage


async def call_one(client, model: str, image_path: Path, prompt: str,
                    max_tokens: int) -> tuple[str, dict]:
    provider = detect_provider(model)
    if provider == "openai":
        return await call_one_openai(client, model, image_path, prompt, max_tokens)
    if provider == "anthropic":
        return await call_one_anthropic(client, model, image_path, prompt, max_tokens)
    if provider == "google":
        return await call_one_google(client, model, image_path, prompt, max_tokens)
    if provider == "deepseek":
        return await call_one_deepseek(client, model, image_path, prompt, max_tokens)
    if provider == "together":
        return await call_one_together(client, model, image_path, prompt, max_tokens)
    raise ValueError(f"unknown provider for {model}")


def load_probes(format_: str) -> list[dict]:
    path = PROBES_MCQ if format_ == "mcq" else PROBES_OPEN
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r["probe_id"])
            except Exception:
                continue
    return done


async def process_one(client, sem, model: str, format_: str, probe: dict,
                       max_tokens: int, no_image: bool = False) -> dict:
    pid = probe["probe_id"]
    img = None if no_image else resolve_image_path(probe)
    prompt = build_mcq_prompt(probe) if format_ == "mcq" else build_open_prompt(probe)

    async with sem:
        for attempt in range(4):
            try:
                t0 = time.time()
                raw, usage = await call_one(client, model, img, prompt, max_tokens)
                dt_ms = int((time.time() - t0) * 1000)
                row = {
                    "probe_id": pid,
                    "case_id": probe["case_id"],
                    "probe_kind": probe["probe_kind"],
                    "format": format_,
                    "model_id": model,
                    "image_file": probe.get("image_file", ""),
                    "question": probe.get("question", ""),
                    "model_answer": raw,
                    "latency_ms": dt_ms,
                    "usage": usage,
                    "status": "ok",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if format_ == "mcq":
                    row["model_letter"] = parse_letter(raw)
                    row["correct_letter"] = probe.get("correct_letter", "")
                    row["expected_behavior"] = probe.get("expected_behavior", "")
                else:
                    row["expected_answer_or_set"] = probe.get("expected_answer_or_set", "")
                    row["halluc_explanation"] = probe.get("halluc_explanation", "")
                    row["expected_behavior"] = probe.get("expected_behavior", "")
                return row
            except Exception as e:
                msg = str(e).lower()
                if attempt < 3 and any(k in msg for k in ("rate", "429", "500", "502", "503", "overloaded", "timeout", "connection")):
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                return {
                    "probe_id": pid,
                    "case_id": probe.get("case_id", ""),
                    "probe_kind": probe.get("probe_kind", ""),
                    "format": format_,
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
        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
    elif provider == "together":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
    else:
        raise ValueError(provider)
    print(f"[init] provider={provider} model={args.model}", flush=True)

    probes = load_probes(args.format)
    print(f"[init] {len(probes)} probes loaded ({args.format})", flush=True)

    safe_model_name = args.model.replace("/", "--")
    fmt_suffix = f"{args.format}_noimg" if args.no_image else args.format
    out_path = RESULTS / f"{safe_model_name}__{fmt_suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path) if args.resume else set()
    if done:
        print(f"[init] resume: {len(done)} already in {out_path.name}", flush=True)

    todo = [p for p in probes if p["probe_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[init] todo: {len(todo)} (skipping {len(probes) - len(todo) - (0 if args.limit else 0)} done; "
          f"model={args.model})", flush=True)

    if args.shuffle:
        random.shuffle(todo)

    if args.model in REASONING_MODELS:
        max_tokens = REASONING_TOKENS
    elif provider == "together":
        # Together-hosted models (Kimi, Qwen3.5, Llama, Gemma) often consume
        # reasoning tokens before producing visible output; budget generously.
        max_tokens = 2000
    else:
        max_tokens = DEFAULT_TOKENS
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = n_err = 0
    correct_letter_match = 0  # MCQ only

    out_f = open(out_path, "a" if args.resume else "w")
    tasks = [asyncio.create_task(process_one(client, sem, args.model, args.format, p, max_tokens, no_image=args.no_image))
             for p in todo]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        row = await fut
        if row["status"] == "ok":
            n_ok += 1
            if args.format == "mcq" and row.get("model_letter") == row.get("correct_letter"):
                correct_letter_match += 1
        else:
            n_err += 1
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        if i % 50 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(todo) - i) / max(rate, 1e-3)
            extra = ""
            if args.format == "mcq" and n_ok:
                extra = f"  rough_acc={correct_letter_match/n_ok:.2%}"
            print(f"[{args.model}] [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                  f"{rate*60:.0f}/min ETA {eta/60:.1f}min{extra}", flush=True)
    out_f.close()

    print(f"\n[{args.model}] done: ok={n_ok} err={n_err}  total {(time.time()-t0)/60:.1f}min", flush=True)
    if args.format == "mcq" and n_ok:
        print(f"[{args.model}] rough exact-letter accuracy on completed: {correct_letter_match/n_ok:.2%}", flush=True)
    print(f"[{args.model}] -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenAI model id (e.g. <MODEL>, gpt-4o-mini)")
    ap.add_argument("--format", choices=["mcq", "open"], default="mcq")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-image", action="store_true",
                    help="Run without sending image (text-only ablation). "
                         "Output goes to <model>__<format>_noimg.jsonl")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the todo list (useful when several processes "
                         "share the same JSONL via --resume on different shards)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
