"""Visual-information-decay ablation: HuatuoGPT-Vision-7B local runner.

Loads HuatuoGPT-Vision once and iterates over all sigma levels, processing
the 300 original probes for each. Use one process per GPU and shard across
probe IDs to parallelise.

For sigma=inf (no-image) the runner passes an empty image list to the bot,
matching the "blind LM" semantics used in bench_run_baseline.py's
--no-image path.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/bench_blur_sweep_run_huatuo.py \
        --sigmas 0,2,4,8,16,32,64,inf --shard 0/4 &
    CUDA_VISIBLE_DEVICES=1 python3 scripts/bench_blur_sweep_run_huatuo.py \
        --sigmas 0,2,4,8,16,32,64,inf --shard 1/4 &
    ... etc
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HuatuoGPT-Vision"))

BENCH = ROOT / "data/medvlm_bench_v1"
SWEEP_ROOT = BENCH / "_blur_sweep"
RESULTS = ROOT / "results/blur_sweep"
MODEL_NAME = "huatuogpt-vision-7b"
MODEL_DIR = ROOT / "checkpoints/huatuogpt-vision-7b"


def load_sigma_probes(sigma: str) -> list[dict]:
    if sigma == "inf":
        path = SWEEP_ROOT / "probes_sigma0.csv"
    else:
        path = SWEEP_ROOT / f"probes_sigma{sigma}.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(probe: dict) -> Path | None:
    img_rel = probe.get("image_file", "").strip()
    if not img_rel:
        return None
    return BENCH / img_rel


def build_mcq_prompt(probe):
    parts = [probe["question"], "", "Options:"]
    for L in "ABCDE":
        c = probe.get(f"choice_{L}", "").strip()
        if c:
            parts.append(f"{L}. {c}")
    parts.append("")
    parts.append("Which one is correct (A, B, C, D, or E)?")
    return "\n".join(parts)


def parse_letter(text):
    if not text:
        return ""
    s = text.strip().upper()
    m = re.match(r"^\s*[\(\[\*]*\s*([ABCDE])\b", s)
    if m: return m.group(1)
    m = re.search(r"\b([ABCDE])[\.\)]", s)
    if m: return m.group(1)
    m = re.search(r"\b([ABCDE])\b", s)
    if m: return m.group(1)
    return ""


def in_shard(probe_id, shard_i, shard_n):
    return int(hashlib.md5(probe_id.encode()).hexdigest(), 16) % shard_n == shard_i


def load_done(out_path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0,2,4,8,16,32,64,inf",
                    help="Comma-separated sigma list to sweep")
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    sigmas = [s.strip() for s in args.sigmas.split(",") if s.strip()]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    print(f"[init] sigmas={sigmas} shard={shard_i}/{shard_n} gpu={gpu}", flush=True)

    from cli import HuatuoChatbot
    t0 = time.time()
    bot = HuatuoChatbot(str(MODEL_DIR), device="cuda")
    bot.gen_kwargs.update({
        "do_sample": False,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
    })
    bot.gen_kwargs.pop("min_new_tokens", None)
    bot.debug = False
    print(f"[init] huatuo loaded in {time.time()-t0:.1f}s", flush=True)

    for sigma in sigmas:
        no_image = (sigma == "inf")
        probes_all = load_sigma_probes(sigma)
        todo = [p for p in probes_all if in_shard(p["probe_id"], shard_i, shard_n)]
        out_path = RESULTS / f"{MODEL_NAME}__sigma{sigma}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = load_done(out_path)
        todo = [p for p in todo if p["probe_id"] not in done]

        print(f"\n[sigma={sigma:>3}] shard {shard_i}/{shard_n} owns {len(todo)} new probes (no_image={no_image})", flush=True)

        out_f = open(out_path, "a")
        n_ok = n_err = n_correct = 0
        t_start = time.time()
        for i, probe in enumerate(todo, 1):
            pid = probe["probe_id"]
            img = None if no_image else resolve_image_path(probe)
            prompt = build_mcq_prompt(probe)
            try:
                t1 = time.time()
                imgs_arg = [str(img)] if img and img.exists() else []
                answers = bot.inference(prompt, imgs_arg)
                answer = answers[0] if isinstance(answers, list) and answers else ""
                if "<|assistant|>" in answer:
                    answer = answer.split("<|assistant|>")[-1].strip()
                row = {
                    "probe_id": pid,
                    "case_id": probe["case_id"],
                    "model_id": MODEL_NAME,
                    "image_file": probe.get("image_file", ""),
                    "model_answer": answer,
                    "model_letter": parse_letter(answer),
                    "correct_letter": probe.get("correct_letter", ""),
                    "latency_ms": int((time.time() - t1) * 1000),
                    "status": "ok",
                    "shard": f"{shard_i}/{shard_n}",
                }
                if row["model_letter"] == row["correct_letter"]:
                    n_correct += 1
                n_ok += 1
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
            except Exception as e:
                n_err += 1
                err_row = {
                    "probe_id": pid,
                    "case_id": probe.get("case_id", ""),
                    "model_id": MODEL_NAME,
                    "model_answer": "",
                    "status": f"error: {str(e)[:200]}",
                    "shard": f"{shard_i}/{shard_n}",
                }
                out_f.write(json.dumps(err_row) + "\n")
                out_f.flush()

            if i % 30 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-3)
                eta = (len(todo) - i) / max(rate, 1e-3)
                extra = f"  acc={n_correct/max(n_ok,1):.1%}"
                print(f"[sigma={sigma} shard {shard_i}/{shard_n}] [{i}/{len(todo)}] ok={n_ok} err={n_err} {rate*60:.0f}/min ETA {eta/60:.1f}min{extra}", flush=True)
        out_f.close()
        print(f"[sigma={sigma}] shard done. ok={n_ok} err={n_err} acc={n_correct/max(n_ok,1):.2%} ({(time.time()-t_start)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
