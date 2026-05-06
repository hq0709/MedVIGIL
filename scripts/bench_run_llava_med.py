"""LLaVA-Med benchmark runner with multi-GPU sharding.

Loads LLaVA-Med-v1.5-mistral-7b on a single GPU and processes a hash-shard
of the probes. Run multiple instances in parallel (one per GPU) to use
all 6 GPUs. Output: results/baselines/llava-med__<format>.jsonl
(merged via deterministic file-append; no per-shard files needed because
each row is uniquely keyed by probe_id and we use --resume).

Usage (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python3 scripts/bench_run_llava_med.py --format mcq --shard 0/6 &
    CUDA_VISIBLE_DEVICES=1 python3 scripts/bench_run_llava_med.py --format mcq --shard 1/6 &
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
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "data/medvlm_bench_v1"
PROBES_MCQ = BENCH / "probes_mcq.csv"
PROBES_OPEN = BENCH / "probes_open.csv"
RESULTS = ROOT / "results/baselines"
MODEL_NAME = "llava-med"


def load_probes(format_):
    p = PROBES_MCQ if format_ == "mcq" else PROBES_OPEN
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(probe):
    img_rel = probe.get("image_file", "").strip()
    if not img_rel:
        # knowledge_only probes have no image; LLaVA-Med requires one.
        # Use a fixed placeholder so the model still runs; the question
        # is text-only-answerable by design so the placeholder shouldn't
        # confound the answer.
        return BENCH / "images" / "MVB-0001.jpg"
    if img_rel.startswith("images_perturbed/"):
        return BENCH / img_rel
    return BENCH / "images" / img_rel


def build_mcq_prompt(probe):
    """LLaVA-Med dislikes 'reply with one letter' instructions and produces empty
    output. The natural-question form 'Which one is correct (A-E)?' yields
    answers like 'The correct answer is A.' which parse_letter can extract."""
    parts = [probe["question"], "", "Options:"]
    for L in "ABCDE":
        c = probe.get(f"choice_{L}", "").strip()
        if c:
            parts.append(f"{L}. {c}")
    parts.append("")
    parts.append("Which one is correct (A, B, C, D, or E)?")
    return "\n".join(parts)


def build_open_prompt(probe):
    return probe["question"]


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
    ap.add_argument("--format", choices=["mcq", "open"], required=True)
    ap.add_argument("--shard", type=str, default="0/1",
                    help="i/n - this process handles probes whose hash%n==i")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    print(f"[init] format={args.format} shard={shard_i}/{shard_n} gpu={gpu}", flush=True)

    from inference import MedVLMInference
    t0 = time.time()
    infer = MedVLMInference(model_key="llava-med")
    print(f"[init] loaded in {time.time()-t0:.1f}s", flush=True)

    probes = load_probes(args.format)
    todo = [p for p in probes if in_shard(p["probe_id"], shard_i, shard_n)]
    print(f"[init] shard owns {len(todo)}/{len(probes)} probes", flush=True)

    out_path = RESULTS / f"{MODEL_NAME}__{args.format}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip probe_ids already in the shared file
    done = load_done(out_path)
    todo = [p for p in todo if p["probe_id"] not in done]
    print(f"[init] resume: {len(done)} already in file, {len(todo)} todo for this shard", flush=True)

    n_ok = n_err = 0
    n_correct = 0
    t_start = time.time()
    out_f = open(out_path, "a")
    for i, probe in enumerate(todo, 1):
        pid = probe["probe_id"]
        img = resolve_image_path(probe)
        prompt = build_mcq_prompt(probe) if args.format == "mcq" else build_open_prompt(probe)

        try:
            t0 = time.time()
            answer = infer.ask(str(img) if img and img.exists() else None,
                               prompt, max_new_tokens=args.max_new_tokens)
            dt_ms = int((time.time() - t0) * 1000)
            row = {
                "probe_id": pid,
                "case_id": probe["case_id"],
                "probe_kind": probe["probe_kind"],
                "format": args.format,
                "model_id": MODEL_NAME,
                "image_file": probe.get("image_file", ""),
                "question": probe.get("question", ""),
                "model_answer": answer,
                "latency_ms": dt_ms,
                "status": "ok",
                "shard": f"{shard_i}/{shard_n}",
            }
            if args.format == "mcq":
                row["model_letter"] = parse_letter(answer)
                row["correct_letter"] = probe.get("correct_letter", "")
                row["expected_behavior"] = probe.get("expected_behavior", "")
                if row["model_letter"] == row["correct_letter"]:
                    n_correct += 1
            else:
                row["expected_answer_or_set"] = probe.get("expected_answer_or_set", "")
                row["halluc_explanation"] = probe.get("halluc_explanation", "")
                row["expected_behavior"] = probe.get("expected_behavior", "")
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_ok += 1
        except Exception as e:
            n_err += 1
            err_row = {
                "probe_id": pid,
                "case_id": probe["case_id"],
                "probe_kind": probe["probe_kind"],
                "format": args.format,
                "model_id": MODEL_NAME,
                "model_answer": "",
                "status": f"error: {str(e)[:200]}",
                "shard": f"{shard_i}/{shard_n}",
            }
            out_f.write(json.dumps(err_row) + "\n")
            out_f.flush()

        if i % 50 == 0 or i == len(todo):
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 1e-3)
            eta = (len(todo) - i) / max(rate, 1e-3)
            extra = f"  acc={n_correct/n_ok:.1%}" if (args.format == "mcq" and n_ok) else ""
            print(f"[shard {shard_i}/{shard_n}] [{i}/{len(todo)}] "
                  f"ok={n_ok} err={n_err}  {rate*60:.0f}/min  ETA {eta/60:.1f}min{extra}",
                  flush=True)
    out_f.close()
    print(f"[shard {shard_i}/{shard_n}] done. ok={n_ok} err={n_err} "
          f"total {(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
