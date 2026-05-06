"""Visual-token ablation pilot runner.

For each pilot case, partition the ROI bounding box into 4 steps and
progressively replace the top fraction of ROI pixels with mid-grey:
  step 0: full image
  step 1: top 33% of ROI masked
  step 2: top 67% of ROI masked
  step 3: 100% of ROI masked (= the ROI-masked probe variant)

For every (model, case, step) we run 5 self-consistency samples at
temperature 0.7 and record the modal letter and modal-letter share
(confidence proxy), then aggregate per (model, step) into a tidy CSV.

Output: data/medvlm_bench_v1/visual_token_ablation.csv +
data/medvlm_bench_v1/_visual_token_raw.jsonl + the per-case step
images under images_visualtoken/.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvlm_bench_v1"
OUT_IMG_DIR = BENCH / "images_visualtoken"
OUT_IMG_DIR.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(ROOT))
from api_models import _load_dotenv  # noqa: E402
_load_dotenv()
sys.path.insert(0, str(ROOT / "scripts"))
import bench_run_baseline as runner  # noqa: E402
import bench_token_ablation as tok_runner  # noqa: E402


def make_variants(case_id: str, image_path: Path, roi_bbox_norm: list[float],
                   n_steps: int = 4) -> dict[int, Path]:
    """Generate progressive ROI-mask variants for one case.

    `roi_bbox_norm` is a length-4 list [x_min, y_min, x_max, y_max] or
    [x, y, w, h] in normalised image coordinates. We auto-detect by
    checking whether all four values are <= 1.0 (then it's the corner
    form xyxy, since width/height in normalised form would also be <=1).
    Many of our manifest entries store xyxy under the key
    `roi_bbox_norm`, so we accept either convention.
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    x0n, y0n, x1n, y1n = roi_bbox_norm
    # Normalised xyxy by convention: 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1
    if x1n <= x0n or y1n <= y0n:
        # Treat as xywh
        xn, yn, wn, hn = roi_bbox_norm
        x0n, y0n, x1n, y1n = xn, yn, xn + wn, yn + hn
    x0, y0 = int(round(x0n * W)), int(round(y0n * H))
    x1, y1 = int(round(x1n * W)), int(round(y1n * H))

    out = {}
    for k in range(n_steps):
        v = img.copy()
        if k > 0:
            frac = k / (n_steps - 1)
            mask_h = int(round(frac * (y1 - y0)))
            d = ImageDraw.Draw(v)
            d.rectangle([x0, y0, x1, y0 + mask_h], fill=(128, 128, 128))
        path = OUT_IMG_DIR / f"{case_id}_step{k}.jpg"
        v.save(path, quality=92)
        out[k] = path
    return out


def _parse_bbox(s: str) -> list[float] | None:
    if not s:
        return None
    s = s.strip()
    try:
        v = json.loads(s)
        if isinstance(v, list) and len(v) == 4:
            return [float(x) for x in v]
    except Exception:
        pass
    return None


def select_pilot_cases(n_per_tier: int = 4) -> list[dict]:
    """Pick the same 13 stratified cases the text-token pilot used,
    but require a usable ROI bbox.
    """
    manifest = list(csv.DictReader((BENCH / "manifest.csv").open()))
    grounding = {r["case_id"]: r for r in csv.DictReader((BENCH / "grounding.csv").open())}
    by_tier = collections.defaultdict(list)
    for row in manifest:
        if (row.get("source_dataset") or "").strip() == "cxr":
            continue  # skip credentialed CXR
        gr = grounding.get(row["case_id"])
        if not gr:
            continue
        bbox = _parse_bbox(gr.get("roi_bbox_norm", ""))
        if not bbox:
            continue
        cats = tok_runner.categorize_tokens(row["question"])
        kinds = set(cats.values())
        if {"laterality", "anatomy", "finding"} & kinds and len(kinds) >= 2:
            row["_bbox"] = bbox
            by_tier[row["risk_tier"]].append(row)
    picked = []
    for t in ("L1", "L2", "L3", "L4", "L5"):
        bucket = by_tier.get(t, [])
        picked.extend(bucket[:n_per_tier])
    return picked


def load_probes_lookup() -> dict:
    out = {}
    with (BENCH / "probes_mcq.csv").open() as f:
        for r in csv.DictReader(f):
            if r.get("probe_kind") == "original":
                out[r["case_id"]] = r
    return out


async def query_self_consistency(client, model, image_path, prompt, n=5):
    is_reasoning = "5.5" in model.lower() or model in {"o1", "o1-mini", "o1-preview"}
    max_tok = 4000 if is_reasoning else 96
    letters = []
    for _ in range(n):
        try:
            raw, _ = await tok_runner.call_one_with_temp(client, model, image_path, prompt,
                                                         temperature=0.7, max_tokens=max_tok)
            letters.append(runner.parse_letter(raw))
        except Exception as e:
            print(f"   call err: {e}", flush=True)
            letters.append("")
    return letters


async def main_async(args):
    cases = select_pilot_cases(n_per_tier=args.cases_per_tier)
    if args.limit:
        cases = cases[:args.limit]
    probes = load_probes_lookup()
    print(f"[info] {len(cases)} pilot cases selected (with usable ROI bbox)")

    # Generate variants
    variants_per_case = {}
    for case in cases:
        img_rel = probes.get(case["case_id"], {}).get("image_file", "").strip()
        if not img_rel or img_rel.startswith("images_perturbed/"):
            continue
        ip = BENCH / "images" / img_rel
        if not ip.exists():
            continue
        variants_per_case[case["case_id"]] = make_variants(case["case_id"], ip, case["_bbox"])
    print(f"[info] generated {sum(len(v) for v in variants_per_case.values())} variant images")

    # Spin up clients (3 models, matching the text-token pilot)
    from openai import AsyncOpenAI
    from anthropic import AsyncAnthropic
    from google import genai as gen_ai

    clients = {
        "gpt-5.5":                AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        "claude-opus-4-7":        AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "gemini-3-flash-preview": gen_ai.Client(api_key=os.environ["GOOGLE_API_KEY"]),
    }
    display = {
        "gpt-5.5":                "GPT-5.5",
        "claude-opus-4-7":        "Claude Opus 4.7",
        "gemini-3-flash-preview": "Gemini 3 Flash",
    }

    sem = asyncio.Semaphore(args.concurrency)

    async def run_case_step(model_id, dm, case, step, image_path, prompt):
        async with sem:
            letters = await query_self_consistency(clients[model_id], model_id,
                                                    image_path, prompt, n=args.samples)
            letter, conf = tok_runner.mode_and_share(letters)
            return {"model": dm, "step": step, "case_id": case["case_id"],
                    "letter": letter, "confidence": conf,
                    "all_letters": "".join(letters)}

    tasks = []
    for case in cases:
        cid = case["case_id"]
        if cid not in variants_per_case:
            continue
        probe = probes.get(cid)
        if not probe:
            continue
        prompt = runner.build_mcq_prompt(probe)
        for step, ip in variants_per_case[cid].items():
            for model_id, dm in display.items():
                tasks.append(run_case_step(model_id, dm, case, step, ip, prompt))

    print(f"[info] {len(tasks)} (model x case x step) jobs · {args.samples} samples per job")
    res = await asyncio.gather(*tasks)

    # Aggregate
    by_ms = collections.defaultdict(list)
    for r in res:
        by_ms[(r["model"], r["step"])].append(r)
    step0_letter = {}
    for (m, s), recs in by_ms.items():
        if s == 0:
            for r in recs:
                step0_letter[(m, r["case_id"])] = r["letter"]

    rows_out = []
    step_label = {0: "full", 1: "roi_33%_masked", 2: "roi_67%_masked", 3: "roi_100%_masked"}
    for (m, s), recs in sorted(by_ms.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(recs)
        mean_conf = sum(r["confidence"] for r in recs) / n
        if s == 0:
            stability = 100.0
        else:
            agree = sum(1 for r in recs
                         if r["letter"] and r["letter"] == step0_letter.get((m, r["case_id"]), ""))
            stability = 100.0 * agree / n
        rows_out.append({
            "model": m, "step": s, "step_label": step_label[s], "n_cases": n,
            "mean_confidence_pct": round(mean_conf, 1),
            "letter_stability_pct": round(stability, 1),
            "note": "",
        })

    out_path = BENCH / "visual_token_ablation.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","step","step_label","n_cases",
                                            "mean_confidence_pct","letter_stability_pct","note"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"[ok] wrote {out_path}")
    for r in rows_out:
        print(f"  {r['model']:24s} step={r['step']} ({r['step_label']:18s}) "
              f"conf={r['mean_confidence_pct']:5.1f}  stab={r['letter_stability_pct']:5.1f}  n={r['n_cases']}")

    raw_path = BENCH / "_visual_token_raw.jsonl"
    with raw_path.open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"[ok] wrote {raw_path} ({len(res)} per-job records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-per-tier", type=int, default=4)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
