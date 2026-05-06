"""Token-ablation pilot runner.

For 20 stratified cases (4 per CRT tier), identify clinically informative
tokens in the anchor question (laterality / anatomy / finding) and run
each of four representative models on four step variants:
  step 0: full question
  step 1: laterality removed
  step 2: laterality + anatomy removed
  step 3: laterality + anatomy + finding removed (skeleton)

Per (model, step) we collect 5 self-consistency samples at temperature 0.7
and record:
  - selected letter (mode of the 5 samples)
  - confidence (mode-share, in %)
  - letter-stability vs. step 0 (1 if step-k mode == step-0 mode)

Output: data/medvlm_bench_v1/token_ablation.csv (overwrites the placeholder).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvlm_bench_v1"
sys.path.insert(0, str(ROOT))

from api_models import _load_dotenv  # noqa: E402
_load_dotenv()

# Reuse the runner's API helpers
sys.path.insert(0, str(ROOT / "scripts"))
import bench_run_baseline as runner  # noqa: E402

# --- key-token lexicon ---
LATERALITY = {"right", "left", "bilateral", "unilateral"}
ANATOMY = {
    "kidney", "kidneys", "liver", "spleen", "heart", "lung", "lungs",
    "brain", "cerebellum", "cerebellar", "parietal", "frontal", "temporal",
    "occipital", "thalamus", "thalami", "abdomen", "abdominal", "chest",
    "thoracic", "lumbar", "cervical", "spine", "spinal", "rib", "ribs",
    "humerus", "femur", "tibia", "knee", "shoulder", "pelvis", "pelvic",
    "renal", "hepatic", "splenic", "cardiac", "pulmonary", "mediastinal",
    "apical", "basal", "subcortical", "cortical", "ventricle", "ventricular",
    "aorta", "aortic", "intracranial", "extracranial", "subdural",
    "subarachnoid", "epidural", "intraventricular",
}
FINDING = {
    "pneumothorax", "infarct", "infarction", "haemorrhage", "hemorrhage",
    "haematoma", "hematoma", "mass", "lesion", "tumor", "tumour", "cyst",
    "fracture", "dissection", "aneurysm", "consolidation", "effusion",
    "edema", "oedema", "pneumonia", "atelectasis", "nodule", "stenosis",
    "thrombus", "thrombosis", "calcification", "obstruction", "perforation",
    "rupture", "abscess", "metastasis", "metastases", "metastatic",
    "tuberculosis", "fibrosis", "emphysema", "appendicitis", "diverticulitis",
}


def _strip_punct(tok: str) -> str:
    return re.sub(r"[^\w]", "", tok).lower()


def categorize_tokens(question: str) -> dict:
    """Return {token_index: category} for laterality / anatomy / finding hits."""
    out = {}
    toks = question.split()
    for i, t in enumerate(toks):
        clean = _strip_punct(t)
        if clean in LATERALITY:
            out[i] = "laterality"
        elif clean in ANATOMY:
            out[i] = "anatomy"
        elif clean in FINDING:
            out[i] = "finding"
    return out


def make_variants(question: str, cats: dict) -> dict:
    """Build the 4 step-variants by progressively removing tokens.

    step 0: full
    step 1: drop laterality tokens
    step 2: also drop anatomy tokens
    step 3: also drop finding tokens
    """
    toks = question.split()
    drop_keys = [
        set(),
        {"laterality"},
        {"laterality", "anatomy"},
        {"laterality", "anatomy", "finding"},
    ]
    out = {}
    for k, drop in enumerate(drop_keys):
        keep_idx = [i for i in range(len(toks))
                     if cats.get(i) not in drop]
        out[k] = " ".join(toks[i] for i in keep_idx)
    return out


def select_pilot_cases(n_per_tier: int = 4) -> list[dict]:
    """Pick cases that have usable laterality/anatomy/finding tokens, stratified."""
    manifest = list(csv.DictReader((BENCH / "manifest.csv").open()))
    by_tier = collections.defaultdict(list)
    for row in manifest:
        if (row.get("source_dataset") or "").strip() == "cxr":
            continue  # skip credentialed CXR
        cats = categorize_tokens(row["question"])
        kinds = set(cats.values())
        if {"laterality", "anatomy", "finding"} & kinds and len(kinds) >= 2:
            row["_cats"] = cats
            by_tier[row["risk_tier"]].append(row)
    picked = []
    for t in ("L1", "L2", "L3", "L4", "L5"):
        bucket = by_tier.get(t, [])
        picked.extend(bucket[:n_per_tier])
    return picked


def load_probes_lookup() -> dict:
    """Map case_id -> the 'original' MCQ probe (for the option set)."""
    out = {}
    with (BENCH / "probes_mcq.csv").open() as f:
        for r in csv.DictReader(f):
            if r.get("probe_kind") == "original":
                out[r["case_id"]] = r
    return out


def build_variant_prompt(probe: dict, variant_q: str) -> str:
    """Same as runner.build_mcq_prompt but with overridden question text."""
    parts = [variant_q, ""]
    for L in "ABCDE":
        c = probe.get(f"choice_{L}", "").strip()
        if c:
            parts.append(f"{L}) {c}")
    parts.append("")
    parts.append("Reply with ONLY the single letter of the correct option (A, B, C, D, or E). No explanation.")
    return "\n".join(parts)


async def call_one_with_temp(client, model: str, image_path, prompt: str,
                              temperature: float, max_tokens: int):
    """Lightweight wrapper that overrides temperature where the SDK supports it.

    For OpenAI/Together (chat.completions): pass temperature=0.7.
    For Anthropic: messages.create supports temperature.
    For Google: GenerateContentConfig supports temperature.
    """
    provider = runner.detect_provider(model)
    if provider == "openai":
        msgs = []
        if image_path and image_path.exists():
            b64, media = runner.encode_image(image_path)
            msgs.append({"type": "image_url",
                         "image_url": {"url": f"data:{media};base64,{b64}"}})
        msgs.append({"type": "text", "text": prompt})
        r = await client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": msgs}],
        )
        return r.choices[0].message.content or "", {}
    if provider == "anthropic":
        if image_path and image_path.exists():
            b64, media = runner.encode_image(image_path)
            content = [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": media, "data": b64}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt
        # Newer Claude models reject `temperature` (deprecated). Try with,
        # fall back to without on 400.
        try:
            r = await client.messages.create(
                model=model, max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            if "temperature" in str(e):
                r = await client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
            else:
                raise
        return ("".join(b.text for b in r.content if hasattr(b, "text")), {})
    if provider == "google":
        from google.genai import types as genai_types
        from PIL import Image as PILImage
        contents = []
        if image_path and image_path.exists():
            contents.append(PILImage.open(image_path).convert("RGB"))
        contents.append(prompt)
        cfg = genai_types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens,
        )
        r = await client.aio.models.generate_content(
            model=model, contents=contents, config=cfg,
        )
        return (r.text or "", {})
    if provider == "together":
        msgs = []
        if image_path and image_path.exists():
            b64, media = runner.encode_image(image_path)
            msgs.append({"type": "image_url",
                         "image_url": {"url": f"data:{media};base64,{b64}"}})
        msgs.append({"type": "text", "text": prompt})
        r = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": msgs}],
        )
        return r.choices[0].message.content or "", {}
    raise ValueError(f"unsupported provider: {provider}")


async def query_self_consistency(client, model, image_path, prompt, n=5):
    """Return list of selected letters from n samples at T=0.7."""
    letters = []
    for _ in range(n):
        try:
            raw, _ = await call_one_with_temp(client, model, image_path, prompt,
                                               temperature=0.7, max_tokens=8)
            letters.append(runner.parse_letter(raw))
        except Exception as e:
            print(f"   call err: {e}", flush=True)
            letters.append("")
    return letters


def mode_and_share(letters: list[str]) -> tuple[str, float]:
    bag = collections.Counter(l for l in letters if l)
    if not bag:
        return "", 0.0
    letter, n = bag.most_common(1)[0]
    return letter, 100.0 * n / len(letters)


async def main_async(args):
    cases = select_pilot_cases(n_per_tier=args.cases_per_tier)
    if args.limit:
        cases = cases[:args.limit]
    probes = load_probes_lookup()
    print(f"[info] {len(cases)} pilot cases selected")

    # Spin up clients
    from openai import AsyncOpenAI
    from anthropic import AsyncAnthropic
    from google import genai as gen_ai

    clients = {
        "gpt-4o":                       AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        "claude-opus-4-7":              AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "gemini-3.1-flash-lite-preview":gen_ai.Client(api_key=os.environ["GOOGLE_API_KEY"]),
    }
    display = {
        "gpt-4o":                        "GPT-4o",
        "claude-opus-4-7":               "Claude Opus 4.7",
        "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    }

    results = []  # rows for CSV: (display_model, step, case_id, letter, confidence)
    sem = asyncio.Semaphore(args.concurrency)

    async def run_case_step(model_id, dm, case, step, prompt, image_path):
        async with sem:
            letters = await query_self_consistency(clients[model_id], model_id,
                                                    image_path, prompt, n=args.samples)
            letter, conf = mode_and_share(letters)
            return {"model": dm, "step": step, "case_id": case["case_id"],
                    "letter": letter, "confidence": conf,
                    "all_letters": "".join(letters)}

    tasks = []
    for case in cases:
        cats = case["_cats"]
        if not cats:
            continue
        variants = make_variants(case["question"], cats)
        probe = probes.get(case["case_id"])
        if probe is None:
            continue
        img_rel = probe.get("image_file", "").strip()
        image_path = (BENCH / "images" / img_rel) if img_rel else None
        for step, vq in variants.items():
            prompt = build_variant_prompt(probe, vq)
            for model_id, dm in display.items():
                tasks.append(run_case_step(model_id, dm, case, step, prompt, image_path))

    print(f"[info] {len(tasks)} (model x case x step) jobs · samples={args.samples} per job")
    res = await asyncio.gather(*tasks)
    results.extend(res)

    # Aggregate
    # Per (model, step): mean confidence, letter-stability vs step 0
    by_ms = collections.defaultdict(list)  # (model, step) -> [{case_id, letter, conf}]
    for r in res:
        by_ms[(r["model"], r["step"])].append(r)

    # Letter stability per (model, step) = #cases where step-k letter == step-0 letter / #cases
    step0_letter = {}  # (model, case_id) -> letter
    for (m, s), recs in by_ms.items():
        if s == 0:
            for r in recs:
                step0_letter[(m, r["case_id"])] = r["letter"]

    rows_out = []
    step_label = {0: "full", 1: "laterality_drop", 2: "anatomy_drop", 3: "skeleton"}
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
            "model": m, "step": s, "step_label": step_label[s],
            "n_cases": n,
            "mean_confidence_pct": round(mean_conf, 1),
            "letter_stability_pct": round(stability, 1),
            "note": "",
        })

    out_path = BENCH / "token_ablation.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","step","step_label","n_cases",
                                            "mean_confidence_pct","letter_stability_pct","note"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"[ok] wrote {out_path}")
    for r in rows_out:
        print(f"  {r['model']:24s} step={r['step']} ({r['step_label']:18s}) "
              f"conf={r['mean_confidence_pct']:5.1f}  stab={r['letter_stability_pct']:5.1f}  n={r['n_cases']}")

    raw_path = BENCH / "_token_ablation_raw.jsonl"
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
