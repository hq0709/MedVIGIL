"""Bootstrap 95% CIs over per-probe correctness signals for MCS components.

We resample case_ids with replacement (clustered bootstrap) so that the
per-probe correctness signals from the same case stay together. For each
resample we recompute Acc_orig, PR (T-CF accuracy), NEG, SDR, SFR_w (with
fixed harm weights), VGR, ROI-masked accuracy, and the harmonic-mean MCS.
We then report 2.5/97.5 percentiles.

Output: results/bootstrap_ci.csv with model, axis, point estimate, lo, hi.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BENCH = ROOT / "data/medvigil_v1"
BASELINES = RESULTS / "baselines"

WEIGHTS = {"L1": 1, "L2": 2, "L3": 3, "L4": 5, "L5": 8}
N_BOOT = 500
random.seed(20260505)

MODEL_FILES = {
    "Claude Opus 4.7": "claude-opus-4-7__mcq.jsonl",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite-preview__mcq.jsonl",
    "Gemini 3 Flash": "gemini-3-flash-preview__mcq.jsonl",
    "<MODEL>": "<MODEL>__mcq.jsonl",
    "GPT-5.4": "gpt-5.4__mcq.jsonl",
    "Claude Sonnet 4.6": "claude-sonnet-4-6__mcq.jsonl",
    "HuatuoGPT-V 7B": "huatuogpt-vision-7b__mcq.jsonl",
    "GPT-4o": "gpt-4o__mcq.jsonl",
    "Qwen3.5-9B": "Qwen--Qwen3.5-9B__mcq.jsonl",
}


def load_manifest():
    out = {}
    with (BENCH / "manifest.csv").open() as f:
        for r in csv.DictReader(f):
            out[r["case_id"]] = {"risk_tier": r["risk_tier"]}
    return out


def load_records(path: Path):
    """case_id -> {probe_kind -> [correct (0/1), ...]} (multiple probes per kind per case)."""
    by_case: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "ok":
                continue
            cid = r.get("case_id")
            kind = r.get("probe_kind")
            ml = r.get("model_letter")
            cl = r.get("correct_letter")
            if not (cid and kind and cl):
                continue
            ok = 1 if (ml and ml == cl) else 0
            by_case[cid][kind].append(ok)
    return by_case


def resample_cases(case_ids: list[str]) -> list[str]:
    return [random.choice(case_ids) for _ in case_ids]


def axis_metrics(case_ids: list[str], by_case: dict, manifest: dict):
    # accumulate
    n_orig = c_orig = 0
    n_tcf = c_tcf = 0
    n_neg = c_neg = 0
    n_sdr = c_sdr = 0
    # halluc trap by tier — for SFR_w
    sfr_n: dict[str, int] = {t: 0 for t in WEIGHTS}
    sfr_f: dict[str, int] = {t: 0 for t in WEIGHTS}
    # VGR & ROI masked
    n_roi_only = c_roi_only = 0
    n_roi_mask = c_roi_mask = 0
    for cid in case_ids:
        cm = by_case.get(cid)
        if not cm:
            continue
        for v in cm.get("original", []):
            n_orig += 1; c_orig += v
        for v in cm.get("tcf", []):
            n_tcf += 1; c_tcf += v
        for v in cm.get("negation", []):
            n_neg += 1; c_neg += v
        for v in cm.get("specificity_drop", []):
            n_sdr += 1; c_sdr += v
        if "halluc_trap" in cm:
            tier = manifest.get(cid, {}).get("risk_tier")
            if tier in WEIGHTS:
                for v in cm["halluc_trap"]:
                    sfr_n[tier] += 1
                    sfr_f[tier] += (1 - v)
        for v in cm.get("roi_only", []):
            n_roi_only += 1; c_roi_only += v
        for v in cm.get("roi_masked", []):
            n_roi_mask += 1; c_roi_mask += v

    def safe_div(a, b):
        return 100.0 * a / b if b else float("nan")

    acc_orig = safe_div(c_orig, n_orig)
    pr = safe_div(c_tcf, n_tcf)
    neg = safe_div(c_neg, n_neg)
    sdr = safe_div(c_sdr, n_sdr)
    roi_only = safe_div(c_roi_only, n_roi_only)
    roi_mask = safe_div(c_roi_mask, n_roi_mask)
    vgr = roi_only - roi_mask if not (roi_only != roi_only or roi_mask != roi_mask) else float("nan")

    # SFR_w
    num = 0.0; den = 0.0
    for t, w in WEIGHTS.items():
        if sfr_n[t]:
            num += w * (sfr_f[t] / sfr_n[t])
            den += w
    sfrw = 100.0 * num / den if den else float("nan")

    cap = (acc_orig + pr + neg + sdr) / 4
    safe = 100.0 - sfrw
    ground = (max(0.0, min(100.0, vgr + 50)) + roi_mask) / 2
    if cap and safe and ground:
        denom = cap*safe + cap*ground + safe*ground
        mcs = 3 * cap * safe * ground / denom if denom else float("nan")
    else:
        mcs = float("nan")
    return {
        "Acc_orig": acc_orig, "PR": pr, "NEG": neg, "SDR": sdr,
        "SFR_w": sfrw, "VGR": vgr, "ROI_masked": roi_mask,
        "Cap": cap, "Safe": safe, "Ground": ground, "MCS": mcs,
    }


def main() -> None:
    manifest = load_manifest()
    out_rows: list[dict] = []
    for label, fname in MODEL_FILES.items():
        path = BASELINES / fname
        if not path.exists():
            print(f"[skip] {label}: {path} missing")
            continue
        by_case = load_records(path)
        case_ids = list(by_case.keys())
        if not case_ids:
            continue
        point = axis_metrics(case_ids, by_case, manifest)
        boot: dict[str, list[float]] = {k: [] for k in point}
        for _ in range(N_BOOT):
            samp = resample_cases(case_ids)
            m = axis_metrics(samp, by_case, manifest)
            for k, v in m.items():
                if v == v:  # not nan
                    boot[k].append(v)
        for k in point:
            xs = sorted(boot[k])
            if not xs:
                continue
            lo = xs[int(0.025 * len(xs))]
            hi = xs[int(0.975 * len(xs)) - 1]
            out_rows.append({
                "model": label, "axis": k,
                "point": round(point[k], 2),
                "lo95": round(lo, 2),
                "hi95": round(hi, 2),
                "halfwidth": round((hi - lo) / 2, 2),
            })
        print(f"[ok] {label}: MCS={point['MCS']:.1f} [{round(sorted(boot['MCS'])[int(0.025*len(boot['MCS']))], 1)},{round(sorted(boot['MCS'])[int(0.975*len(boot['MCS']))-1], 1)}]")

    out_path = RESULTS / "bootstrap_ci.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "axis", "point", "lo95", "hi95", "halfwidth"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"[ok] {len(out_rows)} rows -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
