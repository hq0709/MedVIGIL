"""Reanalyze existing exp4_<model>_results.json files through the silent-failure
metric lens. Produces a single CSV/JSON with SFR/CFHR/SDM per (model, degradation)
suitable for dropping into the paper's Result-1/2/3 tables.

Maps exp4 image-type buckets to canonical IIST degradations:
    real         -> clean
    black        -> delta_black
    white        -> delta_white
    random_noise -> delta_noise
    natural      -> delta_swap

We use the 'leading' prompt variant (which presupposes a finding) since that
isolates whether the model resists language-induced silent failure, and the
'neutral' variant as a sanity-check baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.silent_failure.metrics import (  # noqa: E402
    ScoredRecord,
    compute_cfhr,
    compute_sdm,
    compute_sfr,
    severity_histogram,
)
from analysis.silent_failure.severity import SeverityScorer  # noqa: E402
from analysis.silent_failure.refusal import RefusalJudge  # noqa: E402


IMAGE_TYPE_TO_DEGRADATION = {
    "real": "clean",
    "black": "delta_black",
    "white": "delta_white",
    "random_noise": "delta_noise",
    "natural": "delta_swap",
}

MODELS = ["gpt4o", "claude", "gemini", "llava-med", "llava-general"]


def records_from_exp4(exp4_path: Path, prompt_variant: str = "leading") -> List[ScoredRecord]:
    with open(exp4_path, "r") as f:
        d = json.load(f)
    results = d.get("results", {})
    scorer = SeverityScorer()
    judge = RefusalJudge(scorer=scorer)
    out: List[ScoredRecord] = []

    leading_qids = {"lead_tumor", "lead_cancer", "lead_fracture"}

    def emit(rec: dict, deg: str, image_type: str, idx: int):
        ans = rec.get("answer", "")
        q = rec.get("question", "")
        sev = scorer.score(ans)
        decision = judge.judge(q, ans)
        case_id = rec.get("image_name") or f"{image_type}_{idx}"
        out.append(
            ScoredRecord(
                case_id=str(case_id),
                question=q,
                answer=ans,
                degradation=deg,
                seed=0,
                text_only_answerable=False,
                gold_severity=None,
                pred_severity=sev.sigma,
                is_refusal=(decision.label == "REFUSAL"),
                confidence=None,
            )
        )

    for key in [f"{prompt_variant}_on_real", f"{prompt_variant}_on_black"]:
        if key not in results:
            continue
        image_type = key.split("_on_")[1]
        deg = IMAGE_TYPE_TO_DEGRADATION.get(image_type, image_type)
        for i, r in enumerate(results[key]):
            emit(r, deg, image_type, i)

    for i, r in enumerate(results.get("control_details", [])):
        image_type = r.get("image_type", "unknown")
        if image_type == "black":
            continue
        qid = r.get("question_id", "")
        is_leading = qid in leading_qids
        if (prompt_variant == "leading") != is_leading:
            continue
        deg = IMAGE_TYPE_TO_DEGRADATION.get(image_type, image_type)
        emit(r, deg, image_type, i)

    return out


def model_summary(scored: List[ScoredRecord]) -> Dict:
    summary: Dict = {}
    degs = sorted({s.degradation for s in scored})
    for d in degs:
        summary[d] = {
            "n": sum(1 for s in scored if s.degradation == d),
            "sfr": compute_sfr(scored, d),
            "cfhr": compute_cfhr(scored, d),
            "sdm": compute_sdm(scored, d) if d != "clean" else 0.0,
            "severity_hist": severity_histogram(scored, d),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results"),
        help="directory containing exp4_<model>_results.json files",
    )
    parser.add_argument("--prompt-variant", default="leading", choices=["leading", "neutral"])
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "silent_failure_v0.json"),
        help="output JSON path",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_summary: Dict[str, Dict] = {}
    for model in MODELS:
        path = results_dir / f"exp4_{model}_results.json"
        if not path.exists():
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        scored = records_from_exp4(path, prompt_variant=args.prompt_variant)
        all_summary[model] = model_summary(scored)
        s = all_summary[model]
        print(f"\n=== {model} ({args.prompt_variant} prompt) ===")
        for deg in ["clean", "delta_black", "delta_white", "delta_noise", "delta_swap"]:
            if deg not in s:
                continue
            row = s[deg]
            sfr = row["sfr"]
            cfhr = row["cfhr"]
            sdm = row["sdm"]
            print(
                f"  {deg:14s} n={row['n']:3d}  "
                f"SFR={sfr:.2%}  CFHR={cfhr:.2%}  SDM={sdm:+.3f}  "
                f"hist={row['severity_hist']}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"prompt_variant": args.prompt_variant, "summary": all_summary},
            f,
            indent=2,
        )
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
