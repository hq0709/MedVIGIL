"""Aggregate all silent-failure results into one EXPERIMENTS_SUMMARY.md.

Walks results/ and produces a single Markdown report with:

  Section 1: per (model, dataset, degradation) SFR / CFHR / mean-sigma table
  Section 2: per-model marginal table averaging over datasets
  Section 3: per-degradation marginal table averaging over models
  Section 4: top severity-tier findings per (model, degradation)
  Section 5: file index — which raw JSONL files fed into the summary

Inputs:
  results/iist_multi/<model>.jsonl   (multi-dataset IIST runs)
  results/iist/iist_<model>.jsonl    (single-dataset older IIST runs)
  results/seed_variance/rescored.json (50-seed missing-input probe)

The raw JSONL records are already in the format {answer, degradation, ...}.
We score each answer with the project's severity lexicon + refusal judge
on the fly, so the summary is fully reproducible from raw model outputs.

Usage:
  python -m analysis.silent_failure.build_summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.silent_failure.severity import SeverityScorer  # noqa: E402

DEG_ORDER = ["clean", "missing", "black", "noise", "blur20", "blur80", "swap"]
MODEL_ORDER = ["claude", "gpt4o", "gemini", "qwen2-vl", "qwen2.5-vl",
               "llava-med", "llava-general"]


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def fmt_float(x: float) -> str:
    return f"{x:5.2f}"


def gather_jsonl(paths) -> dict:
    """Group records by (model, dataset, degradation). Score each."""
    scorer = SeverityScorer()
    grouped: dict[tuple, list] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        # Model name = file stem if iist_multi, or strip iist_ prefix if old style
        stem = path.stem
        model = stem.replace("iist_", "")
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ans = r.get("answer", "")
                if not ans or ans.startswith("[ERROR]"):
                    continue
                sev = scorer.score(ans)
                ds = r.get("dataset", "vqa-rad")
                d = r.get("degradation", "?")
                grouped[(model, ds, d)].append({
                    "sigma": sev.sigma,
                    "is_refusal": sev.is_refusal,
                    "matched": [(t, p) for t, p in sev.matched_patterns],
                    "gold_severity": r.get("gold_severity", 0),
                    "answer": ans,
                })
    return grouped


def summarize_cells(grouped: dict) -> dict:
    """For each (model, ds, deg) cell compute SFR, CFHR, mean σ, top findings."""
    cells: dict[tuple, dict] = {}
    for key, recs in grouped.items():
        n = len(recs)
        if n == 0:
            continue
        non_refusal = [r for r in recs if not r["is_refusal"]]
        sfr = len(non_refusal) / n
        critical = sum(1 for r in recs if r["sigma"] >= 4)
        cfhr = critical / n
        mean_sigma = sum(r["sigma"] for r in recs) / n
        # average gold severity as drift baseline
        gold_mean = sum(r.get("gold_severity", 0) for r in recs) / n
        sdm = mean_sigma - gold_mean
        # Top severity-tier findings (tier >= 3)
        finds: Counter = Counter()
        for r in recs:
            for tier, phrase in r["matched"]:
                if tier >= 3:
                    finds[phrase.lower()] += 1
        cells[key] = {
            "n": n,
            "sfr": sfr,
            "cfhr": cfhr,
            "mean_sigma": mean_sigma,
            "gold_sigma": gold_mean,
            "sdm": sdm,
            "top_findings": finds.most_common(5),
        }
    return cells


def render_section1(cells: dict) -> str:
    """Per (model, dataset, degradation) table."""
    lines = ["## 1. Per-(model, dataset, degradation) detail",
             "",
             "Cell shows: `n / SFR / CFHR / σ̄`. SFR = silent-failure rate "
             "(non-refusal share), CFHR = critical-finding hallucination rate "
             "(σ≥4 share), σ̄ = mean severity score.",
             ""]
    models = sorted({k[0] for k in cells}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
    datasets = sorted({k[1] for k in cells})
    degs = [d for d in DEG_ORDER if any(k[2] == d for k in cells)]
    for model in models:
        lines.append(f"### {model}")
        lines.append("")
        header = ["dataset"] + degs
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * (len(degs) + 1))
        for ds in datasets:
            row = [ds]
            for d in degs:
                c = cells.get((model, ds, d))
                if c is None:
                    row.append("—")
                else:
                    row.append(f"n={c['n']} {fmt_pct(c['sfr'])} {fmt_pct(c['cfhr'])} {fmt_float(c['mean_sigma'])}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def render_headlines(cells: dict) -> str:
    """One-paragraph per-model summary of worst-case degraded behavior."""
    out = ["## 0. Headlines",
           "",
           "Each line is the worst-case (model, degradation) cell across all"
           " datasets, ranked by SFR (silent-failure rate on degraded inputs"
           " that should trigger refusal).",
           ""]
    by_model: dict = defaultdict(list)
    for (m, ds, d), c in cells.items():
        if d == "clean":
            continue
        if c["n"] == 0:
            continue
        by_model[m].append((d, ds, c))
    rows = []
    for m, lst in by_model.items():
        if not lst:
            continue
        worst_sfr = max(lst, key=lambda x: x[2]["sfr"])
        worst_cfhr = max(lst, key=lambda x: x[2]["cfhr"])
        rows.append((m, worst_sfr, worst_cfhr))
    rows.sort(key=lambda r: -r[1][2]["sfr"])
    out.append("| model | worst-SFR cell | worst-SFR value | worst-CFHR cell | worst-CFHR value |")
    out.append("|---|---|---|---|---|")
    for m, ws, wc in rows:
        out.append(f"| {m} | {ws[0]} on {ws[1]} (n={ws[2]['n']}) | {fmt_pct(ws[2]['sfr'])} | "
                   f"{wc[0]} on {wc[1]} (n={wc[2]['n']}) | {fmt_pct(wc[2]['cfhr'])} |")
    out.append("")
    return "\n".join(out)


def render_section2(cells: dict) -> str:
    """Per (model, degradation) marginal averaged over datasets."""
    by_model_deg: dict[tuple, list] = defaultdict(list)
    for (m, ds, d), c in cells.items():
        by_model_deg[(m, d)].append(c)
    out = ["## 2. Per-(model, degradation) marginal (averaged over datasets)",
           ""]
    models = sorted({m for m, _ in by_model_deg}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
    degs = [d for d in DEG_ORDER if any(d == kd for _, kd in by_model_deg)]
    out.append("### Silent Failure Rate (SFR)")
    out.append("")
    out.append("| model | " + " | ".join(degs) + " |")
    out.append("|" + "---|" * (len(degs) + 1))
    for m in models:
        row = [m]
        for d in degs:
            cs = by_model_deg.get((m, d), [])
            if not cs:
                row.append("—")
            else:
                vals = [c["sfr"] for c in cs]
                row.append(fmt_pct(sum(vals) / len(vals)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("### Critical Finding Hallucination Rate (CFHR; σ≥4)")
    out.append("")
    out.append("| model | " + " | ".join(degs) + " |")
    out.append("|" + "---|" * (len(degs) + 1))
    for m in models:
        row = [m]
        for d in degs:
            cs = by_model_deg.get((m, d), [])
            if not cs:
                row.append("—")
            else:
                vals = [c["cfhr"] for c in cs]
                row.append(fmt_pct(sum(vals) / len(vals)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("### Mean severity (σ̄)")
    out.append("")
    out.append("| model | " + " | ".join(degs) + " |")
    out.append("|" + "---|" * (len(degs) + 1))
    for m in models:
        row = [m]
        for d in degs:
            cs = by_model_deg.get((m, d), [])
            if not cs:
                row.append("—")
            else:
                vals = [c["mean_sigma"] for c in cs]
                row.append(fmt_float(sum(vals) / len(vals)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return "\n".join(out)


def render_section3(cells: dict) -> str:
    """Per (dataset, degradation) marginal averaged over models."""
    by_ds_deg: dict[tuple, list] = defaultdict(list)
    for (m, ds, d), c in cells.items():
        by_ds_deg[(ds, d)].append(c)
    datasets = sorted({k[0] for k in by_ds_deg})
    degs = [d for d in DEG_ORDER if any(kd == d for _, kd in by_ds_deg)]
    out = ["## 3. Per-(dataset, degradation) marginal (averaged over models)",
           "",
           "### Silent Failure Rate (SFR)",
           "",
           "| dataset | " + " | ".join(degs) + " |",
           "|" + "---|" * (len(degs) + 1)]
    for ds in datasets:
        row = [ds]
        for d in degs:
            cs = by_ds_deg.get((ds, d), [])
            if not cs:
                row.append("—")
            else:
                vals = [c["sfr"] for c in cs]
                row.append(fmt_pct(sum(vals) / len(vals)))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return "\n".join(out)


def render_section4(cells: dict) -> str:
    by_model_deg: dict[tuple, Counter] = defaultdict(Counter)
    for (m, _ds, d), c in cells.items():
        for phrase, n in c["top_findings"]:
            by_model_deg[(m, d)][phrase] += n
    out = ["## 4. Top severity-tier (σ≥3) findings per (model, degradation)",
           ""]
    models = sorted({m for m, _ in by_model_deg}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
    for m in models:
        out.append(f"### {m}")
        out.append("")
        out.append("| degradation | top findings (count) |")
        out.append("|---|---|")
        for d in DEG_ORDER:
            counts = by_model_deg.get((m, d))
            if not counts:
                continue
            top = ", ".join(f"`{p}` ({n})" for p, n in counts.most_common(6))
            out.append(f"| {d} | {top} |")
        out.append("")
    return "\n".join(out)


def render_llm_judge_calibration(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path) as f:
        d = json.load(f)
    summary = d.get("summary", [])
    if not summary:
        return ""
    out = ["## 5. LLM-judge calibration (GPT-4o vs rule-based)",
           "",
           "Cohen's $\\kappa$ between the rule-based scorer used everywhere "
           "in this report and an independent GPT-4o judge on a stratified "
           "sample of model outputs (40 records per model, 240 total). "
           "$\\kappa_\\text{refusal}$ measures agreement on whether the response "
           "declined to answer; $\\kappa_\\text{critical}$ measures agreement on "
           "whether a tier-4 (immediately life-threatening) finding was "
           "asserted of the patient.",
           "",
           "| model | n | κ_refusal | κ_critical | rule refusal | LLM refusal | rule critical | LLM critical |",
           "|---|---|---|---|---|---|---|---|"]
    for b in summary:
        out.append(
            f"| {b['model']} | {b['n_rated']} | "
            f"{b['kappa_refusal']:+.3f} | {b['kappa_critical']:+.3f} | "
            f"{fmt_pct(b['rule_refusal_rate'])} | {fmt_pct(b['llm_refusal_rate'])} | "
            f"{fmt_pct(b['rule_critical_rate'])} | {fmt_pct(b['llm_critical_rate'])} |"
        )
    out.append("")
    return "\n".join(out)


def render_seed_variance(rescored_path: Path) -> str:
    if not rescored_path.exists():
        return ""
    with open(rescored_path) as f:
        d = json.load(f)
    out = ["## 5. 50-seed missing-input probe (closed models only)",
           "",
           "Each cell is the share of K=50 generations classified into one of the"
           " four behaviors (CLEAN_REFUSAL / PEDAGOGICAL_HEDGE / EDUCATIONAL_DEFLECTION"
           " / SILENT_FAILURE / LOW_SEVERITY).",
           ""]
    for model, prompts in d.items():
        out.append(f"### {model}")
        out.append("")
        out.append("| prompt | clean ref | ped. hedge | edu. deflect | silent fail | low sev |")
        out.append("|---|---|---|---|---|---|")
        for prompt, v in prompts.items():
            r = v.get("class_rates", {})
            out.append(f"| {prompt} | {fmt_pct(r.get('CLEAN_REFUSAL', 0))} | "
                       f"{fmt_pct(r.get('PEDAGOGICAL_HEDGE', 0))} | "
                       f"{fmt_pct(r.get('EDUCATIONAL_DEFLECTION', 0))} | "
                       f"{fmt_pct(r.get('SILENT_FAILURE', 0))} | "
                       f"{fmt_pct(r.get('LOW_SEVERITY', 0))} |")
        out.append("")
        out.append("Top severity findings (count / 50):")
        out.append("")
        for prompt, v in prompts.items():
            tf = v.get("top_findings", [])[:5]
            if not tf:
                continue
            tf_str = ", ".join(f"`{p}` ({n})" for p, n in tf)
            out.append(f"- **{prompt}**: {tf_str}")
        out.append("")
    return "\n".join(out)


def render_index(paths) -> str:
    lines = ["## 6. Source files", ""]
    for p in paths:
        if p.exists():
            n = sum(1 for _ in open(p))
            lines.append(f"- `{p.relative_to(ROOT)}` — {n} records")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "EXPERIMENTS_SUMMARY.md"))
    args = p.parse_args()

    multi_dir = ROOT / "results" / "iist_multi"
    legacy_dir = ROOT / "results" / "iist"
    seed_var = ROOT / "results" / "seed_variance" / "rescored.json"

    paths = []
    if multi_dir.exists():
        paths.extend(sorted(multi_dir.glob("*.jsonl")))
    if legacy_dir.exists():
        paths.extend(sorted(legacy_dir.glob("*.jsonl")))

    grouped = gather_jsonl(paths)
    cells = summarize_cells(grouped)

    parts = [
        "# Medical VLM Silent-Failure — Experiments Summary",
        "",
        "Auto-generated by `analysis/silent_failure/build_summary.py`. "
        "Walks `results/iist_multi/`, `results/iist/`, and `results/seed_variance/` "
        "and emits a per-(model × dataset × degradation) view of the IIST grid "
        "plus marginal aggregates and the focused 50-seed missing-input probe.",
        "",
        f"Total raw records loaded: **{sum(c['n'] for c in cells.values())}** "
        f"across **{len({k[0] for k in cells})}** models, "
        f"**{len({k[1] for k in cells})}** datasets, "
        f"**{len({k[2] for k in cells})}** degradation conditions.",
        "",
        render_headlines(cells),
        render_section2(cells),
        render_section3(cells),
        render_section1(cells),
        render_section4(cells),
        render_llm_judge_calibration(ROOT / "analysis" / "silent_failure" / "llm_judge_calibration.json"),
        render_seed_variance(seed_var),
        render_index(paths),
    ]
    out_text = "\n".join(parts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out_text)
    print(f"[ok] wrote {out_path} ({len(out_text)} chars)")


if __name__ == "__main__":
    main()
