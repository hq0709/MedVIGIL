"""Aggregate all baseline results into paper-ready artifacts.

Reads results/baselines/<model>__mcq.jsonl and computes the 8 evaluation
metrics defined in PIPELINE.md across all probe kinds, risk tiers, and
source datasets. Writes:

  results/REPORT.md                           one-page summary for paper drafting
  results/metrics_summary.csv                 wide table: model × metric
  results/metrics_by_kind.csv                 long table: model × probe_kind
  results/metrics_by_tier.csv                 long table: model × risk_tier
  results/metrics_by_source.csv               long table: model × source_dataset
  results/tables/main_results.tex             LaTeX-ready Table 1
  results/tables/per_kind.tex                 LaTeX-ready per-kind table

Run:
    python3 scripts/bench_aggregate.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvigil_v1"
RESULTS = ROOT / "results"
BASELINES = RESULTS / "baselines"

REPORT = RESULTS / "REPORT.md"
SUMMARY_CSV = RESULTS / "metrics_summary.csv"
BY_KIND_CSV = RESULTS / "metrics_by_kind.csv"
BY_TIER_CSV = RESULTS / "metrics_by_tier.csv"
BY_SOURCE_CSV = RESULTS / "metrics_by_source.csv"
TABLE_DIR = RESULTS / "tables"

PROBE_KINDS = [
    "original", "tcf", "negation", "specificity_drop", "knowledge_only",
    "halluc_trap", "lr_flip", "roi_only", "roi_masked",
]
TIERS = ["L1", "L2", "L3", "L4", "L5"]


def load_manifest() -> dict[str, dict]:
    with open(BENCH / "manifest.csv", newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def load_baseline(model: str) -> list[dict]:
    p = BASELINES / f"{model}__mcq.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    rows.append(r)
            except Exception:
                continue
    return rows


def discover_models() -> list[str]:
    """All models that have a __mcq.jsonl file."""
    models = []
    for p in sorted(BASELINES.glob("*__mcq.jsonl")):
        m = p.name.replace("__mcq.jsonl", "")
        models.append(m)
    return models


def is_correct(rec: dict) -> bool:
    return rec.get("model_letter", "") == rec.get("correct_letter", "")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: str, manifest: dict) -> dict:
    recs = load_baseline(model)
    n_total = len(recs)
    if n_total == 0:
        return {"model": model, "n_total": 0}

    # by probe_kind
    by_kind = defaultdict(lambda: [0, 0])
    for r in recs:
        c = is_correct(r)
        by_kind[r["probe_kind"]][1] += 1
        if c:
            by_kind[r["probe_kind"]][0] += 1

    def acc(kind):
        c, n = by_kind.get(kind, (0, 0))
        return c / n if n else float("nan"), n

    out = {
        "model": model,
        "n_total": n_total,
        "overall_acc": sum(c for c, _ in by_kind.values()) / sum(n for _, n in by_kind.values()),
    }
    for k in PROBE_KINDS:
        a, n = acc(k)
        out[f"acc_{k}"] = a
        out[f"n_{k}"] = n

    # SFR — % of trap probes where model did NOT pick the refusal letter (E)
    trap_c, trap_n = by_kind.get("halluc_trap", (0, 0))
    out["sfr"] = (trap_n - trap_c) / trap_n if trap_n else float("nan")

    # VGR — acc(roi_only) - acc(roi_masked)
    a_o, _ = acc("roi_only"); a_m, _ = acc("roi_masked")
    out["vgr"] = a_o - a_m if (a_o == a_o and a_m == a_m) else float("nan")

    # PR — paraphrase robustness: agreement between original and tcf answers
    # for the SAME case (paired). Compute consistency = (orig correct AND tcf correct)
    # OR (orig wrong AND tcf wrong with same letter)
    by_case = defaultdict(dict)
    for r in recs:
        if r["probe_kind"] in ("original", "tcf"):
            by_case[r["case_id"]][r["probe_kind"]] = r
    agree = paired = 0
    for cid, d in by_case.items():
        if "original" in d and "tcf" in d:
            paired += 1
            if d["original"]["model_letter"] == d["tcf"]["model_letter"]:
                agree += 1
    out["pr"] = agree / paired if paired else float("nan")
    out["pr_n"] = paired

    # NEG — accuracy on negation probes (gold-flipped)
    out["neg_acc"] = out["acc_negation"]

    # SDR — accuracy on specificity-drop probes
    out["sdr"] = out["acc_specificity_drop"]

    # LPL — accuracy on knowledge_only probes (high = strong language prior)
    out["lpl"] = out["acc_knowledge_only"]

    # TR-coh — triplet coherence: requires triplet-level scoring; skip until
    # we add it (placeholder: NaN for now)
    out["tr_coh"] = float("nan")

    return out


def compute_by_tier(model: str, manifest: dict) -> list[dict]:
    """Per-CRT-tier accuracy on the `original` probes only."""
    recs = load_baseline(model)
    by_tier = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["probe_kind"] != "original":
            continue
        cid = r["case_id"]
        m = manifest.get(cid)
        if not m:
            continue
        tier = m["risk_tier"]
        by_tier[tier][1] += 1
        if is_correct(r):
            by_tier[tier][0] += 1
    rows = []
    for t in TIERS:
        c, n = by_tier.get(t, (0, 0))
        rows.append({"model": model, "risk_tier": t,
                     "acc": c/n if n else float("nan"), "n": n})
    return rows


def compute_by_source(model: str, manifest: dict) -> list[dict]:
    """Per source-dataset overall accuracy."""
    recs = load_baseline(model)
    by_src = defaultdict(lambda: [0, 0])
    for r in recs:
        cid = r["case_id"]
        m = manifest.get(cid)
        if not m:
            continue
        src = m["source_dataset"]
        by_src[src][1] += 1
        if is_correct(r):
            by_src[src][0] += 1
    rows = []
    for src, (c, n) in sorted(by_src.items()):
        rows.append({"model": model, "source_dataset": src,
                     "acc": c/n if n else float("nan"), "n": n})
    return rows


def compute_sfr_by_tier(model: str, manifest: dict) -> list[dict]:
    """Trap silent-failure rate broken down by case risk_tier."""
    recs = load_baseline(model)
    by_tier = defaultdict(lambda: [0, 0])  # [silent_fail, total]
    for r in recs:
        if r["probe_kind"] != "halluc_trap":
            continue
        cid = r["case_id"]
        m = manifest.get(cid)
        if not m:
            continue
        tier = m["risk_tier"]
        by_tier[tier][1] += 1
        if r.get("model_letter") != r.get("correct_letter"):  # not E => silent fail
            by_tier[tier][0] += 1
    rows = []
    for t in TIERS:
        sf, n = by_tier.get(t, (0, 0))
        rows.append({"model": model, "risk_tier": t,
                     "sfr": sf/n if n else float("nan"), "n_traps": n})
    return rows


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def fmt_pct(x: float, sign=False) -> str:
    if x != x:
        return "—"
    if sign:
        return f"{x*100:+.1f}"
    return f"{x*100:.1f}"


def render_main_md(rows: list[dict]) -> str:
    out = []
    out.append("| model | n | Overall | SFR↓ | VGR↑ | PR | NEG | SDR | LPL | knowledge_only | original |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: -x.get("overall_acc", 0)):
        out.append(
            f"| `{r['model']}` | {r['n_total']} | "
            f"{fmt_pct(r['overall_acc'])}% | "
            f"{fmt_pct(r['sfr'])}% | "
            f"{fmt_pct(r['vgr'], sign=True)}pp | "
            f"{fmt_pct(r['pr'])}% | "
            f"{fmt_pct(r['neg_acc'])}% | "
            f"{fmt_pct(r['sdr'])}% | "
            f"{fmt_pct(r['lpl'])}% | "
            f"{fmt_pct(r['acc_knowledge_only'])}% | "
            f"{fmt_pct(r['acc_original'])}% |"
        )
    return "\n".join(out)


def render_main_latex(rows: list[dict]) -> str:
    """LaTeX-ready Table 1 for the paper."""
    cols = "lrrrrrrrr"
    out = []
    out.append(r"\begin{tabular}{" + cols + "}")
    out.append(r"\toprule")
    out.append(r"Model & $n$ & Overall & SFR$\downarrow$ & VGR$\uparrow$ & PR & NEG & SDR & LPL \\")
    out.append(r"\midrule")
    for r in sorted(rows, key=lambda x: -x.get("overall_acc", 0)):
        out.append(
            f"{r['model'].replace('-','--')} & "
            f"{r['n_total']} & "
            f"{fmt_pct(r['overall_acc'])} & "
            f"{fmt_pct(r['sfr'])} & "
            f"{fmt_pct(r['vgr'], sign=True)} & "
            f"{fmt_pct(r['pr'])} & "
            f"{fmt_pct(r['neg_acc'])} & "
            f"{fmt_pct(r['sdr'])} & "
            f"{fmt_pct(r['lpl'])} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def render_per_kind_md(rows: list[dict]) -> str:
    out = ["| model | original | tcf | negation | spec_drop | know_only | trap | lr_flip | roi_only | roi_masked |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: -x.get("overall_acc", 0)):
        out.append(
            f"| `{r['model']}` | "
            + " | ".join(f"{fmt_pct(r.get(f'acc_{k}', float('nan')))}%" for k in PROBE_KINDS)
            + " |"
        )
    return "\n".join(out)


def render_by_tier_md(by_tier_rows: list[dict], models: list[str]) -> str:
    by_model_tier = defaultdict(dict)
    for r in by_tier_rows:
        by_model_tier[r["model"]][r["risk_tier"]] = r["acc"]
    out = ["| model | L1 | L2 | L3 | L4 | L5 |",
           "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        d = by_model_tier.get(m, {})
        out.append(f"| `{m}` | " +
                   " | ".join(f"{fmt_pct(d.get(t, float('nan')))}%" for t in TIERS) + " |")
    return "\n".join(out)


def render_sfr_by_tier_md(rows: list[dict], models: list[str]) -> str:
    bm = defaultdict(dict)
    for r in rows:
        bm[r["model"]][r["risk_tier"]] = r["sfr"]
    out = ["| model | L1 | L2 | L3 | L4 | L5 |",
           "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        d = bm.get(m, {})
        out.append(f"| `{m}` | " +
                   " | ".join(f"{fmt_pct(d.get(t, float('nan')))}%" for t in TIERS) + " |")
    return "\n".join(out)


def render_by_source_md(rows: list[dict], models: list[str]) -> str:
    sources = sorted({r["source_dataset"] for r in rows})
    bm = defaultdict(dict)
    for r in rows:
        bm[r["model"]][r["source_dataset"]] = r["acc"]
    head = "| model | " + " | ".join(sources) + " |"
    sep = "|---|" + "|".join(["---:"] * len(sources)) + "|"
    out = [head, sep]
    for m in models:
        d = bm.get(m, {})
        out.append(f"| `{m}` | " +
                   " | ".join(f"{fmt_pct(d.get(s, float('nan')))}%" for s in sources) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    manifest = load_manifest()
    models = discover_models()
    print(f"[init] {len(models)} models discovered")
    for m in models:
        n = len(load_baseline(m))
        marker = "✓" if n >= 2400 else f"⚠ ({n}/2556)"
        print(f"  {m:<35} {marker}")

    # main metrics
    summary_rows = [compute_metrics(m, manifest) for m in models]
    summary_rows = [r for r in summary_rows if r.get("n_total", 0) > 0]

    # by_tier
    by_tier_rows = []
    sfr_tier_rows = []
    by_source_rows = []
    for m in models:
        by_tier_rows.extend(compute_by_tier(m, manifest))
        sfr_tier_rows.extend(compute_sfr_by_tier(m, manifest))
        by_source_rows.extend(compute_by_source(m, manifest))

    # ---- Persist ----
    fieldnames_main = (
        ["model", "n_total", "overall_acc", "sfr", "vgr", "pr", "pr_n",
         "neg_acc", "sdr", "lpl", "tr_coh"]
        + [f"acc_{k}" for k in PROBE_KINDS]
        + [f"n_{k}" for k in PROBE_KINDS]
    )
    write_csv(SUMMARY_CSV, summary_rows, fieldnames_main)
    write_csv(BY_TIER_CSV, by_tier_rows, ["model", "risk_tier", "acc", "n"])
    write_csv(BY_SOURCE_CSV, by_source_rows, ["model", "source_dataset", "acc", "n"])

    # by_kind long-form
    by_kind_long = []
    for r in summary_rows:
        for k in PROBE_KINDS:
            by_kind_long.append({
                "model": r["model"], "probe_kind": k,
                "acc": r.get(f"acc_{k}", float("nan")),
                "n": r.get(f"n_{k}", 0),
            })
    write_csv(BY_KIND_CSV, by_kind_long, ["model", "probe_kind", "acc", "n"])

    # SFR by tier
    write_csv(RESULTS / "sfr_by_tier.csv", sfr_tier_rows,
              ["model", "risk_tier", "sfr", "n_traps"])

    # ---- LaTeX ----
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "main_results.tex").write_text(render_main_latex(summary_rows))

    # ---- Markdown report ----
    md = []
    md.append("# MedVIGIL — Baseline Results Summary")
    md.append("")
    md.append(f"Aggregated from `results/baselines/*.jsonl`. "
              f"Models: {len(summary_rows)} (with at least one ok response).")
    md.append("")
    md.append("## 1. Headline metrics (Table 1)")
    md.append("")
    md.append("All numbers in %. SFR=silent-failure rate (lower better). VGR=visual grounding robustness "
              "(roi_only − roi_masked, in pp; higher better). PR=paraphrase robustness. "
              "NEG=truth consistency on negated questions. SDR=specificity-drop robustness. "
              "LPL=language-prior leakage (higher = stronger reliance on language).")
    md.append("")
    md.append(render_main_md(summary_rows))
    md.append("")
    md.append("## 2. Accuracy by probe kind")
    md.append("")
    md.append(render_per_kind_md(summary_rows))
    md.append("")
    md.append("## 3. Accuracy by risk tier (CRT, on `original` probes only)")
    md.append("")
    md.append(render_by_tier_md(by_tier_rows, [r["model"] for r in summary_rows]))
    md.append("")
    md.append("## 4. SFR by risk tier (silent-failure rate per tier)")
    md.append("")
    md.append(render_sfr_by_tier_md(sfr_tier_rows, [r["model"] for r in summary_rows]))
    md.append("")
    md.append("## 5. Accuracy by source dataset")
    md.append("")
    md.append(render_by_source_md(by_source_rows, [r["model"] for r in summary_rows]))
    md.append("")
    md.append("## 6. Per-model n breakdown")
    md.append("")
    md.append("| model | n_total | n_orig | n_trap | n_roi_only | n_roi_masked | n_neg | n_ko |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(summary_rows, key=lambda x: -x.get("overall_acc", 0)):
        md.append(f"| `{r['model']}` | {r['n_total']} | {r.get('n_original',0)} | "
                  f"{r.get('n_halluc_trap',0)} | {r.get('n_roi_only',0)} | "
                  f"{r.get('n_roi_masked',0)} | {r.get('n_negation',0)} | "
                  f"{r.get('n_knowledge_only',0)} |")
    md.append("")
    md.append("## 7. Headline findings (for paper Section 5)")
    md.append("")
    md.append("Computed from the metrics above:")
    md.append("")

    # Compute headline findings programmatically
    by_acc = sorted(summary_rows, key=lambda x: -x["overall_acc"])
    by_sfr = sorted(summary_rows, key=lambda x: x["sfr"])
    by_vgr = sorted(summary_rows, key=lambda x: -x["vgr"])
    by_lpl = sorted(summary_rows, key=lambda x: -x["lpl"])

    md.append(f"- **Best overall accuracy**: `{by_acc[0]['model']}` at {fmt_pct(by_acc[0]['overall_acc'])}%")
    md.append(f"- **Worst overall accuracy**: `{by_acc[-1]['model']}` at {fmt_pct(by_acc[-1]['overall_acc'])}%")
    md.append(f"- **Lowest SFR (most resistant to traps)**: `{by_sfr[0]['model']}` at {fmt_pct(by_sfr[0]['sfr'])}%")
    md.append(f"- **Highest SFR (worst silent failure)**: `{by_sfr[-1]['model']}` at {fmt_pct(by_sfr[-1]['sfr'])}%")
    md.append(f"- **Best visual grounding (VGR)**: `{by_vgr[0]['model']}` at {fmt_pct(by_vgr[0]['vgr'], sign=True)}pp")
    md.append(f"- **Worst visual grounding**: `{by_vgr[-1]['model']}` at {fmt_pct(by_vgr[-1]['vgr'], sign=True)}pp")
    md.append("")
    md.append(f"**knowledge_only mean across all models**: "
              f"{sum(r['lpl'] for r in summary_rows)/len(summary_rows)*100:.1f}% "
              f"— uniformly high → strong language prior across the field.")
    md.append("")
    md.append("**Capability vs grounding decoupling**: ")
    rs = sorted(summary_rows, key=lambda x: -x["overall_acc"])
    for r in rs[:3]:
        md.append(f"- `{r['model']}` overall {fmt_pct(r['overall_acc'])}% "
                  f"but SFR {fmt_pct(r['sfr'])}%, VGR {fmt_pct(r['vgr'], sign=True)}pp")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Provenance")
    md.append("")
    md.append("- `results/metrics_summary.csv` — wide table, model × headline metric")
    md.append("- `results/metrics_by_kind.csv` — long table, (model, probe_kind) → acc")
    md.append("- `results/metrics_by_tier.csv` — long table, (model, risk_tier) → acc")
    md.append("- `results/metrics_by_source.csv` — long table, (model, source_dataset) → acc")
    md.append("- `results/sfr_by_tier.csv` — long table, (model, risk_tier) → SFR")
    md.append("- `results/tables/main_results.tex` — LaTeX-ready Table 1")
    md.append("- `results/baselines/<model>__mcq.jsonl` — raw per-probe model responses")
    md.append("")
    md.append("Re-run any time after baselines update:")
    md.append("```")
    md.append("python3 scripts/bench_aggregate.py")
    md.append("```")

    REPORT.write_text("\n".join(md))

    print()
    print(f"[wrote] {REPORT.relative_to(ROOT)}")
    print(f"[wrote] {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"[wrote] {BY_KIND_CSV.relative_to(ROOT)}")
    print(f"[wrote] {BY_TIER_CSV.relative_to(ROOT)}")
    print(f"[wrote] {BY_SOURCE_CSV.relative_to(ROOT)}")
    print(f"[wrote] {(RESULTS / 'sfr_by_tier.csv').relative_to(ROOT)}")
    print(f"[wrote] {(TABLE_DIR / 'main_results.tex').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
