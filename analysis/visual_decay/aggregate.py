"""Aggregate blur-sweep predictions into a tidy CSV for plotting.

For every (model, sigma) JSONL under results/blur_sweep/, this script joins
each prediction with manifest.csv to retrieve `risk_tier` and
`text_only_answerable`, then computes exact-letter accuracy per
(model, sigma, group, tier) and writes one tidy CSV.

Output columns:
    model_id, sigma (str), n, acc, group, tier
where:
    group in {all, image_required, text_answerable}
    tier  in {all, L1, L2, L3, L4, L5}

Re-running is idempotent — it overwrites the output CSV.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data/medvlm_bench_v1"
RESULTS = ROOT / "results/blur_sweep"
OUT_CSV = RESULTS / "aggregate.csv"

SIGMAS = ["0", "2", "4", "8", "16", "32", "64", "inf"]


def load_manifest() -> dict[str, dict]:
    by_case = {}
    with (BENCH / "manifest.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            by_case[r["case_id"]] = {
                "risk_tier": r["risk_tier"],
                "text_only_answerable": (r["text_only_answerable"] or "").strip().lower() in ("true", "1", "yes"),
            }
    return by_case


def iter_predictions(path: Path):
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    cases = load_manifest()

    rows: list[dict] = []
    seen_files = []
    for jf in sorted(RESULTS.glob("*.jsonl")):
        # filename: <safe_model>__sigma<sigma>.jsonl
        stem = jf.stem
        if "__sigma" not in stem:
            continue
        model_id, _, sigma = stem.partition("__sigma")
        # skip aggregate / scratch files
        if not sigma:
            continue
        seen_files.append(jf.name)

        # group preds by case_id (keep latest entry per probe_id; jsonl is append-only with resume)
        by_pid = {}
        for p in iter_predictions(jf):
            if p.get("status") != "ok":
                continue
            by_pid[p["probe_id"]] = p

        # bucket counts
        n = {("all", "all"): 0, ("image_required", "all"): 0, ("text_answerable", "all"): 0}
        c = {("all", "all"): 0, ("image_required", "all"): 0, ("text_answerable", "all"): 0}
        for tier in ("L1", "L2", "L3", "L4", "L5"):
            n[("all", tier)] = 0
            c[("all", tier)] = 0
            n[("image_required", tier)] = 0
            c[("image_required", tier)] = 0

        for pid, p in by_pid.items():
            cid = p.get("case_id")
            meta = cases.get(cid)
            if not meta:
                continue
            tier = meta["risk_tier"]
            txtonly = meta["text_only_answerable"]
            ok = (p.get("model_letter") == p.get("correct_letter") and p.get("model_letter"))

            for grp in ("all",
                        ("image_required" if not txtonly else "text_answerable")):
                n[(grp, "all")] = n.get((grp, "all"), 0) + 1
                c[(grp, "all")] = c.get((grp, "all"), 0) + (1 if ok else 0)
                key_t = (grp, tier)
                n[key_t] = n.get(key_t, 0) + 1
                c[key_t] = c.get(key_t, 0) + (1 if ok else 0)

        for (grp, tier), ni in n.items():
            if ni == 0:
                continue
            rows.append({
                "model_id": model_id,
                "sigma": sigma,
                "group": grp,
                "tier": tier,
                "n": ni,
                "correct": c[(grp, tier)],
                "acc": round(c[(grp, tier)] / ni, 6),
            })

    rows.sort(key=lambda r: (r["model_id"], r["sigma"], r["group"], r["tier"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "sigma", "group", "tier", "n", "correct", "acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} rows from {len(seen_files)} files -> {OUT_CSV.relative_to(ROOT)}")
    print(f"[ok] files seen: {seen_files}")


if __name__ == "__main__":
    main()
