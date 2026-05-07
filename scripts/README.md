# `scripts/` — MedVIGIL pipeline

Each script is independently runnable from the project root with
`python3 scripts/<name>.py [--help]`. The pipeline in `bench_freeze.py`
runs them in the canonical order.

## Active pipeline (build the benchmark)

| # | Script | Role | Inputs | Outputs |
|---|--------|------|--------|---------|
| 1 | `bench_init_manifest.py`         | Layer A: copy human annotations + images into `data/medvigil_v1/` | `data/annotations/master_300.csv` | `manifest.csv`, `probes.csv` (initial), `images/` |
| 2 | `bench_generate_layers_bcd.py`   | <MODEL> mega-prompt: Layer B grounding + Layer C1 new text probes + Layer D V-CF anchor | `manifest.csv` + images | `raw_clinician/<case>.json` |
| 3 | `bench_make_image_perturbations.py` | Layer C2 programmatic image perturbations (ROI-masked / ROI-only / LR-flip) | `raw_clinician/*.json` (for bbox) + images | `images_perturbed/`, `_image_probes.csv` |
| 4 | `bench_consolidate_probes.py`    | Merge initial + Layer C1 + Layer C2 → unified probe table | `_image_probes.csv` + `raw_clinician/*` | `probes_open.csv`, `grounding.csv`, `_vcf_anchors.csv` |
| 5 | `bench_build_triplets.py`        | Layer D triplet assembly + 3 invariant checks | `probes_open.csv` + `_vcf_anchors.csv` | `triplets.csv` |
| 6 | `bench_build_splits.py`          | Layer E stratified splits + Layer F text-only subset | `manifest.csv` + `probes_open.csv` | `splits/*.json` |
| 7 | `bench_generate_mcq.py`          | Wrap each text probe in 5-option MCQ via <MODEL> (async, concurrency=8, resumable) | `probes_open.csv` + `grounding.csv` + `raw_clinician/*` | `mcq_provenance/<probe>.json` |
| 8 | `bench_consolidate_mcq.py`       | Merge MCQ payloads + image-variant inheritance + lr_flip letter logic | `mcq_provenance/*` + `probes_open.csv` | `probes_mcq.csv` |
| 9 | `bench_fix_lr_unmatched.py`      | Regenerate `__orig` MCQ for laterality-dependent cases whose flipped gold didn't match | `probes_mcq.csv` + `raw_clinician/*` | overwrites `mcq_provenance/<case>__orig.json` |
| 10 | `bench_consolidate_mcq.py`      | (re-run after step 9) | as above | `probes_mcq.csv` |
| 11 | `bench_validate_full.py`        | 14-check end-to-end integrity report | all outputs | `validation_report.md` |

`bench_freeze.py` orchestrates all 11 steps idempotently. Each step is
resumable independently.

## QA / inspection (no LLM cost)

| Script | Purpose |
|--------|---------|
| `bench_validate_full.py` | Run anytime to get pass/warn/fail on 14 integrity gates |
| `bench_visualize_roi.py` | Render 4-panel diagnostic PNGs (`qa_review/<case>.png`) so a doctor can spot-check 30-50 ROI bboxes in <30 minutes |

## Downstream evaluation

| Script | Purpose |
|--------|---------|
| `bench_judge_open.py` | LLM-judge for **open-ended** model outputs. Takes `<model>.jsonl` of model responses → emits per-row verdicts (`correct` / `incorrect` / `refused_appropriately` / `silent_failure` / `hedged`). Uses fast heuristics first, <MODEL> LLM-judge for ambiguous cases. |

Note: MCQ-format scoring needs no LLM-judge — score by string-matching
the model's letter against `correct_letter` in `probes_mcq.csv`.

## Environment

All scripts that call OpenAI auto-load `.env` via `api_models._load_dotenv`.
Required: `OPENAI_API_KEY` set in `.env` at the project root.

Models used:
- `<MODEL>` — for all dataset construction (Layer B / C1 / D / MCQ wrapping)
  and for the LLM-judge fallback in `bench_judge_open.py`.

## Resumability

Every <MODEL> generation script writes one JSON per row (per case or per
probe). Re-running the script skips rows whose JSON already has
`status: "ok"`. To force regeneration of a single case/probe, delete the
corresponding file under `raw_clinician/` or `mcq_provenance/`.

## `legacy/`

Pre-MedVIGIL scripts (Phase-1 annotation pack builder, audit, etc.)
are archived under `scripts/legacy/`. They are kept for reproducibility
of the upstream data preparation but are not part of the v1 pipeline.
