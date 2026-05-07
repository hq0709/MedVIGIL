# MedVIGIL — Evaluation Harness

**Risk-stratified, multi-axis evaluation harness for trustworthy medical Vision–Language Models.**

This repository holds the **evaluation code** that scores any vision-capable model against the radiologist-supervised MedVIGIL benchmark and reproduces the headline tables and figures in the paper. The dataset itself — manifest, probes, ROI annotations, doctor-finalised gold options, Croissant 1.0 metadata, datasheet — lives separately on Hugging Face:

> 🤗 **Dataset:** <https://huggingface.co/datasets/jhq0709/MedVIGIL>

The dataset is **doctor-authored end to end**: two attending radiologists (R1, R2) annotate every case in parallel, a senior consolidating radiologist (P3) adjudicates and finalises the released manifest, and a separate fourth radiologist (R4) supplies the human reference baseline. This repository deliberately does **not** ship the dataset content or any construction-side tooling — it is read-only with respect to the dataset and focuses solely on evaluation.

---

## Repository layout

```
scripts/
  bench_run_baseline.py                run an OpenAI-compatible vision API on the released MCQ probes
  bench_run_huatuo.py                  run HuatuoGPT-Vision-7B locally (multi-GPU sharding supported)
  bench_run_llava_med.py               run LLaVA-Med-v1.5-mistral-7B locally
  bench_blur_sweep_make.py             pre-render Gaussian-blurred images for the visual-decay ablation
  bench_blur_sweep_run.py              run one API model across all sigma levels
  bench_blur_sweep_run_huatuo.py       same blur sweep for HuatuoGPT-Vision-7B
  run_blur_sweep_one_model.sh          driver shell script for the blur sweep
  bench_visual_token_ablation.py       progressive ROI-mask ablation runner (App. I)
  bench_make_image_perturbations.py    PIL utility: ROI-mask, ROI-only, lr_flip image variants
  bench_judge_open.py                  LLM-judge scorer for the open-ended (free-form) variant
  bench_aggregate.py                   aggregate per-model JSONL results into the metric CSVs
  bench_validate_full.py               14-check end-to-end integrity validator
  bench_visualize_roi.py               quick ROI-box overlay for inspection
  bench_clean_errors.py                re-run failed probes utility

analysis/
  recompute_mcs.py                     MCS under the corrected-PR definition (headline Table)
  bootstrap_mcs.py                     case-clustered bootstrap CIs (Appendix table)
  mcs_sensitivity.py                   weight / Grounding-form sensitivity (Appendix table)
  clinician_baseline_mcs.py            independent-radiologist (R4) baseline composite
  silent_failure/                      headline figure generators + LLM-judge tooling
    llm_judge.py                       open-ended judge with calibration
    llm_judge_calibration.json         per-class calibration thresholds
    refusal.py                         refusal-pattern classifier
    severity.py                        severity-aware metric heads (designed-but-deferred)
    metrics.py                         shared metric primitives
    figure_*.py                        per-figure generators
  visual_decay/                        blur ablation aggregation + figures
    aggregate.py                       case-level blur-sweep aggregation
    figure_visual_decay.py             paper-side blur-sweep figures
  figure_visual_token_ablation.py      visual-token (progressive ROI mask) figure (App. I)
  analyze_results.py                   legacy aggregate analyser
  enhanced_analysis.py                 legacy headline analyser
  generate_figures.py                  legacy figure runner

requirements.txt                        pinned dependencies
setup.sh                                one-shot environment bootstrapper
```

---

## Quick start

```bash
git clone https://github.com/hq0709/MedVIGIL.git
cd MedVIGIL && bash setup.sh
```

### 1. Pull the dataset from Hugging Face

```bash
hf download --repo-type=dataset jhq0709/MedVIGIL --local-dir data/medvigil_v1
```

### 2. Score a new model

```bash
# Any OpenAI-compatible vision API (GPT-5.x, Claude, Gemini, Qwen, Kimi, ...)
python scripts/bench_run_baseline.py \
  --model gpt-4o \
  --probes data/medvigil_v1/probes_mcq.csv \
  --out results/baselines/gpt-4o__mcq.jsonl

# Local open-weight medical VLMs
python scripts/bench_run_huatuo.py    --probes data/medvigil_v1/probes_mcq.csv
python scripts/bench_run_llava_med.py --probes data/medvigil_v1/probes_mcq.csv
```

### 3. Aggregate + score

```bash
python scripts/bench_aggregate.py            # per-model metric CSVs
python analysis/recompute_mcs.py             # headline Table values (Cap, Safe, Ground, MCS)
python analysis/bootstrap_mcs.py             # 95% bootstrap CIs
python analysis/clinician_baseline_mcs.py    # R4 reference baseline
```

### 4. Open-ended LLM judge (optional, for the auxiliary open-ended variant)

```bash
python scripts/bench_judge_open.py \
  --responses results/baselines/gpt-4o__open.jsonl \
  --out results/baselines/gpt-4o__open_judged.jsonl
```

### 5. Visual-decay ablation (Gaussian-blur sweep)

```bash
# Pre-render blurred images at sigma in {0,2,4,8,16,32,64,inf}
python scripts/bench_blur_sweep_make.py
# Sweep one model across all sigmas
bash scripts/run_blur_sweep_one_model.sh gpt-4o 8
# Aggregate per-(model, sigma, tier) accuracy
python analysis/visual_decay/aggregate.py
python analysis/visual_decay/figure_visual_decay.py
```

---

## Reproducibility notes

- Every model is queried with the same prompt template and decoding settings (temperature 0, top-p 1, ≤ 8 output tokens, thinking modes off where exposed). Provider-side snapshot identifiers are listed in the paper Appendix.
- Letter parsing, retry policy, and per-tier weighting match the audit reported in the paper.
- The dataset on Hugging Face is the canonical input; this harness is read-only with respect to the dataset.
- The four-radiologist construction pipeline that produced the released manifest is described in the paper; we do not ship the construction workflow as code.

---

## Citation

```bibtex
@misc{jiang2026medvigil,
  title  = {{MedVIGIL}: Evaluating Trustworthy Medical {VLM}s Under Broken Visual Evidence},
  author = {Jiang, Hanqi and Chen, Junhao and Pan, Yi and Chen, Lifeng and
            You, Weihang and Gong, Haozhen and Yan, Ruiyu and Lv, Jinglei and
            Zhao, Lin and Ren, Hui and Li, Quanzheng and Liu, Tianming and Li, Xiang},
  year   = {2026},
  note   = {Preprint, under review}
}
```

## License

MIT for the harness code in this repository. The dataset on Hugging Face is governed by per-source licences (VQA-RAD CC0, SLAKE CC BY-SA 4.0, ROCO CC BY-NC-SA 4.0, MIMIC-CXR / CheXpert credentialed access). See the dataset README for details.
