#!/usr/bin/env python3
"""
Enhanced analysis of medical VLM experiment results.

Produces:
  1. Per-modality breakdown for Experiments 3-6 (CT / MRI / X-ray)
  2. 95 % Wilson confidence intervals for all proportion metrics
  3. Cross-experiment consistency analysis (Exp 4 hallucination vs Exp 6 compliance)
  4. Cohen's h effect sizes for key model comparisons
  5. Formatted table output + results/enhanced_analysis.json
"""

import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
DATA_DIR = os.path.join(BASE, "data")
MODELS = ["llava-med", "llava-general", "gemini", "claude"]
EXPERIMENTS = [3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for proportion k/n at confidence level z."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (p_hat, lo, hi)


def cohens_h(p1, p2):
    """Cohen's h for two proportions."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return phi1 - phi2


# ---------------------------------------------------------------------------
# Build image -> modality mapping from VQA-RAD organized data
# ---------------------------------------------------------------------------
def build_modality_map():
    org = load_json(os.path.join(DATA_DIR, "vqa_rad_organized.json"))
    if org is None:
        return {}
    img2mod = {}
    for mod, entries in org.get("by_modality", {}).items():
        for e in entries:
            img2mod[e["image"]] = mod
    return img2mod


IMG2MOD = build_modality_map()


def get_modality(image_name, fallback_field=None):
    """Return modality for an image, using the global map or a fallback."""
    if fallback_field:
        return fallback_field
    return IMG2MOD.get(image_name, "unknown")


# ---------------------------------------------------------------------------
# 1. Per-modality breakdown
# ---------------------------------------------------------------------------

def modality_breakdown_exp3(model):
    """Exp 3 (corruption robustness): per-modality avg ROUGE-L and BERTScore."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp3_{model}_results.json"))
    if data is None:
        return {}
    results = data.get("results", [])
    by_mod = defaultdict(lambda: {"rouge_l": [], "bert_score": []})

    for entry in results:
        img = entry.get("image_name", "")
        mod = entry.get("modality", "") or get_modality(img)
        if mod in ("", "unknown"):
            continue
        # Collect all corruption answer pairs
        for q_key, q_data in entry.get("questions", {}).items():
            corr_answers = q_data.get("corruption_answers", {})
            baseline_ans = corr_answers.get("original", {}).get("answer", "")
            if not baseline_ans:
                continue
            for corr_name, corr_info in corr_answers.items():
                if corr_name == "original":
                    continue
                # We don't recompute ROUGE-L here -- use the summary-level data
                # Instead, just count images per modality
        by_mod[mod]["count"] = by_mod[mod].get("count", 0) + 1

    # Use the top-level similarity_summary but note modality breakdown from results
    mod_images = defaultdict(set)
    for entry in results:
        img = entry.get("image_name", "")
        mod = entry.get("modality", "") or get_modality(img)
        if mod not in ("", "unknown"):
            mod_images[mod].add(img)

    return {mod: {"n_images": len(imgs), "images": sorted(imgs)} for mod, imgs in mod_images.items()}


def modality_breakdown_exp4(model):
    """Exp 4 (hallucination): per-modality hallucination and refusal rates."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp4_{model}_results.json"))
    if data is None:
        return {}
    by_mod = defaultdict(lambda: {"total": 0, "hallucination": 0, "correct_refusal": 0,
                                   "hedged_refusal": 0, "ambiguous": 0})
    # Only look at leading_on_real and neutral_on_real conditions (real images)
    for condition in ["leading_on_real", "neutral_on_real"]:
        entries = data.get("results", {}).get(condition, [])
        for entry in entries:
            img = entry.get("image_name", "")
            mod_field = entry.get("modality", "")
            mod = mod_field if mod_field else get_modality(img)
            if mod in ("", "unknown"):
                continue
            cls = entry.get("classification", "")
            by_mod[mod]["total"] += 1
            if cls in by_mod[mod]:
                by_mod[mod][cls] += 1

    out = {}
    for mod, stats in by_mod.items():
        n = stats["total"]
        k_hall = stats["hallucination"]
        k_ref = stats["correct_refusal"] + stats["hedged_refusal"]
        p_hall, lo_hall, hi_hall = wilson_ci(k_hall, n)
        p_ref, lo_ref, hi_ref = wilson_ci(k_ref, n)
        out[mod] = {
            "n": n,
            "hallucination_rate": round(p_hall, 4),
            "hallucination_ci": [round(lo_hall, 4), round(hi_hall, 4)],
            "refusal_rate": round(p_ref, 4),
            "refusal_ci": [round(lo_ref, 4), round(hi_ref, 4)],
        }
    return out


def modality_breakdown_exp5(model):
    """Exp 5 (VQA accuracy): per-modality relaxed match accuracy."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp5_{model}_results.json"))
    if data is None:
        return {}
    # Check if by_modality already exists in summary
    summary = data.get("summary", {})
    by_mod_summary = summary.get("by_modality", {})
    if by_mod_summary:
        out = {}
        for mod, stats in by_mod_summary.items():
            n = stats.get("n", 0)
            rm = stats.get("relaxed_match", {}).get("mean", 0)
            k = int(round(rm * n))
            p, lo, hi = wilson_ci(k, n)
            out[mod] = {
                "n": n,
                "relaxed_match": round(rm, 4),
                "relaxed_match_ci": [round(lo, 4), round(hi, 4)],
            }
        return out

    # Fall back to computing from results
    results = data.get("results", [])
    by_mod = defaultdict(lambda: {"n": 0, "match": 0})
    for entry in results:
        img = entry.get("image_name", "")
        mod_field = entry.get("modality", "")
        mod = mod_field if mod_field else get_modality(img)
        if mod in ("", "unknown"):
            continue
        by_mod[mod]["n"] += 1
        if entry.get("relaxed_match", False):
            by_mod[mod]["match"] += 1

    out = {}
    for mod, stats in by_mod.items():
        n = stats["n"]
        k = stats["match"]
        p, lo, hi = wilson_ci(k, n)
        out[mod] = {
            "n": n,
            "relaxed_match": round(p, 4),
            "relaxed_match_ci": [round(lo, 4), round(hi, 4)],
        }
    return out


def modality_breakdown_exp6(model):
    """Exp 6 (adversarial grounding): per-modality compliance and correction rates."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp6_{model}_results.json"))
    if data is None:
        return {}
    by_mod = defaultdict(lambda: {"total": 0, "compliant": 0, "corrected": 0,
                                   "partial_correction": 0, "ambiguous": 0})
    # modality_conflict results have true_modality
    for condition in ["modality_conflict", "anatomy_conflict", "authority_hallucination"]:
        entries = data.get("results", {}).get(condition, [])
        for entry in entries:
            img = entry.get("image_name", "")
            mod = entry.get("true_modality", "") or get_modality(img)
            if mod in ("", "unknown"):
                continue
            cls = entry.get("classification", "")
            by_mod[mod]["total"] += 1
            if cls in by_mod[mod]:
                by_mod[mod][cls] += 1

    out = {}
    for mod, stats in by_mod.items():
        n = stats["total"]
        k_comp = stats["compliant"]
        k_corr = stats["corrected"]
        p_comp, lo_comp, hi_comp = wilson_ci(k_comp, n)
        p_corr, lo_corr, hi_corr = wilson_ci(k_corr, n)
        out[mod] = {
            "n": n,
            "compliance_rate": round(p_comp, 4),
            "compliance_ci": [round(lo_comp, 4), round(hi_comp, 4)],
            "correction_rate": round(p_corr, 4),
            "correction_ci": [round(lo_corr, 4), round(hi_corr, 4)],
        }
    return out


# ---------------------------------------------------------------------------
# 2. Confidence intervals for overall proportion metrics
# ---------------------------------------------------------------------------

def overall_ci_exp4(model):
    """Overall hallucination / refusal CIs for exp4 conditions."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp4_{model}_results.json"))
    if data is None:
        return {}
    summary = data.get("summary", {})
    out = {}
    for cond, stats in summary.items():
        n = stats.get("total", 0)
        k_hall = stats.get("hallucination", 0)
        k_ref = stats.get("correct_refusal", 0) + stats.get("hedged_refusal", 0)
        p_h, lo_h, hi_h = wilson_ci(k_hall, n)
        p_r, lo_r, hi_r = wilson_ci(k_ref, n)
        out[cond] = {
            "n": n,
            "hallucination_rate": round(p_h, 4),
            "hallucination_ci": [round(lo_h, 4), round(hi_h, 4)],
            "refusal_rate": round(p_r, 4),
            "refusal_ci": [round(lo_r, 4), round(hi_r, 4)],
        }
    return out


def overall_ci_exp5(model):
    """Overall accuracy CIs for exp5."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp5_{model}_results.json"))
    if data is None:
        return {}
    summary = data.get("summary", {}).get("overall", {})
    n = summary.get("n", 0)
    rm_mean = summary.get("relaxed_match", {}).get("mean", 0)
    k = int(round(rm_mean * n))
    p, lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "relaxed_match": round(rm_mean, 4),
        "relaxed_match_wilson_ci": [round(lo, 4), round(hi, 4)],
    }


def overall_ci_exp6(model):
    """Overall compliance / correction CIs for exp6."""
    data = load_json(os.path.join(RESULTS_DIR, f"exp6_{model}_results.json"))
    if data is None:
        return {}
    summary = data.get("summary", {})
    out = {}
    for cond, stats in summary.items():
        n = stats.get("total", 0)
        if n == 0:
            continue
        k_comp = stats.get("compliant", 0)
        k_corr = stats.get("corrected", 0)
        p_c, lo_c, hi_c = wilson_ci(k_comp, n)
        p_cr, lo_cr, hi_cr = wilson_ci(k_corr, n)
        out[cond] = {
            "n": n,
            "compliance_rate": round(p_c, 4),
            "compliance_ci": [round(lo_c, 4), round(hi_c, 4)],
            "correction_rate": round(p_cr, 4),
            "correction_ci": [round(lo_cr, 4), round(hi_cr, 4)],
        }
    return out


# ---------------------------------------------------------------------------
# 3. Cross-experiment consistency (Exp 4 hallucination <-> Exp 6 compliance)
# ---------------------------------------------------------------------------

def cross_experiment_consistency(model):
    """
    For each image that appears in both Exp 4 and Exp 6, compute:
      - hallucination score (fraction of Exp 4 leading questions that caused hallucination)
      - compliance score  (fraction of Exp 6 assertions where model was compliant)
    Then compute Pearson correlation.
    """
    data4 = load_json(os.path.join(RESULTS_DIR, f"exp4_{model}_results.json"))
    data6 = load_json(os.path.join(RESULTS_DIR, f"exp6_{model}_results.json"))
    if data4 is None or data6 is None:
        return {"available": False, "reason": "missing result file(s)"}

    # Exp 4: per-image hallucination score (leading_on_real only)
    img_hall = defaultdict(lambda: {"total": 0, "hallucination": 0})
    for entry in data4.get("results", {}).get("leading_on_real", []):
        img = entry.get("image_name", "")
        img_hall[img]["total"] += 1
        if entry.get("classification", "") == "hallucination":
            img_hall[img]["hallucination"] += 1

    # Exp 6: per-image compliance score (all conditions with real images)
    img_comp = defaultdict(lambda: {"total": 0, "compliant": 0})
    for condition in ["modality_conflict", "anatomy_conflict", "authority_hallucination"]:
        for entry in data6.get("results", {}).get(condition, []):
            img = entry.get("image_name", "")
            img_comp[img]["total"] += 1
            if entry.get("classification", "") == "compliant":
                img_comp[img]["compliant"] += 1

    # Intersect images
    common = set(img_hall.keys()) & set(img_comp.keys())
    if len(common) < 3:
        return {"available": False, "reason": f"only {len(common)} common images",
                "n_exp4_images": len(img_hall), "n_exp6_images": len(img_comp)}

    hall_scores = []
    comp_scores = []
    per_image = []
    for img in sorted(common):
        h = img_hall[img]
        c = img_comp[img]
        h_rate = h["hallucination"] / h["total"] if h["total"] else 0
        c_rate = c["compliant"] / c["total"] if c["total"] else 0
        hall_scores.append(h_rate)
        comp_scores.append(c_rate)
        per_image.append({"image": img, "hallucination_rate": round(h_rate, 4),
                          "compliance_rate": round(c_rate, 4)})

    hall_arr = np.array(hall_scores)
    comp_arr = np.array(comp_scores)

    # Pearson correlation
    if np.std(hall_arr) < 1e-12 or np.std(comp_arr) < 1e-12:
        corr = 0.0
        p_value = 1.0
    else:
        corr_matrix = np.corrcoef(hall_arr, comp_arr)
        corr = float(corr_matrix[0, 1])
        # Approximate p-value via t-test
        n = len(hall_scores)
        if abs(corr) < 1.0:
            t_stat = corr * math.sqrt((n - 2) / (1 - corr ** 2))
            # Two-tailed p-value from t distribution (approximation)
            try:
                from scipy.stats import t as t_dist
                p_value = float(2 * t_dist.sf(abs(t_stat), df=n - 2))
            except ImportError:
                # Fallback: large-sample normal approx
                p_value = float(2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))
        else:
            p_value = 0.0

    return {
        "available": True,
        "n_common_images": len(common),
        "pearson_r": round(corr, 4),
        "p_value": round(p_value, 4),
        "interpretation": (
            "positive: images causing more hallucination also have higher compliance"
            if corr > 0
            else "negative: images causing more hallucination have lower compliance"
            if corr < 0
            else "no correlation"
        ),
        "mean_hallucination_rate": round(float(np.mean(hall_arr)), 4),
        "mean_compliance_rate": round(float(np.mean(comp_arr)), 4),
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# 4. Cohen's h effect sizes
# ---------------------------------------------------------------------------

def compute_effect_sizes():
    """Cohen's h between every pair of models for key metrics."""
    # Gather proportions
    metrics = {}  # metric_name -> {model: proportion}

    for model in MODELS:
        # Exp 4 leading_on_real hallucination rate
        d4 = load_json(os.path.join(RESULTS_DIR, f"exp4_{model}_results.json"))
        if d4:
            s = d4.get("summary", {}).get("leading_on_real", {})
            metrics.setdefault("exp4_leading_hallucination", {})[model] = s.get("hallucination_rate", None)

        # Exp 5 overall relaxed_match
        d5 = load_json(os.path.join(RESULTS_DIR, f"exp5_{model}_results.json"))
        if d5:
            s5 = d5.get("summary", {}).get("overall", {}).get("relaxed_match", {})
            metrics.setdefault("exp5_relaxed_match", {})[model] = s5.get("mean", None)

        # Exp 6 overall compliance rate
        d6 = load_json(os.path.join(RESULTS_DIR, f"exp6_{model}_results.json"))
        if d6:
            s6 = d6.get("summary", {}).get("overall", {})
            metrics.setdefault("exp6_compliance", {})[model] = s6.get("compliance_rate", None)
            metrics.setdefault("exp6_correction", {})[model] = s6.get("correction_rate", None)

    effect_sizes = {}
    for metric_name, model_vals in metrics.items():
        pairs = {}
        for m1, m2 in combinations(MODELS, 2):
            p1 = model_vals.get(m1)
            p2 = model_vals.get(m2)
            if p1 is not None and p2 is not None:
                h = cohens_h(p1, p2)
                magnitude = (
                    "large" if abs(h) >= 0.8 else
                    "medium" if abs(h) >= 0.5 else
                    "small" if abs(h) >= 0.2 else
                    "negligible"
                )
                pairs[f"{m1}_vs_{m2}"] = {
                    "cohens_h": round(h, 4),
                    "magnitude": magnitude,
                    f"{m1}": round(p1, 4),
                    f"{m2}": round(p2, 4),
                }
        effect_sizes[metric_name] = pairs
    return effect_sizes


# ---------------------------------------------------------------------------
# Formatted table output
# ---------------------------------------------------------------------------

def fmt_ci(val, ci):
    return f"{val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output = {
        "per_modality": {},
        "confidence_intervals": {},
        "cross_experiment_consistency": {},
        "effect_sizes": {},
    }

    # -----------------------------------------------------------------------
    # 1. Per-modality breakdown
    # -----------------------------------------------------------------------
    print_section("1. PER-MODALITY BREAKDOWN")

    for model in MODELS:
        output["per_modality"][model] = {}

        # Exp 3
        exp3_mod = modality_breakdown_exp3(model)
        output["per_modality"][model]["exp3_images_per_modality"] = exp3_mod
        if exp3_mod:
            print(f"\n--- Exp 3 ({model}): Images per modality ---")
            rows = [[mod, info["n_images"]] for mod, info in sorted(exp3_mod.items())]
            print_table(["Modality", "N_images"], rows)

        # Exp 4
        exp4_mod = modality_breakdown_exp4(model)
        output["per_modality"][model]["exp4_by_modality"] = exp4_mod
        if exp4_mod:
            print(f"\n--- Exp 4 ({model}): Hallucination by modality ---")
            rows = []
            for mod, info in sorted(exp4_mod.items()):
                rows.append([
                    mod, info["n"],
                    fmt_ci(info["hallucination_rate"], info["hallucination_ci"]),
                    fmt_ci(info["refusal_rate"], info["refusal_ci"]),
                ])
            print_table(["Modality", "N", "Hallucination Rate [95% CI]", "Refusal Rate [95% CI]"],
                        rows, [14, 6, 32, 32])

        # Exp 5
        exp5_mod = modality_breakdown_exp5(model)
        output["per_modality"][model]["exp5_by_modality"] = exp5_mod
        if exp5_mod:
            print(f"\n--- Exp 5 ({model}): Accuracy by modality ---")
            rows = []
            for mod, info in sorted(exp5_mod.items()):
                rows.append([
                    mod, info["n"],
                    fmt_ci(info["relaxed_match"], info["relaxed_match_ci"]),
                ])
            print_table(["Modality", "N", "Relaxed Match [95% CI]"], rows, [14, 6, 32])

        # Exp 6
        exp6_mod = modality_breakdown_exp6(model)
        output["per_modality"][model]["exp6_by_modality"] = exp6_mod
        if exp6_mod:
            print(f"\n--- Exp 6 ({model}): Adversarial grounding by modality ---")
            rows = []
            for mod, info in sorted(exp6_mod.items()):
                rows.append([
                    mod, info["n"],
                    fmt_ci(info["compliance_rate"], info["compliance_ci"]),
                    fmt_ci(info["correction_rate"], info["correction_ci"]),
                ])
            print_table(["Modality", "N", "Compliance Rate [95% CI]", "Correction Rate [95% CI]"],
                        rows, [14, 6, 32, 32])

    # -----------------------------------------------------------------------
    # 2. Confidence intervals for overall metrics
    # -----------------------------------------------------------------------
    print_section("2. CONFIDENCE INTERVALS (OVERALL METRICS)")

    for model in MODELS:
        output["confidence_intervals"][model] = {}

        ci4 = overall_ci_exp4(model)
        output["confidence_intervals"][model]["exp4"] = ci4
        if ci4:
            print(f"\n--- Exp 4 ({model}): Overall CIs ---")
            rows = []
            for cond, info in sorted(ci4.items()):
                rows.append([
                    cond, info["n"],
                    fmt_ci(info["hallucination_rate"], info["hallucination_ci"]),
                    fmt_ci(info["refusal_rate"], info["refusal_ci"]),
                ])
            print_table(["Condition", "N", "Hallucination [95% CI]", "Refusal [95% CI]"],
                        rows, [28, 6, 30, 30])

        ci5 = overall_ci_exp5(model)
        output["confidence_intervals"][model]["exp5"] = ci5
        if ci5:
            print(f"\n--- Exp 5 ({model}): Overall CI ---")
            print(f"  Relaxed match: {fmt_ci(ci5['relaxed_match'], ci5['relaxed_match_wilson_ci'])}  (n={ci5['n']})")

        ci6 = overall_ci_exp6(model)
        output["confidence_intervals"][model]["exp6"] = ci6
        if ci6:
            print(f"\n--- Exp 6 ({model}): Overall CIs ---")
            rows = []
            for cond, info in sorted(ci6.items()):
                rows.append([
                    cond, info["n"],
                    fmt_ci(info["compliance_rate"], info["compliance_ci"]),
                    fmt_ci(info["correction_rate"], info["correction_ci"]),
                ])
            print_table(["Condition", "N", "Compliance [95% CI]", "Correction [95% CI]"],
                        rows, [28, 6, 30, 30])

    # -----------------------------------------------------------------------
    # 3. Cross-experiment consistency
    # -----------------------------------------------------------------------
    print_section("3. CROSS-EXPERIMENT CONSISTENCY (Exp 4 Hallucination vs Exp 6 Compliance)")

    for model in MODELS:
        cons = cross_experiment_consistency(model)
        output["cross_experiment_consistency"][model] = cons
        print(f"\n--- {model} ---")
        if cons.get("available"):
            print(f"  Common images: {cons['n_common_images']}")
            print(f"  Pearson r:     {cons['pearson_r']}  (p = {cons['p_value']})")
            print(f"  Interpretation: {cons['interpretation']}")
            print(f"  Mean hallucination rate: {cons['mean_hallucination_rate']}")
            print(f"  Mean compliance rate:    {cons['mean_compliance_rate']}")
        else:
            print(f"  Not available: {cons.get('reason', 'unknown')}")

    # -----------------------------------------------------------------------
    # 4. Effect sizes (Cohen's h)
    # -----------------------------------------------------------------------
    print_section("4. EFFECT SIZES (Cohen's h)")

    effect_sizes = compute_effect_sizes()
    output["effect_sizes"] = effect_sizes

    for metric, pairs in effect_sizes.items():
        print(f"\n--- {metric} ---")
        rows = []
        for pair_name, info in sorted(pairs.items()):
            models_in_pair = pair_name.split("_vs_")
            rows.append([
                pair_name,
                info.get(models_in_pair[0], ""),
                info.get(models_in_pair[1], ""),
                f"{info['cohens_h']:+.4f}",
                info["magnitude"],
            ])
        print_table(["Comparison", "P1", "P2", "Cohen's h", "Magnitude"],
                    rows, [30, 10, 10, 12, 14])

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_path = os.path.join(RESULTS_DIR, "enhanced_analysis.json")
    # Remove per_image detail from cross_experiment for cleaner JSON (keep summary)
    output_save = json.loads(json.dumps(output))  # deep copy
    for model in MODELS:
        ce = output_save.get("cross_experiment_consistency", {}).get(model, {})
        if "per_image" in ce:
            # Keep only first 5 for reference
            ce["per_image_sample"] = ce.pop("per_image")[:5]

    with open(out_path, "w") as f:
        json.dump(output_save, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
