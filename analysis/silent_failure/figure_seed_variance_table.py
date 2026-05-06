"""Generate the seed-variance Table 4 + top-findings figure for the paper.
Reads results/seed_variance/rescored.json and emits LaTeX into
paper/figures/table_seed_variance.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


MODEL_DISPLAY = {"claude": "Claude", "gpt4o": "GPT-4o", "gemini": "Gemini",
                 "llava-med": "LLaVA-Med", "llava-general": "LLaVA-1.6"}
PROMPT_DISPLAY = {"chest_xray": "chest X-ray", "head_ct": "head CT", "ecg": "ECG"}
CLASS_ORDER = ["CLEAN_REFUSAL", "PEDAGOGICAL_HEDGE",
               "EDUCATIONAL_DEFLECTION", "SILENT_FAILURE", "LOW_SEVERITY"]
CLASS_DISPLAY = {
    "CLEAN_REFUSAL": "clean refusal",
    "PEDAGOGICAL_HEDGE": "pedag. hedge",
    "EDUCATIONAL_DEFLECTION": "educ. deflect.",
    "SILENT_FAILURE": "silent fail.",
    "LOW_SEVERITY": "low sev.",
}


def fmt_pct(x):
    return f"{x*100:.0f}\\%"


def render_class_table(data: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Seed-variance probe under $\delta_\text{missing}$. Each cell is the share of $K=50$ generations classified into one of four behaviors. \emph{Clean refusal} is the only fully safe category; the remaining three all leak severity-tier vocabulary into the response with progressively weaker disclaimers (Section~\ref{sec:taxonomy-output}).}",
        r"\label{tab:seed-variance}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Prompt & " + " & ".join(CLASS_DISPLAY[c] for c in CLASS_ORDER) + r" \\",
        r"\midrule",
    ]
    for model in ["claude", "gpt4o"]:
        if model not in data:
            continue
        for i, prompt in enumerate(["chest_xray", "head_ct", "ecg"]):
            row = data[model].get(prompt, {})
            rates = row.get("class_rates", {})
            cells = [
                MODEL_DISPLAY[model] if i == 0 else "",
                PROMPT_DISPLAY[prompt],
            ] + [fmt_pct(rates.get(c, 0)) for c in CLASS_ORDER]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if model == "claude" else "")
    lines = [ln for ln in lines if ln != ""]
    if lines[-1].strip() == r"\midrule":
        lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_findings_table(data: dict, top_n: int = 6) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Top severity-tier ($\sigma \geq 3$) findings surfaced by each model under $\delta_\text{missing}$, with raw counts across $K=50$ generations. GPT-4o on the head-CT prompt mentions \emph{hemorrhage} in every generation despite no image being provided.}",
        r"\label{tab:seed-variance-findings}",
        r"\begin{tabular}{llp{8cm}}",
        r"\toprule",
        r"Model & Prompt & Top severity-tier findings (count / 50) \\",
        r"\midrule",
    ]
    for model in ["claude", "gpt4o"]:
        if model not in data:
            continue
        for i, prompt in enumerate(["chest_xray", "head_ct", "ecg"]):
            row = data[model].get(prompt, {})
            findings = row.get("top_findings", [])[:top_n]
            if not findings:
                phrase_str = "--"
            else:
                phrase_str = ", ".join(f"\\emph{{{p}}} ({n})" for p, n in findings)
            cells = [
                MODEL_DISPLAY[model] if i == 0 else "",
                PROMPT_DISPLAY[prompt],
                phrase_str,
            ]
            lines.append(" & ".join(cells) + r" \\")
        if model == "claude":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "results" / "seed_variance" / "rescored.json"))
    p.add_argument("--out", default=str(ROOT / "paper" / "figures" / "table_seed_variance.tex"))
    args = p.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    out_lines = [
        "% =====================================================================\n"
        "% Auto-generated from results/seed_variance/rescored.json\n"
        "% Regenerate via: python -m analysis.silent_failure.figure_seed_variance_table\n"
        "% =====================================================================\n\n",
        render_class_table(data),
        "\n",
        render_findings_table(data),
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(out_lines)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
