"""Render the token-ablation main figure: per-case panels with masked-image
strip on the left and dual-axis (severity / confidence) trajectory on the right.

Inputs:
    --trajectory   results/token_ablation.jsonl produced by experiments/silent_failure/token_ablation.py
    --image-root   directory with original images
    --out          output PDF or PNG

Layout per case (one row of the final figure):
    [orig | mask10 | mask25 | mask50 | mask75 | mask90 | grey ]   |   trajectory plot
    Below each masked image: short caption with the model's current diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.silent_failure.degradations import delta_topk_mask  # noqa: E402

K_DISPLAY = [0, 25, 50, 75, 100]


def shorten(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def render(args):
    rows: List[dict] = []
    with open(args.trajectory, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.cases:
        rows = [r for r in rows if r["case_id"] in args.cases]

    n_cases = len(rows)
    if n_cases == 0:
        print("[err] no trajectory rows to render", file=sys.stderr)
        sys.exit(1)

    n_imgs = len(K_DISPLAY)
    fig_h = 2.6 * n_cases
    fig, axes = plt.subplots(
        n_cases, n_imgs + 1,
        figsize=(2.0 * n_imgs + 4.0, fig_h),
        gridspec_kw={"width_ratios": [1] * n_imgs + [3]},
    )
    if n_cases == 1:
        axes = np.array([axes])

    for row_idx, rec in enumerate(rows):
        case_id = rec["case_id"]
        traj = rec["trajectory"]
        img_path = Path(args.image_root) / rec.get("image_file", f"{case_id}.jpg")
        img = Image.open(img_path).convert("RGB") if img_path.exists() else None

        traj_by_k = {t["k_pct"]: t for t in traj}
        for col_idx, k in enumerate(K_DISPLAY):
            ax = axes[row_idx, col_idx]
            if img is not None and k in traj_by_k:
                if k == 0:
                    disp = img
                elif k == 100:
                    disp = Image.new("RGB", img.size, color=(128, 128, 128))
                else:
                    H = img.size[1] // 14
                    W = img.size[0] // 14
                    sal = np.ones((H, W))
                    disp = delta_topk_mask(img, saliency=sal, k_pct=k)
                ax.imshow(np.array(disp))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"k={k}%", fontsize=8)
            if k in traj_by_k:
                cap = shorten(traj_by_k[k]["answer"], 40)
                ax.set_xlabel(cap, fontsize=6, labelpad=3)
            if col_idx == 0:
                ax.set_ylabel(case_id, fontsize=9)

        ax = axes[row_idx, -1]
        ks = sorted(t["k_pct"] for t in traj)
        sigmas = [traj_by_k[k]["sigma"] for k in ks]
        refs = [int(traj_by_k[k]["is_refusal"]) for k in ks]
        ax.plot(ks, sigmas, "-o", color="tab:red", label=r"severity $\sigma$")
        ax2 = ax.twinx()
        ax2.plot(ks, refs, "-s", color="tab:blue", label="refusal")
        ax.set_xlabel("top-k% patches masked", fontsize=8)
        ax.set_ylabel("severity", color="tab:red", fontsize=8)
        ax2.set_ylabel("refusal", color="tab:blue", fontsize=8)
        ax.set_ylim(-0.2, 4.2)
        ax2.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="both", labelsize=7)
        ax2.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--cases", nargs="+", help="case_ids to plot (default: all)")
    p.add_argument("--out", default=str(ROOT / "paper" / "figures" / "token_ablation.pdf"))
    args = p.parse_args()
    render(args)


if __name__ == "__main__":
    main()
