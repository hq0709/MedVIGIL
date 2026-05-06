"""ROI bbox spot-check visualizer.

For each case with a Layer B ROI bbox, render a 4-panel diagnostic image:
  panel 1: original
  panel 2: original with red bbox overlay
  panel 3: ROI-masked variant
  panel 4: ROI-only variant

Saves to data/medvlm_bench_v1/qa_review/<case_id>.png so a doctor can
flip through 30-50 of them in <30 minutes and accept/reject.

Defaults: render the first 50 cases that have a non-trivial ROI.

Run:
    python3 scripts/bench_visualize_roi.py            # 50 cases
    python3 scripts/bench_visualize_roi.py --all      # all 192 cases
    python3 scripts/bench_visualize_roi.py --case MVB-2034
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvlm_bench_v1"
MANIFEST = BENCH / "manifest.csv"
GROUNDING = BENCH / "grounding.csv"
IMAGES = BENCH / "images"
PERT = BENCH / "images_perturbed"
OUT = BENCH / "qa_review"


def load_grounding() -> dict[str, dict]:
    with open(GROUNDING, newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def load_manifest() -> dict[str, dict]:
    with open(MANIFEST, newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def has_roi(g: dict) -> bool:
    try:
        b = json.loads(g["roi_bbox_norm"])
        if len(b) != 4:
            return False
        x0, y0, x1, y1 = b
        return not (x1 - x0 >= 0.97 and y1 - y0 >= 0.97)
    except Exception:
        return False


def panel_with_bbox(im: Image.Image, bbox_norm: list, label: str) -> Image.Image:
    out = im.copy().convert("RGB")
    w, h = out.size
    x0, y0, x1, y1 = bbox_norm
    px = [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)]
    draw = ImageDraw.Draw(out)
    draw.rectangle(px, outline=(255, 0, 0), width=max(2, int(0.005 * max(w, h))))
    return out


def composite_panel(panels: list[Image.Image], labels: list[str],
                     case_id: str, q: str, gold: str, roi_text: str) -> Image.Image:
    """Combine 4 panels into a 2x2 grid with a header."""
    target = 384
    resized = []
    for p in panels:
        p = p.copy()
        p.thumbnail((target, target))
        resized.append(p)
    pad = 12
    grid_w = target * 2 + pad * 3
    grid_h = target * 2 + pad * 3 + 110  # extra header
    canvas = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_b = font
    draw.text((pad, 4), f"{case_id}", fill=(0, 0, 0), font=font_b)
    draw.text((pad, 26), f"Q: {q[:120]}", fill=(0, 0, 0), font=font)
    draw.text((pad, 48), f"gold: {gold[:120]}", fill=(0, 0, 0), font=font)
    draw.text((pad, 70), f"ROI: {roi_text[:120]}", fill=(0, 80, 0), font=font)
    draw.text((pad, 92), "→ Doctor: does the red bbox cover the region the answer depends on? "
                         "Are masked/only variants medically reasonable?",
              fill=(120, 0, 0), font=font)

    for idx, (p, lab) in enumerate(zip(resized, labels)):
        col = idx % 2
        row = idx // 2
        x = pad + col * (target + pad)
        y = 110 + pad + row * (target + pad)
        # center
        x += (target - p.width) // 2
        y += (target - p.height) // 2
        canvas.paste(p, (x, y))
        draw.text((pad + col * (target + pad), 110 + row * (target + pad) + target),
                  lab, fill=(60, 60, 60), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", type=str, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    grounding = load_grounding()

    if args.case:
        cases = [args.case]
    else:
        cases = [cid for cid, g in grounding.items() if has_roi(g)]
        if not args.all:
            cases = cases[: args.limit]

    print(f"rendering {len(cases)} cases -> {OUT.relative_to(ROOT)}")
    n_ok = 0; n_skip = 0
    for cid in cases:
        m = manifest.get(cid)
        g = grounding.get(cid)
        if not m or not g or not has_roi(g):
            n_skip += 1; continue
        bbox = json.loads(g["roi_bbox_norm"])
        ext = Path(m["image_file"]).suffix.lower()
        orig_p = IMAGES / m["image_file"]
        masked_p = PERT / f"{cid}_roi_masked{ext}"
        only_p = PERT / f"{cid}_roi_only{ext}"
        if not (orig_p.exists() and masked_p.exists() and only_p.exists()):
            n_skip += 1; continue
        try:
            orig = Image.open(orig_p).convert("RGB")
            masked = Image.open(masked_p).convert("RGB")
            only = Image.open(only_p).convert("RGB")
        except Exception as e:
            print(f"  skip {cid}: {e}"); n_skip += 1; continue

        bbox_overlay = panel_with_bbox(orig, bbox, "bbox overlay")
        canvas = composite_panel(
            [orig, bbox_overlay, masked, only],
            ["1) original", "2) bbox overlay (red)", "3) ROI-masked", "4) ROI-only"],
            cid, m["question"], m["gold_answer"], g["roi_pointer"],
        )
        canvas.save(OUT / f"{cid}.png", optimize=True)
        n_ok += 1

    print(f"  rendered {n_ok}, skipped {n_skip}")
    print(f"\nFor doctor review:")
    print(f"  cd {OUT.relative_to(ROOT)} && open MVB-*.png")
    print(f"  flip through; flag any case where the red bbox misses the answer-relevant region.")


if __name__ == "__main__":
    main()
