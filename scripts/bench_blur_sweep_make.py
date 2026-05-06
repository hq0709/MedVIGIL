"""Pre-generate blurred images and per-sigma probe CSVs for the
visual-information-decay ablation.

For each sigma level and each of the 300 original probes, this script:
  1. opens data/medvlm_bench_v1/images/<image_file>
  2. applies a Gaussian blur with the given sigma (in pixels) to the full image
  3. writes the blurred copy to images_blur/sigma<sigma>/<image_file> (same format)
  4. emits a sigma-specific probe CSV at _blur_sweep/probes_sigma<sigma>.csv
     whose `image_file` points at the blurred copy under images_blur/

Sigma=0 (baseline) is materialised by copying the original images path through
in the probes CSV so the runner pipeline stays uniform.

Sigma=inf (no-image) is NOT materialised here; the runner uses --no-image.

Run:
    python3 scripts/bench_blur_sweep_make.py
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/medvlm_bench_v1"
IMAGES = BENCH / "images"
PROBES_MCQ = BENCH / "probes_mcq.csv"
BLUR_ROOT = BENCH / "images_blur"
SWEEP_ROOT = BENCH / "_blur_sweep"

SIGMAS = [0, 2, 4, 8, 16, 32, 64]


def load_original_probes() -> list[dict]:
    rows = []
    with open(PROBES_MCQ, newline="") as f:
        for r in csv.DictReader(f):
            if r["probe_kind"] == "original":
                rows.append(r)
    return rows


def blur_one(src: Path, dst: Path, sigma: int) -> None:
    im = Image.open(src)
    im = im.convert("RGB") if im.mode != "RGB" else im
    if sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        im.save(dst, format="JPEG", quality=92)
    else:
        im.save(dst, format="PNG", optimize=False)


def write_sigma_probe_csv(probes: list[dict], sigma: int) -> Path:
    out = SWEEP_ROOT / f"probes_sigma{sigma}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rel_root = "images" if sigma == 0 else f"images_blur/sigma{sigma}"
    fieldnames = list(probes[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in probes:
            r2 = dict(r)
            r2["image_file"] = f"{rel_root}/{r['image_file']}"
            w.writerow(r2)
    return out


def main() -> None:
    probes = load_original_probes()
    if len(probes) != 300:
        print(f"[warn] expected 300 original probes, got {len(probes)}", file=sys.stderr)

    for sigma in SIGMAS:
        t0 = time.time()
        if sigma > 0:
            for r in probes:
                src = IMAGES / r["image_file"]
                dst = BLUR_ROOT / f"sigma{sigma}" / r["image_file"]
                if dst.exists():
                    continue
                blur_one(src, dst, sigma)
        out_csv = write_sigma_probe_csv(probes, sigma)
        print(f"[sigma={sigma:>3}] {len(probes)} probes  csv={out_csv.relative_to(ROOT)}  took {time.time()-t0:.1f}s", flush=True)

    print(f"\n[done] sigmas processed: {SIGMAS}")
    print(f"[done] blur images under {BLUR_ROOT.relative_to(ROOT)}/")
    print(f"[done] probe CSVs under  {SWEEP_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
