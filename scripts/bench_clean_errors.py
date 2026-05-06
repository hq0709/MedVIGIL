"""Remove error rows from baseline JSONL files so that --resume retries them.

For each file matching results/baselines/<model>__<format>.jsonl:
  - Read all rows
  - Keep only those with status='ok'
  - Atomically rewrite the file
  - Print before/after counts

Usage:
    python3 scripts/bench_clean_errors.py                   # all files
    python3 scripts/bench_clean_errors.py gpt-4o            # one model, all formats
    python3 scripts/bench_clean_errors.py gpt-4o mcq        # one model, one format
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "results/baselines"


def clean(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    total = 0; ok_count = 0; err_count = 0
    ok_rows = []
    for line in open(path):
        if not line.strip():
            continue
        total += 1
        try:
            r = json.loads(line)
            if r.get("status") == "ok":
                ok_count += 1
                ok_rows.append(line if line.endswith("\n") else line + "\n")
            else:
                err_count += 1
        except Exception:
            err_count += 1

    if err_count == 0:
        return total, ok_count, 0

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(ok_rows)
    os.replace(tmp, path)
    return total, ok_count, err_count


def main():
    args = sys.argv[1:]
    if not args:
        files = sorted(BASELINES.glob("*__*.jsonl"))
    elif len(args) == 1:
        files = sorted(BASELINES.glob(f"{args[0]}__*.jsonl"))
    else:
        files = [BASELINES / f"{args[0]}__{args[1]}.jsonl"]

    if not files:
        print("no files matched"); return

    for f in files:
        total, ok, err = clean(f)
        if total == 0:
            print(f"  {f.name:<55} (empty)")
        elif err == 0:
            print(f"  {f.name:<55} clean ({ok}/{total})")
        else:
            print(f"  {f.name:<55} {err} errors removed -> {ok} ok kept")


if __name__ == "__main__":
    main()
