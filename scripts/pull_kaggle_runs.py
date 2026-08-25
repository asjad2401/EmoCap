#!/usr/bin/env python
"""Download a training notebook's output and score it here, against the frozen instrument.

    uv run python scripts/pull_kaggle_runs.py --kernel muhammadasjadali/emocap-train-arm
    uv run python scripts/pull_kaggle_runs.py --kernel <slug> --no-score

**Why scoring is local and the classifier never goes to Kaggle.**

The frozen classifier is a 332 MB safetensors file. Two attempts to upload it stalled at
84% and 97% with the process alive, no socket, and no dataset created -- Kaggle's create
call is atomic, so both attempts bought nothing after 54 and 52 minutes.

It does not need to make the trip. Predictions are 278 bytes per cell, so the *entire*
36-run sweep is ~97 MB moving in the cheap direction, and scoring is inference over short
strings -- a couple of minutes on this laptop. Keeping the instrument in one place is also
better provenance than copying it: there is exactly one classifier, and every score in the
study came from it.

Kaggle therefore does what only a GPU can do (train, beam-decode) and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.runtime import rel  # noqa: E402


def api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    from kaggle.api.kaggle_api_extended import KaggleApi

    a = KaggleApi()
    a.authenticate()
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True,
                    help="owner/kernel-slug of the training notebook")
    ap.add_argument("--out", default="runs/arms")
    ap.add_argument("--no-score", action="store_true",
                    help="download only; score later with scripts/score_arm.py")
    ap.add_argument("--classifier", default="models/register_classifier")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "runs/kaggle-pull"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"downloading output of {args.kernel} ...")
    api().kernels_output(args.kernel, path=str(tmp), force=True, quiet=False)
    for z in tmp.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)

    # A run directory is anything holding predictions.jsonl. Kaggle nests output under
    # whatever path the notebook wrote, so the tree is searched rather than assumed.
    found = sorted({p.parent for p in tmp.rglob("predictions.jsonl")})
    if not found:
        raise SystemExit(f"no predictions.jsonl anywhere under {tmp}")

    pulled: list[Path] = []
    for src in found:
        dst = out / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n = sum(1 for _ in (dst / "predictions.jsonl").open())
        print(f"  {dst.name:<22} {n:>7,} predictions -> {rel(dst)}")
        pulled.append(dst)
    shutil.rmtree(tmp)

    if args.no_score:
        print("\n(--no-score: nothing scored)")
        return

    print()
    for run in pulled:
        subprocess.run([sys.executable, str(ROOT / "scripts/score_arm.py"),
                        "--run", str(run), "--classifier", args.classifier], check=False)


if __name__ == "__main__":
    main()
