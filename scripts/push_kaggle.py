#!/usr/bin/env python
"""Ship the arms, the CLIP features and the frozen classifier to Kaggle.

    uv run python scripts/push_kaggle.py            # create or update, PRIVATE
    uv run python scripts/push_kaggle.py --dry-run  # stage and report, upload nothing

The GPU runs need exactly three things, and none of them are images or API keys:

    arms/        the data arms + manifest.json (per-arm sha256)
    features/    pre-extracted frozen CLIP ViT-B/32 + index
    predictions/ the arms' GENERATED captions, references stripped (see
                 scripts/build_prediction_export.py) -- for GPU-side analysis of model
                 output, and the only part containing no third-party corpus text
    classifier/  the frozen register classifier + its sha256

**Private by default, and there is no flag to make it public.** The arms are derived from
Flickr8k and Personality-Captions, whose terms govern redistribution; publishing them from a
convenience script is not a decision this file should be able to make.

A `provenance.json` is written alongside, recording the git commit, the prereg tag the
working tree is at, and every artifact hash. The training notebook asserts against it, so a
run can never quietly use a stale upload -- which is the failure this project has already
had once, when the V1 arms were built from the wrong corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# TWO datasets, not one, and the split is not cosmetic.
#
# The first attempt pushed all 421 MB as a single dataset. It transferred for 54 minutes,
# sent 429 MB, then hung with no socket and no dataset created -- the whole hour bought
# nothing, because Kaggle's create call is atomic and a 317 MB zip is one failure point.
#
# Splitting matches how the files are actually used: `arms` and `features` are needed at
# TRAIN time and are 86 MB together, so they land in minutes; the classifier is needed only
# at SCORE time, which happens after training. Training can therefore start while the slow
# half is still uploading, and a failure in either does not block the other.
PARTS = {
    "data": {
        "slug": "emocap-v2-arms",
        "title": "EmoCap v2 — data arms and CLIP features",
        # captions_corpus.jsonl (33 MB) is the SOURCE captions, needed by
        # baseline_prior.py --mode text and by nothing else on the GPU side. It was left
        # out of the first uploads, which failed all fifteen text baselines instantly.
        "sources": {"arms": ROOT / "data/arms", "features": ROOT / "data/features",
                    "generated": ROOT / "data/generated/captions_corpus.jsonl"},
    },
    # Generated captions with the reference text stripped, built by
    # scripts/build_prediction_export.py. Needed by analyses that read only what the models
    # WROTE -- P4's judges, and anything like them -- so those can run on a GPU instead of
    # wedging a laptop overnight. Separate from `data` because no training notebook needs
    # it, and separate from `classifier` because it changes whenever an arm is re-decoded.
    "predictions": {
        "slug": "emocap-v2-predictions",
        "title": "EmoCap v2 — generated captions (references removed)",
        "sources": {"predictions": ROOT / "runs/kaggle-preds"},
    },
    "classifier": {
        "slug": "emocap-v2-classifier",
        "title": "EmoCap v2 — frozen register classifier",
        "sources": {"classifier": ROOT / "models/register_classifier"},
    },
}


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_part(part: str, owner: str) -> tuple[Path, dict, int]:
    """Copy one part's sources into its own staging dir with provenance. Returns stats."""
    spec = PARTS[part]
    missing = [k for k, p in spec["sources"].items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing} -- run build_arms / extract_features / "
                         f"train_classifier first")

    stage = ROOT / "runs/kaggle-stage" / part
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    files: dict[str, str] = {}
    total = 0
    for name, src in spec["sources"].items():
        dst = stage / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            # A single file is staged INTO a directory of that name, so the layout the
            # notebooks read (<part>/generated/captions_corpus.jsonl) matches the repo's.
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        for f in sorted(dst.rglob("*")):
            if f.is_file():
                files[str(f.relative_to(stage))] = sha256(f)
                total += f.stat().st_size

    arms_dir = PARTS["data"]["sources"]["arms"]
    clf_dir = PARTS["classifier"]["sources"]["classifier"]
    provenance = {
        "part": part,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "prereg_tag": git("describe", "--tags", "--abbrev=0"),
        "tree_clean": git("status", "--porcelain") == "",
        # Both parts carry BOTH identities, so a notebook can assert that the arms it
        # trained on and the classifier that scores them came from one commit. Pairing a
        # dataset version with the wrong counterpart is the silent failure this guards.
        "arm_manifest": json.loads((arms_dir / "manifest.json").read_text()),
        "classifier_sha256": (clf_dir / "sha256.txt").read_text().strip()
        if (clf_dir / "sha256.txt").exists() else None,
        "files": files,
    }
    (stage / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": spec["title"],
        "id": f"{owner}/{spec['slug']}",
        "licenses": [{"name": "other"}],
    }, indent=2))
    return stage, provenance, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=[*PARTS, "all"], default="all",
                    help="'data' is 86 MB and needed to train; 'classifier' is 317 MB "
                         "and needed only to score")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default=None, help="version note")
    args = ap.parse_args()

    owner = os.environ.get("KAGGLE_USERNAME") or _username()
    parts = list(PARTS) if args.part == "all" else [args.part]

    for part in parts:
        stage, prov, total = stage_part(part, owner)
        slug = PARTS[part]["slug"]
        print(f"\n=== {part}: {len(prov['files'])} files, {total/1e6:.0f} MB -> {slug}")
        print(f"  commit {prov['git_commit'][:8]}  tag {prov['prereg_tag']}  "
              f"clean {prov['tree_clean']}")
        if not prov["tree_clean"]:
            print("  WARNING: dirty tree -- the upload will match no commit")
        if args.dry_run:
            print("  (dry run -- nothing uploaded)")
            continue

        api = _api()
        # `dataset_list` does not reliably return PRIVATE datasets, so it once reported a
        # slug as absent, took the create path, and left the live dataset on its old
        # version while printing success. Existence is probed directly instead.
        try:
            api.dataset_status(f"{owner}/{slug}")
            exists = True
        except Exception:  # noqa: BLE001
            exists = False
        if exists:
            print(f"  updating {owner}/{slug} (new version)...")
            api.dataset_create_version(
                str(stage), version_notes=args.message or prov["git_commit"][:8],
                dir_mode="zip", quiet=False)
        else:
            print(f"  creating {owner}/{slug} (PRIVATE)...")
            api.dataset_create_new(str(stage), public=False, dir_mode="zip", quiet=False)
        print(f"  done: https://www.kaggle.com/datasets/{owner}/{slug}")


def _api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _username() -> str:
    return _api().config_values.get("username", "unknown")


if __name__ == "__main__":
    main()
