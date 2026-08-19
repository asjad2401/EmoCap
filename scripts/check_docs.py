#!/usr/bin/env python
"""Consistency check across every document in the repo.

Docs drift from the code and from each other, silently. This checks the ways that have
actually happened here and can be checked mechanically:

1. **Self-referential facts going stale** — a doc claiming a test count, a git SHA or a
   corpus size that no longer matches reality.
2. **Cross-references that no longer resolve** — a path or a section that was renamed.
3. **Config paths pointing at files that do not exist** — how `captions_file` came to name
   `captions.txt`, a file this project has never had.
4. **Placeholders without a tracker entry**, or a tracker entry for a placeholder that is
   already gone.

Deliberately NOT checked: whether the same quantity carries the same value everywhere. Tried
and abandoned — several genuinely different numbers look alike (8,076 generated images, 8,091
in the dataset, 8,094 API requests), and a regex cannot tell them apart without more false
positives than findings. Cross-document figures are checked by reading.

    uv run python scripts/check_docs.py
    uv run python scripts/check_docs.py --strict     # non-zero exit on any finding

Local-only working documents (`TASKS.md`, `RESEARCH-LOG.md`, `METHODOLOGY.md`,
`FILE-MAP.md`, `PLACEHOLDERS.md`) are checked when present and skipped when not, so this
runs in a fresh clone.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRACKED_DOCS = [
    "README.md", "configs/data.yaml", "configs/prereg.lock.yaml",
    "docs/preregistration.md", "docs/deviations.md", "docs/lab-notebook.md",
    "docs/data-attribution.md", "docs/prompt-open-items.md",
    "docs/v1-pilot-postmortem.md",
]
LOCAL_DOCS = ["TASKS.md", "RESEARCH-LOG.md", "METHODOLOGY.md", "FILE-MAP.md",
              "PLACEHOLDERS.md"]

def truth() -> dict:
    n_tests = len(re.findall(
        r"^def test_", "\n".join(p.read_text(encoding="utf-8")
                                 for p in (ROOT / "tests").glob("test_*.py")), re.M))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    store = ROOT / "data/generated/captions_raw.jsonl"
    images = cells = 0
    if store.exists():
        recs = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
        images = len({r["image_id"] for r in recs})
        cells = sum(len(r.get("captions") or {}) for r in recs)
    excl = ROOT / "data/generated/excluded_images.json"
    n_excl = len(json.loads(excl.read_text())["excluded"]) if excl.exists() else 0
    return {"tests": n_tests, "sha": sha, "images": images, "cells": cells,
            "exclusions": n_excl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if anything is found")
    args = ap.parse_args()

    t = truth()
    docs = {}
    for d in TRACKED_DOCS + LOCAL_DOCS:
        p = ROOT / d
        if p.exists():
            docs[d] = p.read_text(encoding="utf-8")
    missing_local = [d for d in LOCAL_DOCS if not (ROOT / d).exists()]

    findings: list[str] = []
    print(f"ground truth: {t['tests']} tests | HEAD {t['sha']} | "
          f"{t['images']:,} images | {t['cells']:,} cells | {t['exclusions']} exclusions")
    if missing_local:
        print(f"(local-only docs absent, skipped: {', '.join(missing_local)})")

    # ── 1. stale self-referential facts ─────────────────────────────────────
    print("\n1. stale self-references")
    for d, txt in docs.items():
        for m in re.finditer(r"\*?\*?(\d{2,4})\*?\*? (?:tests )?passing", txt):
            if int(m.group(1)) != t["tests"]:
                findings.append(f"{d}: says {m.group(1)} tests passing, actual {t['tests']}")
        for m in re.finditer(r"latest `([0-9a-f]{7,40})`", txt):
            if not t["sha"].startswith(m.group(1)[:7]):
                findings.append(f"{d}: cites HEAD {m.group(1)}, actual {t['sha']}")
    _report(findings, "clean")

    # ── 3. cross-references that must resolve ───────────────────────────────
    print("\n2. cross-references")
    before = len(findings)
    for d, txt in docs.items():
        for m in re.finditer(r"`((?:docs|src|scripts|configs|tests)/[A-Za-z0-9_./-]+\.(?:py|md|yaml))`", txt):
            if not (ROOT / m.group(1)).exists():
                findings.append(f"{d} -> {m.group(1)} does not exist")
        for m in re.finditer(r"`(TASKS|RESEARCH-LOG|METHODOLOGY|FILE-MAP|PLACEHOLDERS)\.md`", txt):
            if not (ROOT / f"{m.group(1)}.md").exists():
                findings.append(f"{d} -> {m.group(1)}.md does not exist")
    # RESEARCH-LOG part references
    rl = docs.get("RESEARCH-LOG.md", "")
    parts = set(re.findall(r"^# PART (\d+)", rl, re.M))
    if parts:
        for d, txt in docs.items():
            for m in re.finditer(r"RESEARCH-LOG\.md.{0,14}?Part (\d+)", txt):
                if m.group(1) not in parts:
                    findings.append(f"{d}: cites RESEARCH-LOG Part {m.group(1)}, which does not exist")
    _report(findings[before:], "all resolve")

    # ── 4. config paths ─────────────────────────────────────────────────────
    print("\n3. config paths")
    before = len(findings)
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    for k, v in (cfg.get("paths") or {}).items():
        exists = (ROOT / v).exists()
        planned = "does not exist yet" in (cfg_text := (ROOT / "configs/data.yaml").read_text()) and \
                  re.search(rf"{re.escape(k)}:.*does not exist yet", cfg_text)
        if exists:
            print(f"   {k:<18} OK")
        elif planned:
            print(f"   {k:<18} absent, ANNOTATED as a planned output")
        else:
            findings.append(f"configs/data.yaml: paths.{k} = {v!r} does not exist and is "
                            f"not annotated as a planned output")
    _report(findings[before:], "all present or annotated")

    # ── 5. placeholders have tracker entries ────────────────────────────────
    if "PLACEHOLDERS.md" in docs:
        print("\n4. placeholders")
        before = len(findings)
        used = set()
        for d in TRACKED_DOCS:
            used |= set(re.findall(r"PH-\d\d", docs.get(d, "")))
        tracked = set(re.findall(r"^## (PH-\d\d)", docs["PLACEHOLDERS.md"], re.M))
        for ph in sorted(used - tracked):
            findings.append(f"{ph} appears in a tracked doc but has no PLACEHOLDERS.md entry")
        cleared = set(re.findall(r"^## (PH-\d\d).*CLEARED", docs["PLACEHOLDERS.md"], re.M))
        for ph in sorted(tracked - used - cleared):
            findings.append(f"{ph} has a tracker entry but appears in no tracked doc "
                            f"(already cleared? mark it CLEARED or remove it)")
        if cleared:
            print(f"   {len(cleared)} cleared: {', '.join(sorted(cleared))}")
        if not (used - tracked) and not (tracked - used):
            print(f"   {len(used)} placeholders, all tracked both ways")
        _report(findings[before:], "consistent")

    print(f"\n{'=' * 60}")
    print(f"{len(findings)} finding(s)" if findings else "no findings — documents consistent")
    if findings and args.strict:
        sys.exit(1)


def _report(new: list[str], ok: str) -> None:
    if new:
        for f in new:
            print(f"   FINDING: {f}")
    else:
        print(f"   {ok}")


if __name__ == "__main__":
    main()
