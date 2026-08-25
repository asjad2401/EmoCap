"""Config loading, and the pre-registration lock check.

One rule: **no number lives in prose.** v1's notebook headers advertised filters
(quality >= 0.5, length-delta <= 12, overlap >= 0.25) that the code below had
overridden to quality-disabled, delta <= 35, overlap >= 0.03. Documentation and
behaviour had silently diverged and nobody noticed for months.

So thresholds live here, are read once, are echoed into every run manifest, and
are checked against the frozen pre-registration before results are computed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "REPO_ROOT",
    "CONFIG_DIR",
    "LOCK_PATH",
    "load_config",
    "config_hash",
    "flatten",
    "lock_violations",
    "assert_matches_lock",
]

def _find_repo_root() -> Path:
    """Walk up for the directory holding `configs/`, so this works whether the
    package is installed editable, installed as a wheel, or run from a notebook
    with an arbitrary working directory."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "configs" / "prereg.lock.yaml").exists():
            return candidate
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "configs" / "prereg.lock.yaml").exists():
            return candidate
    return here.parents[3]  # best effort: src/emocap/runtime/config.py -> repo


REPO_ROOT = _find_repo_root()
CONFIG_DIR = REPO_ROOT / "configs"
LOCK_PATH = CONFIG_DIR / "prereg.lock.yaml"

_YAML_SUFFIXES = (".yaml", ".yml")


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by bare name (``"data"``) or explicit path."""
    import yaml

    p = Path(name_or_path)
    if p.suffix not in _YAML_SUFFIXES:
        # Bare name: "data" -> configs/data.yaml, "prereg.lock" -> configs/prereg.lock.yaml
        p = CONFIG_DIR / f"{p.name}.yaml"
    elif not p.is_absolute():
        candidate = CONFIG_DIR / p
        p = candidate if candidate.exists() else p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{p} must contain a mapping at the top level")
    return loaded


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable sha256 over a config, for the run manifest."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def flatten(cfg: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``("a.b.c", leaf)`` for every leaf. Lists are compared as wholes."""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, cfg


def lock_violations(
    live: dict[str, Any],
    lock: dict[str, Any] | None = None,
    *,
    require_sections: Sequence[str] = (),
) -> list[str]:
    """Every pre-registered key that ``live`` contradicts, or silently drops.

    Keys absent from ``live`` are normally fine -- a config need not restate the
    whole study. Keys present in both must agree exactly.

    ``require_sections`` names top-level sections (``"data"``, ``"decode"``) that
    ``live`` claims to own in full: within those, a locked key that is *missing*
    from ``live`` is also a violation.

    That second check exists because of a real miss: an edit adopting the official
    Flickr8k splits landed in ``configs/data.yaml`` but not in the lock, leaving
    ``split_source: official_flickr8k`` in one file and ``split_ratios: 80/10/10``
    in the other. Contradiction-only checking passed both, because neither key
    appeared in both files. A missing pre-registered setting is exactly as
    dangerous as a contradicting one.
    """
    if lock is None:
        lock = load_config(LOCK_PATH)
    live_flat = dict(flatten(live))
    out: list[str] = []
    for key, locked in flatten(lock):
        if key in live_flat:
            if live_flat[key] != locked:
                out.append(f"{key}: live={live_flat[key]!r} locked={locked!r}")
        elif any(key.startswith(f"{sec}.") for sec in require_sections):
            out.append(f"{key}: MISSING from live config (locked={locked!r})")
    return out


def assert_matches_lock(
    live: dict[str, Any],
    lock: dict[str, Any] | None = None,
    *,
    require_sections: Sequence[str] = (),
) -> None:
    """Raise if a live config has drifted from the pre-registration.

    Stage 10 calls this before computing any metric. Drifting is allowed -- but it
    has to be deliberate: log it in docs/deviations.md and bump the lock tag.
    """
    bad = lock_violations(live, lock, require_sections=require_sections)
    if bad:
        raise ValueError(
            "live config contradicts configs/prereg.lock.yaml:\n  "
            + "\n  ".join(bad)
            + "\n\nIf this change is intended, record it in docs/deviations.md and "
            "bump the lock to prereg-v2. Do not silently edit the lock."
        )


def rel(path) -> "Path":
    """``path`` relative to the repository, or unchanged when it lies outside it.

    Only ever used to make a printed path shorter. ``Path.relative_to`` RAISES when the
    path is outside the root, and every script here has an ``--out``/``--out-root`` that can
    point anywhere -- on Kaggle they always do, since outputs go to ``/kaggle/working`` while
    the clone sits in ``/kaggle/working/EmoCap``.

    That turned into three separate incidents. Twenty complete baseline runs were reported
    as failures by a print statement; a P4 run threw away two trained judges; an artifact
    ablation threw away twenty minutes of cross-validation it had already written to disk.
    Each time the work was finished and the crash was in the line that formats the path
    for a log message. Fixing it twice as one-offs is why it happened a third time.
    """
    from pathlib import Path

    p = Path(path)
    return p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
