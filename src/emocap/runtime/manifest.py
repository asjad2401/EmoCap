"""Run manifests.

Nothing is quotable without one. A folder of checkpoints is not a research
record; a folder of checkpoints each sitting next to the git SHA, config hash,
seed, package versions and wall time that produced it is.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from emocap.runtime.config import REPO_ROOT, config_hash

__all__ = ["Manifest", "git_sha", "package_versions"]

_TRACKED_PACKAGES = ("torch", "transformers", "numpy", "pandas", "peft", "scikit-learn")


def git_sha(*, short: bool = False) -> str:
    """Current commit, with a ``-dirty`` suffix if the tree has changes."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        sha = subprocess.check_output(
            args, stderr=subprocess.DEVNULL, text=True, cwd=str(REPO_ROOT)
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
            cwd=str(REPO_ROOT),
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _accelerator() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "n_devices": 0}
    if not torch.cuda.is_available():
        return {"device": "cpu", "n_devices": 0}
    return {
        "device": "cuda",
        "n_devices": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


@dataclass
class Manifest:
    """Provenance for one run. Write it *before* training, finalise it after.

    Written before, so an interrupted run still leaves a record of what it was
    attempting -- the pilot's abandoned runs left nothing at all.
    """

    run_id: str
    stage: str
    config: dict[str, Any]
    seed: int | None = None
    notes: str = ""

    git_sha: str = field(default_factory=git_sha)
    packages: dict[str, str] = field(default_factory=package_versions)
    accelerator: dict[str, Any] = field(default_factory=_accelerator)
    python: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)

    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    @property
    def wall_seconds(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.started_at

    def record_check(self, name: str, passed: bool, detail: Any = None) -> None:
        """Log a gate result. See docs/preregistration.md section 6."""
        self.checks[name] = {"passed": bool(passed), "detail": detail}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config_hash"] = self.config_hash
        d["wall_seconds"] = self.wall_seconds
        return d

    def save(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    def finalise(self, run_dir: str | Path, status: str = "complete") -> Path:
        self.finished_at = time.time()
        self.status = status
        # Defensive: finalise runs *after* the work is paid for, so a malformed
        # check entry must not lose the manifest. A caller once stored a raw stats
        # dict here and the KeyError discarded the record of a completed batch run.
        failed = [
            k for k, v in self.checks.items()
            if isinstance(v, dict) and v.get("passed") is False
        ]
        if failed and status == "complete":
            self.status = "complete_with_failed_checks"
        return self.save(run_dir)

    @classmethod
    def load(cls, run_dir: str | Path) -> dict[str, Any]:
        return json.loads((Path(run_dir) / "manifest.json").read_text())
