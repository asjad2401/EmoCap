"""Images the generator will not process, and why.

Three test-split images returned zero output tokens on every attempt. All three show
young children in or near water, minimally clothed; the model's child-safety filtering
declines them. That is the filter working correctly and it is **not** to be worked
around -- not by re-encoding the image, not by rewording the prompt.

Two things follow, and both need recording rather than silent handling:

1. **Operationally.** Without a register, such an image is never "done", so it is
   retried on every subsequent run. Worse, `run_stage02.py` preflights on the first
   pending image, so one refused image aborts an otherwise healthy run before it
   submits anything.

2. **For the study.** The exclusion is NOT random -- it correlates with subject matter
   (children, water, swimwear). At ~0.3% of the test split that is roughly 20-25 images
   across the corpus. Small, but a systematic gap tied to content is a limitation to
   disclose, not attrition to average away. See docs/deviations.md.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

__all__ = ["load_exclusions", "record_exclusion", "DEFAULT_PATH"]

DEFAULT_PATH = "data/generated/excluded_images.json"


def load_exclusions(path: str | Path = DEFAULT_PATH) -> dict[str, dict]:
    """image_id -> {"reason": ..., "detail": ..., "recorded": "YYYY-MM-DD"}."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data.get("excluded", {}) if isinstance(data, dict) else {}


def record_exclusion(
    image_id: str, reason: str, detail: str = "", path: str | Path = DEFAULT_PATH
) -> dict[str, dict]:
    """Add an image to the register, preserving any entry already there."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    current = load_exclusions(p)
    current.setdefault(image_id, {
        "reason": reason,
        "detail": detail,
        "recorded": date.today().isoformat(),
    })
    p.write_text(json.dumps({
        "note": "Images the generator will not process. See "
                "src/emocap/data/exclusions.py for why these are recorded rather "
                "than retried, and docs/deviations.md for the study implications.",
        "excluded": current,
    }, indent=2), encoding="utf-8")
    return current
