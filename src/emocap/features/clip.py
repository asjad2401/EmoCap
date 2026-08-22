"""Pre-extract frozen CLIP image features, once, for every arm.

The vision tower is frozen (preregistration §2), so running it inside the training loop
would recompute identical vectors 36 times over. Extracting once to a memmapped array
turns 36 GPU runs into 36 array lookups and -- more importantly -- makes the features a
*fixed artifact* that every arm demonstrably shares. "Held constant by config, not by
discipline" is only true if the constant is on disk.

Two image universes, deliberately kept in one file
--------------------------------------------------
Flickr8k (the S and V1 arms) and YFCC (the human arm) are different photograph
distributions -- that difference is exactly why the human comparisons are *reference*
rather than confirmatory. But they go through the identical encoder and land in the same
array, so nothing about the feature pipeline can be blamed for the gap between them.

Storage
-------
``float32 [N, 512]`` in a ``.npy``, plus a JSON index mapping ``image_id -> row``. Not a
dict of arrays: a memmap lets a Kaggle notebook read one batch's rows without loading
25 MB, and the index is what survives an arm being rebuilt with different membership.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

__all__ = ["extract_features", "load_features", "FeatureStore"]


class FeatureStore:
    """Read-side: ``store[image_id] -> np.ndarray[512]``, backed by a memmap."""

    def __init__(self, array_path: Path, index_path: Path):
        import numpy as np

        self.index: dict[str, int] = json.loads(Path(index_path).read_text())
        self.array = np.load(array_path, mmap_mode="r")
        if len(self.index) != self.array.shape[0]:
            raise ValueError(f"index has {len(self.index)} ids but array has "
                             f"{self.array.shape[0]} rows -- rebuild the features")

    def __contains__(self, image_id: str) -> bool:
        return image_id in self.index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, image_id: str):
        return self.array[self.index[image_id]]

    def rows(self, image_ids: Sequence[str]):
        import numpy as np

        return np.stack([self.array[self.index[i]] for i in image_ids])


def load_features(root: Path) -> FeatureStore:
    return FeatureStore(root / "clip_vit_b32.npy", root / "clip_vit_b32_index.json")


def extract_features(
    image_paths: dict[str, Path],
    *,
    weights: Path,
    out_dir: Path,
    batch_size: int = 64,
    device: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Encode every image once and write the array + index. Returns counts.

    Missing or unreadable images are **recorded, not skipped silently** -- an arm whose
    images half-vanished would otherwise train on whatever survived and report a clean
    number.
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    proc = CLIPImageProcessor.from_pretrained(str(weights))
    model = CLIPVisionModelWithProjection.from_pretrained(str(weights)).to(device).eval()

    ids = sorted(image_paths)
    dim = int(model.config.projection_dim)
    out = np.zeros((len(ids), dim), dtype=np.float32)
    index: dict[str, int] = {}
    failed: list[str] = []
    row = 0
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        imgs, keep = [], []
        for i in chunk:
            try:
                imgs.append(Image.open(image_paths[i]).convert("RGB"))
                keep.append(i)
            except Exception:  # noqa: BLE001
                failed.append(i)
        if imgs:
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                emb = model(pixel_values=px).image_embeds
            emb = emb.float().cpu().numpy()
            for i, v in zip(keep, emb):
                out[row] = v
                index[i] = row
                row += 1
        if progress:
            progress(min(start + batch_size, len(ids)), len(ids))

    out = out[:row]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "clip_vit_b32.npy", out)
    (out_dir / "clip_vit_b32_index.json").write_text(json.dumps(index))
    stats = {"images": row, "dim": dim, "failed": failed, "device": device,
             "weights": str(weights)}
    (out_dir / "clip_vit_b32_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
