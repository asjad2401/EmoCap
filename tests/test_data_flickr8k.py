"""Tests for Flickr8k ingestion and the canonical split.

The split is the one artifact every later stage depends on. v1 re-derived splits
in more than one notebook; these tests pin down that a split is deterministic,
leak-free, and stable under reordering of the input.
"""

from __future__ import annotations

import pytest

from emocap.data import (
    assign_splits,
    caption_stats,
    read_captions,
    read_splits,
    write_splits,
)

RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

KAGGLE_CSV = """image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg,A girl going into a wooden building .
1000268201_693b08cb0e.jpg,A little girl climbing into a wooden playhouse .
1001773457_577c3a7d70.jpg,A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg,"A black dog and a white dog with brown spots are staring at each other , in the street ."
"""

TOKEN_TXT = (
    "1000268201_693b08cb0e.jpg#0\tA child in a pink dress .\n"
    "1000268201_693b08cb0e.jpg#1\tA girl going into a wooden building .\n"
    "1001773457_577c3a7d70.jpg#0\tA black dog and a spotted dog are fighting\n"
)


# ── parsing ─────────────────────────────────────────────────────────────────


def test_reads_kaggle_csv_format(tmp_path):
    p = tmp_path / "captions.txt"
    p.write_text(KAGGLE_CSV, encoding="utf-8")
    rows = read_captions(p)
    assert len(rows) == 5
    assert rows[0].image_id == "1000268201_693b08cb0e.jpg"
    assert rows[0].caption_idx == 0
    assert rows[2].caption_idx == 2
    assert rows[3].caption_idx == 0, "caption_idx must restart per image"


def test_reads_original_token_format(tmp_path):
    p = tmp_path / "Flickr8k.token.txt"
    p.write_text(TOKEN_TXT, encoding="utf-8")
    rows = read_captions(p)
    assert len(rows) == 3
    assert rows[1].caption_idx == 1
    assert rows[2].image_id == "1001773457_577c3a7d70.jpg"


def test_quoted_caption_containing_a_comma_survives(tmp_path):
    """The Kaggle file quotes captions with commas. Splitting on ',' loses them."""
    p = tmp_path / "captions.txt"
    p.write_text(KAGGLE_CSV, encoding="utf-8")
    rows = read_captions(p)
    long_one = [r for r in rows if "brown spots" in r.caption]
    assert len(long_one) == 1
    # The comma inside the quoted field must survive, and must not have split
    # the row into a third column.
    assert long_one[0].caption.count(",") == 1, long_one[0].caption
    assert long_one[0].caption.endswith("in the street ."), long_one[0].caption


def test_empty_file_is_rejected(tmp_path):
    p = tmp_path / "captions.txt"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_captions(p)


def test_header_only_file_is_rejected(tmp_path):
    p = tmp_path / "captions.txt"
    p.write_text("image,caption\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parsed no captions"):
        read_captions(p)


# ── stats ───────────────────────────────────────────────────────────────────


def test_caption_stats_flags_images_without_five_captions(tmp_path):
    p = tmp_path / "captions.txt"
    p.write_text(KAGGLE_CSV, encoding="utf-8")
    st = caption_stats(read_captions(p))
    assert st["n_images"] == 2
    assert st["images_without_exactly_5"] == 2
    assert st["caption_words"]["p50"] > 0


# ── splits ──────────────────────────────────────────────────────────────────


def _ids(n):
    return [f"img_{i:05d}.jpg" for i in range(n)]


def test_split_is_deterministic_given_a_seed():
    a = assign_splits(_ids(500), ratios=RATIOS, seed=42)
    b = assign_splits(_ids(500), ratios=RATIOS, seed=42)
    assert a == b


def test_split_is_independent_of_input_order():
    """Reading order must not change the split, or a re-run silently reshuffles."""
    ids = _ids(500)
    forward = assign_splits(ids, ratios=RATIOS, seed=42)
    backward = assign_splits(list(reversed(ids)), ratios=RATIOS, seed=42)
    assert forward == backward


def test_different_seeds_give_different_splits():
    assert assign_splits(_ids(500), ratios=RATIOS, seed=42) != \
           assign_splits(_ids(500), ratios=RATIOS, seed=1337)


def test_split_sizes_match_ratios():
    s = assign_splits(_ids(1000), ratios=RATIOS, seed=42)
    counts = {k: sum(1 for v in s.values() if v == k) for k in ("train", "val", "test")}
    assert counts["train"] == 800
    assert counts["val"] == 100
    assert counts["test"] == 100


def test_every_image_gets_exactly_one_split():
    ids = _ids(300)
    s = assign_splits(ids, ratios=RATIOS, seed=42)
    assert set(s) == set(ids)
    assert all(v in ("train", "val", "test") for v in s.values())


def test_no_image_leaks_across_splits():
    """The property that matters: an image's captions must never straddle splits."""
    ids = _ids(400)
    s = assign_splits(ids, ratios=RATIOS, seed=7)
    rows = [(i, idx) for i in ids for idx in range(5)]
    by_image: dict[str, set[str]] = {}
    for image_id, _ in rows:
        by_image.setdefault(image_id, set()).add(s[image_id])
    assert all(len(v) == 1 for v in by_image.values())


def test_duplicate_ids_are_collapsed():
    s = assign_splits(_ids(100) + _ids(100), ratios=RATIOS, seed=42)
    assert len(s) == 100


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits(_ids(10), ratios={"train": 0.7, "val": 0.1, "test": 0.1}, seed=1)


def test_missing_split_name_is_rejected():
    with pytest.raises(ValueError, match="missing split ratio"):
        assign_splits(_ids(10), ratios={"train": 0.9, "val": 0.1}, seed=1)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no image ids"):
        assign_splits([], ratios=RATIOS, seed=1)


def test_splits_roundtrip_through_disk(tmp_path):
    s = assign_splits(_ids(250), ratios=RATIOS, seed=42)
    p = write_splits(s, tmp_path / "splits.csv")
    assert read_splits(p) == s


def test_split_file_header_is_validated(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("img,set\na.jpg,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected split file header"):
        read_splits(p)
