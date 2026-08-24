"""The post-hoc arm is only interpretable if it is nested inside the registered ones.

``S_paired_matched`` exists to isolate paired structure from data volume. That argument
collapses the moment its cells stop being a subset of cells the registered arms already
trained on, because "the post-hoc arm drew easier data" becomes an untestable alternative
explanation for whatever it shows. ``scripts/build_posthoc_arms.py`` asserts the three
containments at build time; these tests assert them against ``build_all_arms`` itself, so a
future edit to either selection rule fails here rather than in a paper.

The corpus is synthetic and tiny. What is under test is the selection arithmetic -- which
images, which source caption, which registers -- and that is independent of the text.
"""

from __future__ import annotations

import pytest

from emocap.data.arms import build_all_arms, pick_caption_idx, shared_images
from emocap.data.posthoc import build_s_paired_matched
from emocap.data.prompt import EMOTIONS

N_IMAGES = 60
PER_REGISTER = 6  # 6 images x 5 registers = 30 cells, the same size as the 1/image arms


def _image_ids(n: int) -> list[str]:
    return [f"img{i:04d}.jpg" for i in range(n)]


@pytest.fixture
def corpus_and_pilot(tmp_path):
    """A corpus file with five source captions per image, and a matching v1 pilot."""
    import json

    lines = []
    for img in _image_ids(N_IMAGES):
        for k in range(5):
            lines.append(json.dumps({
                "image_id": img, "caption_idx": k,
                "source_caption": f"{img} source {k}",
                "captions": {e: f"{img} k{k} {e}" for e in EMOTIONS}}))
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(lines) + "\n")
    pilot = {img: {e: f"{img} v1 {e}" for e in EMOTIONS} for img in _image_ids(N_IMAGES)}
    return path, pilot


@pytest.fixture
def arms(corpus_and_pilot):
    path, pilot = corpus_and_pilot
    return build_all_arms(corpus_path=path, v1_captions=pilot, human_cells=[],
                          per_register=PER_REGISTER)


@pytest.fixture
def matched(corpus_and_pilot):
    from emocap.data.arms import load_corpus

    path, pilot = corpus_and_pilot
    return build_s_paired_matched(corpus=load_corpus(path), v1_captions=pilot,
                                  per_register=PER_REGISTER)


def key(c):
    return (c["image_id"], c["caption_idx"], c["emotion"])


def test_matches_the_unpaired_arm_on_cell_count(arms, matched):
    """The whole point: same number of cells, so volume cannot explain a difference."""
    assert len(matched) == len(arms["S_unpaired"]) == PER_REGISTER * len(EMOTIONS)


def test_registers_are_exactly_balanced(matched):
    counts = {e: sum(1 for c in matched if c["emotion"] == e) for e in EMOTIONS}
    assert set(counts.values()) == {PER_REGISTER}


def test_images_are_a_subset_of_the_unpaired_arms(arms, matched):
    mine = {c["image_id"] for c in matched}
    assert mine <= {c["image_id"] for c in arms["S_unpaired"]}
    # Trading images for registers is what pairing means here, so far FEWER images.
    assert len(mine) == PER_REGISTER


def test_cells_are_a_subset_of_the_paired5_arm(arms, matched):
    assert {key(c) for c in matched} <= {key(c) for c in arms["S_paired5"]}


def test_every_unpaired_cell_on_a_shared_image_reappears(arms, matched):
    """Nesting has to hold in this direction too, or the arms are merely overlapping."""
    imgs = {c["image_id"] for c in matched}
    mine = {key(c) for c in matched}
    shared = [c for c in arms["S_unpaired"] if c["image_id"] in imgs]
    assert shared, "the fixture is too small to share any image"
    assert all(key(c) in mine for c in shared)


def test_uses_the_same_source_caption_as_the_paired_arms(matched):
    """A different caption_idx would vary the source text as well as the structure."""
    for c in matched:
        assert c["caption_idx"] == pick_caption_idx(c["image_id"])


def test_selection_is_deterministic_and_order_independent(corpus_and_pilot, matched):
    """Rebuilt from a shuffled pilot dict, the arm must be byte-identical.

    Everything is chosen by hashing the image id precisely so that iteration order cannot
    leak in. This is the property that makes a half-finished study safe to resume.
    """
    import random

    from emocap.data.arms import load_corpus

    path, pilot = corpus_and_pilot
    items = list(pilot.items())
    random.Random(7).shuffle(items)
    again = build_s_paired_matched(corpus=load_corpus(path), v1_captions=dict(items),
                                  per_register=PER_REGISTER)
    assert again == matched


def test_a_dropped_image_does_not_move_the_others(corpus_and_pilot, matched):
    """Excluding an image must not reshuffle the arm around it.

    ``_h``-based ranking means the selected set only loses the excluded image and gains the
    next one down the ranking; the survivors keep their source caption and fold.
    """
    from emocap.data.arms import load_corpus

    path, pilot = corpus_and_pilot
    dropped = matched[0]["image_id"]
    trimmed = build_s_paired_matched(corpus=load_corpus(path), v1_captions=pilot,
                                     excluded=frozenset({dropped}),
                                     per_register=PER_REGISTER)
    assert dropped not in {c["image_id"] for c in trimmed}
    survivors = {key(c) for c in matched if c["image_id"] != dropped}
    assert survivors <= {key(c) for c in trimmed}


def test_shared_images_requires_both_corpora_to_be_complete(corpus_and_pilot):
    """An image the pilot cannot supply in full is not eligible for any Flickr8k arm."""
    from emocap.data.arms import load_corpus

    path, pilot = corpus_and_pilot
    corpus = load_corpus(path)
    crippled = dict(pilot)
    victim = _image_ids(N_IMAGES)[0]
    crippled[victim] = {e: "" for e in EMOTIONS}
    assert victim in shared_images(corpus, pilot)
    assert victim not in shared_images(corpus, crippled)
