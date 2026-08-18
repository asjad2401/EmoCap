"""Tests for config loading, the pre-registration lock, and run manifests.

The lock check is the mechanism that stops v1's failure mode where the documented
pipeline and the running pipeline silently diverged. It is worth testing carefully.
"""

from __future__ import annotations

import json

import pytest

from emocap.runtime import (
    LOCK_PATH,
    Manifest,
    assert_matches_lock,
    config_hash,
    load_config,
    lock_violations,
    seed_everything,
)
from emocap.runtime.config import flatten


# ── config loading ──────────────────────────────────────────────────────────


def test_loads_by_bare_name():
    assert "study" in load_config("prereg.lock")
    assert "data" in load_config("data")


def test_loads_by_explicit_path():
    assert load_config(LOCK_PATH) == load_config("prereg.lock")


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError, match="config not found"):
        load_config("no_such_config")


def test_non_mapping_config_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping at the top level"):
        load_config(p)


# ── hashing ─────────────────────────────────────────────────────────────────


def test_config_hash_is_stable_and_key_order_independent():
    a = config_hash({"x": 1, "y": {"z": 2}})
    b = config_hash({"y": {"z": 2}, "x": 1})
    assert a == b


def test_config_hash_changes_when_a_value_changes():
    assert config_hash({"x": 1}) != config_hash({"x": 2})


def test_flatten_produces_dotted_paths():
    assert dict(flatten({"a": {"b": {"c": 1}}, "d": 2})) == {"a.b.c": 1, "d": 2}


# ── the pre-registration lock ───────────────────────────────────────────────


def test_agreeing_config_has_no_violations():
    assert lock_violations({"decode": {"beam_size": 4, "length_penalty": 1.0}}) == []


def test_drifting_value_is_reported():
    out = lock_violations({"decode": {"beam_size": 8}})
    assert len(out) == 1
    assert "decode.beam_size" in out[0] and "live=8" in out[0] and "locked=4" in out[0]


def test_keys_absent_from_the_live_config_are_fine():
    """A config need not restate the whole study."""
    assert lock_violations({"paths": {"images_dir": "anywhere"}}) == []


def test_assert_matches_lock_explains_how_to_deviate_properly():
    with pytest.raises(ValueError) as exc:
        assert_matches_lock({"metrics": {"primary": "bleu_4"}})
    msg = str(exc.value)
    assert "metrics.primary" in msg
    assert "docs/deviations.md" in msg
    assert "Do not silently edit the lock" in msg


def test_shipped_data_config_agrees_with_the_lock():
    """configs/data.yaml must never contradict the pre-registration.

    This is the guard against v1's actual failure: a header advertising
    quality >= 0.5 while the code ran quality-disabled.
    """
    assert lock_violations(load_config("data")) == []


def test_data_config_restates_every_locked_data_key():
    """A pre-registered key that a config silently *drops* is as dangerous as one
    it contradicts.

    Caught for real: adopting the official Flickr8k splits landed in data.yaml but
    not in the lock, leaving `split_source: official_flickr8k` in one file and
    `split_ratios: 80/10/10` in the other. Contradiction-only checking passed both,
    because neither key appeared in both files.
    """
    assert lock_violations(load_config("data"), require_sections=("data",)) == []


def test_missing_locked_key_is_reported_when_the_section_is_required():
    bad = lock_violations({"data": {"dataset": "flickr8k"}}, require_sections=("data",))
    assert any("MISSING" in v for v in bad)
    assert any("split_source" in v for v in bad)


def test_missing_keys_are_ignored_outside_required_sections():
    assert lock_violations({"data": {"dataset": "flickr8k"}}) == []


def test_seeds_in_the_lock_match_the_code_constant():
    from emocap.runtime import PREREGISTERED_SEEDS

    assert tuple(load_config("prereg.lock")["study"]["seeds"]) == PREREGISTERED_SEEDS


def test_emotion_order_is_consistent_between_lock_and_prompt():
    """emotion_id comes from list position, so a reorder silently relabels data."""
    from emocap.data.prompt import EMOTIONS

    assert tuple(load_config("prereg.lock")["study"]["emotions"]) == EMOTIONS


def test_decode_defaults_match_the_lock():
    """DecodeConfig defaults are what every run uses if a notebook forgets to pass one."""
    from emocap.decode import DecodeConfig

    locked = load_config("prereg.lock")["decode"]
    assert DecodeConfig().as_dict() == locked
    assert lock_violations(
        {"decode": DecodeConfig().as_dict()}, require_sections=("decode",)
    ) == []


# ── manifests ───────────────────────────────────────────────────────────────


def test_manifest_captures_provenance():
    m = Manifest(run_id="r1", stage="07_train_lstm", config={"lr": 1e-3}, seed=42)
    d = m.to_dict()
    assert d["run_id"] == "r1" and d["seed"] == 42
    assert d["config_hash"] == m.config_hash
    assert d["status"] == "running"
    assert "torch" in d["packages"]
    assert d["python"] and d["platform"]


def test_manifest_roundtrips_through_disk(tmp_path):
    m = Manifest(run_id="r1", stage="s", config={"a": 1}, seed=7)
    m.save(tmp_path)
    loaded = Manifest.load(tmp_path)
    assert loaded["run_id"] == "r1" and loaded["seed"] == 7
    assert json.dumps(loaded)  # must be plain JSON, no exotic types


def test_manifest_is_written_before_the_run_finishes(tmp_path):
    """An interrupted run must still leave a record of what it attempted.

    v1's abandoned runs left nothing at all.
    """
    m = Manifest(run_id="r1", stage="s", config={})
    path = m.save(tmp_path)
    assert path.exists()
    assert Manifest.load(tmp_path)["status"] == "running"
    assert Manifest.load(tmp_path)["finished_at"] is None


def test_finalise_records_status_and_wall_time(tmp_path):
    m = Manifest(run_id="r1", stage="s", config={})
    m.finalise(tmp_path)
    d = Manifest.load(tmp_path)
    assert d["status"] == "complete"
    assert d["wall_seconds"] is not None and d["wall_seconds"] >= 0


def test_a_failed_gate_shows_up_in_the_status(tmp_path):
    """The visual-dependence probe must not be able to fail quietly."""
    m = Manifest(run_id="r1", stage="s", config={})
    m.record_check("visual_dependence_probe", passed=False, detail="output unchanged")
    m.finalise(tmp_path)
    d = Manifest.load(tmp_path)
    assert d["status"] == "complete_with_failed_checks"
    assert d["checks"]["visual_dependence_probe"]["passed"] is False
    assert d["checks"]["visual_dependence_probe"]["detail"] == "output unchanged"


def test_passing_gates_leave_the_status_clean(tmp_path):
    m = Manifest(run_id="r1", stage="s", config={})
    m.record_check("visual_dependence_probe", passed=True)
    m.finalise(tmp_path)
    assert Manifest.load(tmp_path)["status"] == "complete"


# ── seeds ───────────────────────────────────────────────────────────────────


def test_seeding_makes_draws_reproducible():
    import random

    import torch

    seed_everything(42)
    a = (random.random(), torch.randn(3).tolist())
    seed_everything(42)
    b = (random.random(), torch.randn(3).tolist())
    assert a == b


def test_different_seeds_diverge():
    import torch

    seed_everything(42)
    a = torch.randn(3).tolist()
    seed_everything(1337)
    assert a != torch.randn(3).tolist()
