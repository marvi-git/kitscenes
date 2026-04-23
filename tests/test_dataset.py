"""Smoke tests for src/dataset.py — skipped when data/ files are absent."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TRAIN_FILES = sorted(DATA_DIR.joinpath("train").glob("train-*.parquet"))
_HAS_TRAIN = bool(_TRAIN_FILES)


def _skip_no_data():
    if not _HAS_TRAIN:
        pytest.skip("data/train/*.parquet not available")


# ── load_instance ─────────────────────────────────────────────────────────────


def test_load_instance_returns_dict_with_expected_keys():
    _skip_no_data()
    from src.dataset import load_instance

    row = load_instance("train", 0)
    assert isinstance(row, dict)
    for key in (
        "scenario_id",
        "frames_camera_front",
        "driving_instruction",
        "scenario_type",
        "trajectory",
        "reasoning",
    ):
        assert key in row, f"missing key: {key}"


def test_load_instance_front_frame_is_jpeg():
    _skip_no_data()
    from src.dataset import load_instance

    row = load_instance("train", 0)
    frames = row["frames_camera_front"]
    assert len(frames) == 21
    first_bytes = frames[0]["bytes"]
    assert isinstance(first_bytes, bytes) and len(first_bytes) > 0
    assert first_bytes[:2] == b"\xff\xd8", "expected JPEG magic"


def test_load_instance_past_has_21_waypoints():
    _skip_no_data()
    from src.dataset import load_instance

    row = load_instance("train", 0)
    assert len(row["trajectory"]["past"]) == 21


def test_load_instance_expert_like_has_25_waypoints():
    _skip_no_data()
    from src.dataset import load_instance

    row = load_instance("train", 0)
    assert len(row["trajectory"]["expert_like"]) == 25


# ── iter_instances ────────────────────────────────────────────────────────────


def test_iter_instances_yields_two_dicts_with_limit():
    _skip_no_data()
    from src.dataset import iter_instances

    rows = list(iter_instances("train", limit=2))
    assert len(rows) == 2
    assert all(isinstance(r, dict) for r in rows)
    assert all("scenario_id" in r for r in rows)


def test_iter_instances_scenario_ids_are_strings():
    _skip_no_data()
    from src.dataset import iter_instances

    for row in iter_instances("train", limit=3):
        assert isinstance(row["scenario_id"], str)


# ── build_prompt integration ──────────────────────────────────────────────────


def test_build_prompt_on_real_instance():
    _skip_no_data()
    from src.dataset import load_instance
    from src.prompts import build_prompt

    row = load_instance("train", 0)
    p = build_prompt(row)
    assert p.system.strip()
    assert p.user_text.strip()
    assert len(p.images) == 1
    assert p.images[0].bytes[:2] == b"\xff\xd8"
