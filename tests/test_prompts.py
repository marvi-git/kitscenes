"""Unit tests for src/prompts.py — uses a synthetic fixture, no real parquet needed."""
import json
import pytest

from src.prompts import FewShotExample, ImagePart, Prompt, build_prompt

# ── synthetic fixture ─────────────────────────────────────────────────────────

_DUMMY_FRAME = b"\xff\xd8\xff\xe0" + b"\x00" * 50  # valid JPEG magic + padding


def _frames(n: int = 21) -> list:
    return [{"bytes": _DUMMY_FRAME, "path": f"frame_{i:02d}.jpg"} for i in range(n)]


def _past(n: int = 21) -> list:
    # straight drive along +x, ending near (0, 0)
    return [[-float(n - 1 - i), 0.0] for i in range(n)]


def _future(n: int = 25) -> list:
    return [[float(i + 1), float(i + 1)] for i in range(n)]


_REASONING_EN = {
    "situational_awareness": "I am approaching an intersection.",
    "acceleration_first_3s": "maintain the current speed",
    "reason_acceleration_first_3s": "to make a left turn",
    "steering_first_3s": "steer to the left",
    "reason_steering_first_3s": "to make a left turn",
    "acceleration_last_2s": "maintain the current speed",
    "reason_acceleration_last_2s": "to finish the left turn",
    "steering_last_2s": "steer straight",
    "reason_steering_last_2s": "to center the car in the lane",
}

_REASONING_ES = {
    "situational_awareness": "Me acerco a una intersección.",
    "acceleration_first_3s": "mantener la velocidad",
    "reason_acceleration_first_3s": "para girar",
    "steering_first_3s": "girar a la izquierda",
    "reason_steering_first_3s": "para girar",
    "acceleration_last_2s": "mantener la velocidad",
    "reason_acceleration_last_2s": "para terminar el giro",
    "steering_last_2s": "enderezar",
    "reason_steering_last_2s": "para centrar el coche",
}

_REASONING_ZH = {
    "situational_awareness": "我接近一个路口。",
    "acceleration_first_3s": "保持速度",
    "reason_acceleration_first_3s": "为了转弯",
    "steering_first_3s": "向左转",
    "reason_steering_first_3s": "为了转弯",
    "acceleration_last_2s": "保持速度",
    "reason_acceleration_last_2s": "为了完成转弯",
    "steering_last_2s": "回正",
    "reason_steering_last_2s": "为了居中",
}

DUMMY = {
    "scenario_id": "TEST0001",
    "frames_camera_front": _frames(21),
    "frames_camera_front_left": _frames(21),
    "frames_camera_front_right": _frames(21),
    "frames_camera_rear_left": _frames(21),
    "frames_camera_rear": _frames(21),
    "frames_camera_rear_right": _frames(21),
    "driving_instruction": "turn left",
    "scenario_type": "4 intersection",
    "trajectory": {
        "past": _past(21),
        "expert_like": _future(25),
        "wrong_speed": _future(25),
        "neglect_instruction": _future(25),
        "off_road": _future(25),
        "crash": _future(25),
    },
    "reasoning": {
        "english": _REASONING_EN,
        "spanish": _REASONING_ES,
        "chinese": _REASONING_ZH,
    },
}

# ── basic structure ───────────────────────────────────────────────────────────


def test_returns_prompt_instance():
    assert isinstance(build_prompt(DUMMY), Prompt)


def test_system_and_user_text_non_empty():
    p = build_prompt(DUMMY)
    assert p.system.strip()
    assert p.user_text.strip()


def test_driving_instruction_in_user_text():
    p = build_prompt(DUMMY)
    assert "turn left" in p.user_text


def test_scenario_id_propagated():
    p = build_prompt(DUMMY)
    assert p.scenario_id == "TEST0001"


def test_no_few_shot_by_default():
    p = build_prompt(DUMMY)
    assert p.few_shot == []


# ── cameras / frame strategy ──────────────────────────────────────────────────


def test_default_gives_one_front_image():
    p = build_prompt(DUMMY)
    assert len(p.images) == 1
    assert p.images[0].bytes == _DUMMY_FRAME


def test_frame_strategy_last_one_image_per_camera():
    p = build_prompt(DUMMY, cameras=("front",), frame_strategy="last")
    assert len(p.images) == 1


def test_frame_strategy_first_middle_last_gives_three_images():
    p = build_prompt(DUMMY, cameras=("front",), frame_strategy="first_middle_last")
    assert len(p.images) == 3


def test_frame_strategy_all_gives_all_frames():
    p = build_prompt(DUMMY, cameras=("front",), frame_strategy="all")
    assert len(p.images) == 21


def test_two_cameras_last_gives_two_images():
    p = build_prompt(DUMMY, cameras=("front", "rear"), frame_strategy="last")
    assert len(p.images) == 2


def test_all_six_cameras_last_gives_six_images():
    p = build_prompt(
        DUMMY,
        cameras=("front", "front_left", "front_right", "rear", "rear_left", "rear_right"),
        frame_strategy="last",
    )
    assert len(p.images) == 6


# ── output format ─────────────────────────────────────────────────────────────


def test_cot_structured_system_has_cot_fields_and_future_waypoints():
    p = build_prompt(DUMMY, output_format="cot_structured")
    assert "situational_awareness" in p.system
    assert "future_waypoints" in p.system


def test_waypoints_only_system_has_future_waypoints_no_cot_fields():
    p = build_prompt(DUMMY, output_format="waypoints_only")
    assert "future_waypoints" in p.system
    assert "situational_awareness" not in p.system


def test_cot_free_system_has_future_waypoints_no_fixed_schema():
    p = build_prompt(DUMMY, output_format="cot_free")
    assert "future_waypoints" in p.system
    assert "situational_awareness" not in p.system


# ── trajectory encoding ───────────────────────────────────────────────────────


def test_numeric_encoding_contains_coordinate_brackets():
    p = build_prompt(DUMMY, trajectory_encoding="numeric")
    # e.g. "[[-20.0, 0.0]"
    assert "[[" in p.user_text


def test_textual_encoding_has_no_raw_coord_brackets():
    p = build_prompt(DUMMY, trajectory_encoding="textual")
    assert "[[" not in p.user_text


def test_both_encoding_longer_than_numeric_alone():
    p_num = build_prompt(DUMMY, trajectory_encoding="numeric")
    p_both = build_prompt(DUMMY, trajectory_encoding="both")
    assert len(p_both.user_text) > len(p_num.user_text)


# ── few-shot ──────────────────────────────────────────────────────────────────


def test_few_shot_creates_one_example():
    p = build_prompt(DUMMY, few_shot_examples=[DUMMY])
    assert len(p.few_shot) == 1
    assert isinstance(p.few_shot[0], FewShotExample)


def test_few_shot_example_has_non_empty_user_text_and_images():
    ex = build_prompt(DUMMY, few_shot_examples=[DUMMY]).few_shot[0]
    assert ex.user_text.strip()
    assert len(ex.images) >= 1


def test_few_shot_cot_structured_assistant_is_valid_json_with_25_waypoints():
    ex = build_prompt(DUMMY, output_format="cot_structured", few_shot_examples=[DUMMY]).few_shot[0]
    data = json.loads(ex.assistant_text)
    assert "future_waypoints" in data
    assert len(data["future_waypoints"]) == 25


def test_few_shot_waypoints_only_assistant_is_valid_json_with_25_waypoints():
    ex = build_prompt(DUMMY, output_format="waypoints_only", few_shot_examples=[DUMMY]).few_shot[0]
    data = json.loads(ex.assistant_text)
    assert "future_waypoints" in data
    assert len(data["future_waypoints"]) == 25


def test_few_shot_cot_structured_assistant_contains_reasoning():
    ex = build_prompt(DUMMY, output_format="cot_structured", few_shot_examples=[DUMMY]).few_shot[0]
    data = json.loads(ex.assistant_text)
    assert data["situational_awareness"] == "I am approaching an intersection."


def test_few_shot_language_es_uses_spanish_reasoning():
    ex = build_prompt(DUMMY, language="es", few_shot_examples=[DUMMY]).few_shot[0]
    data = json.loads(ex.assistant_text)
    assert data["situational_awareness"] == "Me acerco a una intersección."


def test_few_shot_language_zh_uses_chinese_reasoning():
    ex = build_prompt(DUMMY, language="zh", few_shot_examples=[DUMMY]).few_shot[0]
    data = json.loads(ex.assistant_text)
    assert data["situational_awareness"] == "我接近一个路口。"


# ── validation ────────────────────────────────────────────────────────────────


def test_past_wrong_length_raises_value_error():
    bad = {
        **DUMMY,
        "trajectory": {**DUMMY["trajectory"], "past": [[0.0, 0.0]] * 5},
    }
    with pytest.raises(ValueError, match="past"):
        build_prompt(bad)
