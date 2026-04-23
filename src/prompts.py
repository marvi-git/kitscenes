"""Prompt generation for KITScenes LongTail dataset instances.

Converts a raw dataset row (dict from parquet) into a provider-agnostic Prompt
that can be sent to any VLM API (Anthropic, OpenAI, Gemini, …).  API-specific
encoding (base64, MessageParam, etc.) is left to caller-side adapters.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

# ── public types ──────────────────────────────────────────────────────────────

Camera = Literal["front", "front_left", "front_right", "rear", "rear_left", "rear_right"]
FrameStrategy = Literal["last", "all", "first_middle_last"]
OutputFormat = Literal["cot_structured", "waypoints_only", "cot_free"]
TrajectoryEncoding = Literal["numeric", "textual", "both"]
Language = Literal["en", "es", "zh"]

_LANG_KEY = {"en": "english", "es": "spanish", "zh": "chinese"}


@dataclass
class ImagePart:
    bytes: bytes
    mime_type: str = "image/jpeg"


@dataclass
class FewShotExample:
    user_text: str
    images: list[ImagePart]
    assistant_text: str


@dataclass
class Prompt:
    system: str
    user_text: str
    images: list[ImagePart]
    few_shot: list[FewShotExample] = field(default_factory=list)
    scenario_id: str | None = None


# ── public API ────────────────────────────────────────────────────────────────


def build_prompt(
    instance: dict,
    *,
    cameras: Sequence[Camera] = ("front",),
    frame_strategy: FrameStrategy = "last",
    output_format: OutputFormat = "cot_structured",
    trajectory_encoding: TrajectoryEncoding = "both",
    language: Language = "en",
    few_shot_examples: Sequence[dict] = (),
) -> Prompt:
    """Build a provider-agnostic Prompt from a single dataset instance.

    Args:
        instance: Dict matching the KITScenes parquet schema.
        cameras: Which cameras to include (default: front only).
        frame_strategy: "last" | "all" | "first_middle_last".
        output_format: What the model should produce.
        trajectory_encoding: How to encode the past trajectory in the user text.
        language: Which language variant of ``reasoning`` to use in few-shot answers.
        few_shot_examples: Optional list of train instances to include as solved
            examples before the query.  Caller is responsible for selection.

    Returns:
        A Prompt with system/user text, images, and (optionally) few-shot pairs.
    """
    past = instance["trajectory"]["past"]
    if len(past) != 21:
        raise ValueError(f"expected 21 past waypoints, got {len(past)}")

    images = _select_frames(instance, cameras, frame_strategy)
    past_text = _encode_past_trajectory(past, trajectory_encoding)
    system = _build_system(output_format)
    user_text = _build_user_text(instance["driving_instruction"], past_text)

    few_shot = [
        FewShotExample(
            user_text=_build_user_text(
                ex["driving_instruction"],
                _encode_past_trajectory(ex["trajectory"]["past"], trajectory_encoding),
            ),
            images=_select_frames(ex, cameras, frame_strategy),
            assistant_text=_format_expert_answer(ex, output_format, language),
        )
        for ex in few_shot_examples
    ]

    return Prompt(
        system=system,
        user_text=user_text,
        images=images,
        few_shot=few_shot,
        scenario_id=instance.get("scenario_id"),
    )


# ── private helpers ───────────────────────────────────────────────────────────


def _select_frames(
    instance: dict,
    cameras: Sequence[Camera],
    strategy: FrameStrategy,
) -> list[ImagePart]:
    images: list[ImagePart] = []
    for cam in cameras:
        frames = instance[f"frames_camera_{cam}"]
        selected = _apply_strategy(frames, strategy)
        images.extend(ImagePart(bytes=f["bytes"]) for f in selected)
    return images


def _apply_strategy(frames: list, strategy: FrameStrategy) -> list:
    if not frames:
        return []
    if strategy == "last":
        return [frames[-1]]
    if strategy == "all":
        return frames
    # first_middle_last
    n = len(frames)
    indices = sorted({0, n // 2, n - 1})
    return [frames[i] for i in indices]


def _encode_past_trajectory(past: list, mode: TrajectoryEncoding) -> str:
    if mode == "numeric":
        return _numeric(past)
    if mode == "textual":
        return _textual(past)
    return _textual(past) + "\n" + _numeric(past)


def _numeric(past: list) -> str:
    rounded = [[round(x, 2), round(y, 2)] for x, y in past]
    return f"past = {json.dumps(rounded)}"


def _textual(past: list) -> str:
    diffs = [
        math.hypot(past[i + 1][0] - past[i][0], past[i + 1][1] - past[i][1])
        for i in range(len(past) - 1)
    ]
    avg_speed = sum(diffs) / len(diffs) * 5.0  # 5 Hz
    recent_speed = sum(diffs[-5:]) / 5 * 5.0

    total_dy = past[-1][1] - past[0][1]
    if abs(total_dy) < 0.5:
        lateral = "mostly straight"
    elif total_dy > 0:
        lateral = "slight rightward curve"
    else:
        lateral = "slight leftward curve"

    return (
        f"Average speed: {avg_speed:.1f} m/s ({avg_speed * 3.6:.0f} km/h). "
        f"Recent speed: {recent_speed:.1f} m/s. "
        f"Motion: forward, {lateral}."
    )


_COT_FIELDS = [
    "situational_awareness",
    "acceleration_first_3s",
    "reason_acceleration_first_3s",
    "steering_first_3s",
    "reason_steering_first_3s",
    "acceleration_last_2s",
    "reason_acceleration_last_2s",
    "steering_last_2s",
    "reason_steering_last_2s",
]

_COT_SCHEMA = "\n".join(f'  "{f}": "...",' for f in _COT_FIELDS)


def _build_system(output_format: OutputFormat) -> str:
    base = (
        "You are an expert autonomous driving model. "
        "Given one or more front-view images and the vehicle's past trajectory, "
        "predict the next 25 waypoints (5 seconds at 5 Hz) "
        "in the ego-vehicle coordinate frame (x forward, y left, units: metres).\n\n"
    )

    if output_format == "cot_structured":
        schema = (
            '{\n'
            + _COT_SCHEMA + "\n"
            + '  "future_waypoints": [[x1, y1], [x2, y2], ..., [x25, y25]]\n'
            + '}'
        )
        return (
            base
            + "Respond with a single JSON object following this schema exactly:\n"
            + schema
        )

    if output_format == "waypoints_only":
        return (
            base
            + 'Respond with a single JSON object:\n'
            + '{"future_waypoints": [[x1, y1], ..., [x25, y25]]}'
        )

    # cot_free
    return (
        base
        + "Reason step by step about the scene and the driving instruction, "
        "then end your response with a JSON block:\n"
        '{"future_waypoints": [[x1, y1], ..., [x25, y25]]}'
    )


def _build_user_text(driving_instruction: str, past_encoding: str) -> str:
    return (
        f"Driving instruction: {driving_instruction}\n\n"
        f"Past trajectory (21 waypoints, 4.0 s at 5 Hz, ego frame):\n"
        f"{past_encoding}"
    )


def _format_expert_answer(
    instance: dict,
    output_format: OutputFormat,
    language: Language,
) -> str:
    lang_key = _LANG_KEY[language]
    future = instance["trajectory"]["expert_like"]
    waypoints = [[round(x, 4), round(y, 4)] for x, y in future]

    if output_format == "waypoints_only":
        return json.dumps({"future_waypoints": waypoints})

    if output_format == "cot_structured":
        r = instance["reasoning"][lang_key]
        obj = {f: r[f] for f in _COT_FIELDS}
        obj["future_waypoints"] = waypoints
        return json.dumps(obj)

    # cot_free: a short free-form reasoning followed by JSON
    r = instance["reasoning"][lang_key]
    return (
        r["situational_awareness"] + "\n\n"
        + json.dumps({"future_waypoints": waypoints})
    )
