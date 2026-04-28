"""Model-agnostic API client for waypoint prediction.

Usage:
    client = get_client("claude")                        # defaults to claude-opus-4-7
    client = get_client("openai", model="gpt-4o")
    client = get_client("ollama", model="gemma3:12b")    # local, free
    waypoints = client.predict(prompt_string)            # -> list of 25 (x, y) tuples
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

Waypoints = list[tuple[float, float]]

_EXPECTED = 25


def parse_waypoints(text: str) -> Waypoints:
    """Extract 25 (x, y) waypoints from model output.

    Tries JSON first; falls back to regex scanning for numeric pairs.
    Raises ValueError if fewer than expected waypoints are found.
    """
    # --- JSON path ---
    # Look for the outermost [...] or {...} that contains coordinate data.
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pattern, text)
        if not m:
            continue
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            continue

        pairs = _extract_pairs_from_json(data)
        if len(pairs) >= _EXPECTED:
            return pairs[:_EXPECTED]

    # --- Regex fallback ---
    # Match any sequence like (1.2, 3.4), [1.2, 3.4], or bare "1.2, 3.4" lines.
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(numbers) >= _EXPECTED * 2:
        pairs = [
            (float(numbers[i]), float(numbers[i + 1]))
            for i in range(0, _EXPECTED * 2, 2)
        ]
        return pairs

    raise ValueError(
        f"Could not parse {_EXPECTED} waypoints from model output.\n"
        f"Raw text (first 500 chars): {text[:500]}"
    )


def _extract_pairs_from_json(data: object) -> Waypoints:
    """Recursively pull (x, y) pairs out of parsed JSON."""
    pairs: Waypoints = []

    if isinstance(data, list):
        for item in data:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(v, (int, float)) for v in item)
            ):
                pairs.append((float(item[0]), float(item[1])))
            elif isinstance(item, dict):
                x = item.get("x") or item.get("X")
                y = item.get("y") or item.get("Y")
                if x is not None and y is not None:
                    pairs.append((float(x), float(y)))
            else:
                pairs.extend(_extract_pairs_from_json(item))

    elif isinstance(data, dict):
        if "waypoints" in data:
            pairs.extend(_extract_pairs_from_json(data["waypoints"]))
        elif "predictions" in data:
            pairs.extend(_extract_pairs_from_json(data["predictions"]))
        else:
            for v in data.values():
                pairs.extend(_extract_pairs_from_json(v))

    return pairs


class ModelClient(ABC):
    @abstractmethod
    def predict(self, prompt: str) -> Waypoints:
        """Send prompt to model; return 25 (x, y) waypoints."""


class ClaudeClient(ModelClient):
    DEFAULT_MODEL = "claude-opus-4-7"

    def __init__(self, model: str | None = None, **kwargs):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        self._anthropic = anthropic
        self.model = model or self.DEFAULT_MODEL
        self.client = anthropic.Anthropic(**kwargs)

    def predict(self, prompt: str) -> Waypoints:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()

        text = "".join(
            block.text
            for block in message.content
            if block.type == "text"
        )
        return parse_waypoints(text)


class OpenAIClient(ModelClient):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str | None = None, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        self.model = model or self.DEFAULT_MODEL
        self.client = OpenAI(**kwargs)

    def predict(self, prompt: str) -> Waypoints:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return parse_waypoints(text)


class OllamaClient(OpenAIClient):
    """Local Ollama server via its OpenAI-compatible API.

    Requires Ollama running locally (https://ollama.com).
    Pull a model first: ollama pull gemma3:12b
    """

    DEFAULT_MODEL = "gemma3:12b"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, model: str | None = None, **kwargs):
        kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
        kwargs.setdefault("api_key", "ollama")  # Ollama ignores the key but openai SDK requires one
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)


_PROVIDERS: dict[str, type[ModelClient]] = {
    "claude": ClaudeClient,
    "openai": OpenAIClient,
    "ollama": OllamaClient,
}


def get_client(provider: str, model: str | None = None, **kwargs) -> ModelClient:
    """Return a ModelClient for the given provider.

    Args:
        provider: "claude", "openai", or "ollama"
        model: model name override (e.g. "gemma3:12b", "gpt-4o", "claude-sonnet-4-6")
        **kwargs: passed to the underlying SDK client (e.g. api_key, base_url)
    """
    key = provider.lower()
    if key not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    return _PROVIDERS[key](model=model, **kwargs)
