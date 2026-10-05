"""
Injectable model clients for reproducible runs.

ReplayClient answers extraction requests from recordings files instead of calling a
model, keyed by the stable review ID of the review in the request. A review without
a recording raises ReplayError, so a gap is an explicit extraction failure, never an
invented answer.

RecordingClient wraps a real client and keeps every answer it returns, so a live run
can be saved and replayed exactly later.

Recordings file format:
    {"description": "...", "recipe_file": "data/...json",
     "responses": {"rev-0123456789": {"tweaks": [...]}  # or a list: one per attempt
    }}
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from .loader import review_id


class ReplayError(RuntimeError):
    """No recorded response for the requested review."""


def _review_text(messages: List[dict]) -> str:
    match = re.search(r'"""(.*)"""', messages[1]["content"], re.S)
    if not match:
        raise ReplayError("request does not contain a review")
    return match.group(1)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason="stop")])


class ReplayClient:
    """OpenAI-compatible client that replays recorded extraction responses."""

    def __init__(self, *paths: Union[str, Path]):
        self.responses: Dict[str, List] = {}
        for path in paths:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for rid, value in data["responses"].items():
                self.responses[rid] = value if isinstance(value, list) else [value]
        self.calls: Dict[str, int] = {}
        self.chat = SimpleNamespace(completions=self)
        self.description = "replay of recorded responses (" + ", ".join(Path(p).name for p in paths) + ")"

    def create(self, **kwargs) -> SimpleNamespace:
        rid = review_id(_review_text(kwargs["messages"]))
        if rid not in self.responses:
            raise ReplayError(f"no recorded response for review {rid}")
        attempt = self.calls.get(rid, 0)
        self.calls[rid] = attempt + 1
        recorded = self.responses[rid]
        value = recorded[min(attempt, len(recorded) - 1)]
        return _response(value if isinstance(value, str) else json.dumps(value))


class RecordingClient:
    """Wraps a real client and records its answers per review for later replay."""

    def __init__(self, client):
        self.client = client
        self.responses: Dict[str, List[str]] = {}
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        rid = review_id(_review_text(kwargs["messages"]))
        self.responses.setdefault(rid, []).append(response.choices[0].message.content)
        return response

    def save(self, path: Union[str, Path], recipe_file: str, description: str) -> None:
        Path(path).write_text(json.dumps(
            {"description": description, "recipe_file": recipe_file, "responses": self.responses},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
