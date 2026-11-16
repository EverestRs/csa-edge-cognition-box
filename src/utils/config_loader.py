from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
