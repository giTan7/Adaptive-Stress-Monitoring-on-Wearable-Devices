"""
Central configuration for the AdaStress pipeline.

Everything that another user would want to change before running the
pipeline on their own data lives in `configs/config.yaml`. This module just
loads that file into a plain dict-like object so the rest of the codebase
never hardcodes a path or a hyperparameter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"
)


class AttrDict(dict):
    """A dict whose keys are also accessible as attributes, recursively."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str = DEFAULT_CONFIG_PATH) -> AttrDict:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return AttrDict(raw)
