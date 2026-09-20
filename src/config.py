"""Configuration loading: default.yaml + optional override file + CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def to_dict(ns: Any) -> Any:
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> SimpleNamespace:
    """Load default.yaml, overlay `path` (if given) and then `overrides` (nested dict)."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return _to_ns(cfg)
