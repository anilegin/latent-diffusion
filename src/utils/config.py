from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, update: dict) -> dict:
    """
    Recursively merge update into base.
    """
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a config file and recursively merge files listed under `defaults`.

    Example:
        defaults:
          - ../paths.yaml
          - ../data/coco_256.yaml
    """
    config_path = Path(config_path)
    cfg = load_yaml(config_path)

    merged: dict[str, Any] = {}

    defaults = cfg.pop("defaults", [])
    for default_path in defaults:
        default_file = (config_path.parent / default_path).resolve()
        default_cfg = load_config(default_file)
        merged = deep_update(merged, default_cfg)

    merged = deep_update(merged, cfg)
    return merged


def get_by_key(cfg: dict[str, Any], key: str) -> Any:
    """
    Get nested config value using dot notation.

    Example:
        get_by_key(cfg, "data.coco_root")
    """
    value: Any = cfg
    for part in key.split("."):
        value = value[part]
    return value


def resolve_path_key(cfg: dict[str, Any], key: str) -> Path:
    """
    Resolve a path stored in config using dot notation.
    """
    return Path(get_by_key(cfg, key)).expanduser().resolve()