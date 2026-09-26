from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES_DEFAULT = ["rest", "fist", "open", "pinch", "point", "thumb-up"]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config is empty: {cfg_path}")
    cfg["_path"] = str(cfg_path)
    cfg.setdefault("classes", CLASSES_DEFAULT)
    cfg.setdefault("drop_channels", [])
    cfg.setdefault("fallback_data_dir", None)
    return cfg


def resolve_dir(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
