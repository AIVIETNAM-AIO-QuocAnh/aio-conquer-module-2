from pathlib import Path
import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg

def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a relative path from the config into an absolute path anchored at PROJECT_ROOT.

    Paths are required config: a missing 'paths' section or a missing key fails
    immediately (fail fast) instead of silently returning PROJECT_ROOT and writing
    to the wrong place.
    """
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Config is missing the 'paths' section.")
    if key not in paths:
        available = ", ".join(sorted(paths)) or "(empty)"
        raise KeyError(f"Path '{key}' not found in config. Available: {available}")

    return PROJECT_ROOT / paths[key]