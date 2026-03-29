"""
Application-level configuration for MoldVision.

Config file location (platform-aware):
  Windows : %LOCALAPPDATA%\MoldVision\config.json
  macOS   : ~/Library/Application Support/MoldVision/config.json
  Linux   : $XDG_CONFIG_HOME/moldvision/config.json  (default: ~/.config/moldvision/config.json)

Environment variable overrides take precedence over config file values.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Environment variable names
ENV_DATASETS = "MOLDVISION_DATASETS"

_APP_NAME_WIN = "MoldVision"
_APP_NAME_UNIX = "moldvision"


def config_dir() -> Path:
    """Return the platform-appropriate directory that holds config.json."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / _APP_NAME_WIN
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME_WIN
    # Linux / other POSIX
    xdg = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(xdg) / _APP_NAME_UNIX


def config_path() -> Path:
    """Return the full path to the JSON config file."""
    return config_dir() / "config.json"


def load_config() -> Dict[str, Any]:
    """Load config from disk.  Returns an empty dict if the file is missing or unreadable."""
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist *cfg* to the config file, creating parent directories as needed."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Individual setting helpers
# ---------------------------------------------------------------------------

def get_default_dataset_root() -> str:
    """
    Resolve the default dataset root using the priority chain:
      1. Environment variable  MOLDVISION_DATASETS
      2. Config file           config.json  »  default_dataset_root
      3. Hard fallback         "datasets"   (relative to CWD — legacy behaviour)
    """
    env = os.environ.get(ENV_DATASETS)
    if env:
        return env
    cfg = load_config()
    root = cfg.get("default_dataset_root")
    if root and isinstance(root, str):
        return root
    return "datasets"


def set_default_dataset_root(path: str) -> None:
    """Persist a new default_dataset_root to the config file."""
    cfg = load_config()
    cfg["default_dataset_root"] = str(Path(path).expanduser())
    save_config(cfg)


def get_setting(key: str) -> Optional[Any]:
    """Generic getter for any config key."""
    return load_config().get(key)


def set_setting(key: str, value: Any) -> None:
    """Generic setter — persists a single key into the config file."""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)
