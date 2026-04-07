"""Model registry for the ARIA Data Lake.

Commands:
  lake models install  <bundle.mpk> --task detect|seg
  lake models list     [--task detect|seg]
  lake models promote  <bundle_id>  --channel stable|dev

Bundles are extracted under ``models/<task>/<bundle_id>/`` inside the lake.
``models/<task>/registry.json`` is the canonical index.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lake import LakeConfig

# Task name → registry subdirectory name
_TASK_DIR = {
    "detect": "defect_detection",
    "seg":    "monitor_segmentation",
}


def _reg_path(cfg: LakeConfig, task: str) -> Path:
    task_dir = _TASK_DIR.get(task, task)
    return cfg.storage().abs_path(f"models/{task_dir}/registry.json")


def _load_registry(cfg: LakeConfig, task: str) -> Dict[str, Any]:
    p = _reg_path(cfg, task)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"task": task, "bundles": [], "active": {"stable": None, "dev": None}}


def _save_registry(cfg: LakeConfig, task: str, reg: Dict[str, Any]) -> None:
    p = _reg_path(cfg, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# install
# ──────────────────────────────────────────────────────────────────────────────

def models_install(cfg: LakeConfig, *, bundle_path: Path, task: str) -> str:
    """Extract *bundle_path* (.mpk) into the model registry.

    Returns the ``bundle_id`` (read from the bundle's ``manifest.json``).
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    # .mpk is a zip archive
    if not zipfile.is_zipfile(bundle_path):
        raise ValueError(f"Not a valid .mpk (zip) file: {bundle_path}")

    task_dir = _TASK_DIR.get(task, task)
    storage = cfg.storage()

    # Extract to a temp folder first, then rename to bundle_id
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "bundle"
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(tmp_path)

        # Read manifest.json to get bundle_id
        manifest_path = tmp_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("bundle .mpk does not contain manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_id: str = manifest.get("bundle_id", "")
        if not bundle_id:
            raise ValueError("manifest.json must contain a 'bundle_id' field")

        dest_rel = f"models/{task_dir}/{bundle_id}"
        dest_abs = storage.abs_path(dest_rel)

        if dest_abs.exists():
            print(f"Bundle '{bundle_id}' already exists; overwriting.")
            shutil.rmtree(dest_abs)

        shutil.copytree(tmp_path, dest_abs)

    # Update registry
    reg = _load_registry(cfg, task)
    # Remove existing entry with same bundle_id
    reg["bundles"] = [b for b in reg["bundles"] if b.get("bundle_id") != bundle_id]
    reg["bundles"].append({
        "bundle_id":    bundle_id,
        "version":      manifest.get("model_version", ""),
        "channel":      "dev",
        "dataset_uuid": manifest.get("dataset_uuid", ""),
        "created_at":   datetime.utcnow().isoformat() + "Z",
        "path":         f"models/{task_dir}/{bundle_id}/",
    })
    _save_registry(cfg, task, reg)

    print(f"Installed bundle '{bundle_id}' → {storage.abs_path(dest_rel)}")
    return bundle_id


# ──────────────────────────────────────────────────────────────────────────────
# list
# ──────────────────────────────────────────────────────────────────────────────

def models_list(cfg: LakeConfig, task: Optional[str] = None) -> None:
    """Print all registered bundles."""
    tasks = [task] if task else list(_TASK_DIR.keys())
    for t in tasks:
        reg = _load_registry(cfg, t)
        print(f"\n── {t} ──")
        active_stable = reg.get("active", {}).get("stable")
        active_dev = reg.get("active", {}).get("dev")
        bundles = reg.get("bundles", [])
        if not bundles:
            print("  (no bundles installed)")
            continue
        header = f"  {'bundle_id':<40} {'version':<10} {'channel':<8} {'created_at':<22}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for b in sorted(bundles, key=lambda x: x.get("created_at", "")):
            bid = b.get("bundle_id", "?")
            flags = []
            if bid == active_stable:
                flags.append("stable●")
            if bid == active_dev:
                flags.append("dev●")
            flag_str = " ".join(flags)
            print(f"  {bid:<40} {b.get('version',''):<10} {b.get('channel',''):<8} {b.get('created_at',''):<22}  {flag_str}")
        print(f"  active stable: {active_stable or '—'}")
        print(f"  active dev:    {active_dev or '—'}")


# ──────────────────────────────────────────────────────────────────────────────
# promote
# ──────────────────────────────────────────────────────────────────────────────

def models_promote(cfg: LakeConfig, *, bundle_id: str, task: str, channel: str) -> None:
    """Set *bundle_id* as the active bundle for *channel* (``stable`` or ``dev``)."""
    if channel not in ("stable", "dev"):
        raise ValueError(f"channel must be 'stable' or 'dev', got: {channel!r}")

    reg = _load_registry(cfg, task)
    known_ids = {b["bundle_id"] for b in reg.get("bundles", [])}
    if bundle_id not in known_ids:
        raise ValueError(f"Bundle '{bundle_id}' not found in {task} registry. Install it first.")

    reg.setdefault("active", {})
    reg["active"][channel] = bundle_id
    # Also update the 'channel' field on the bundle entry
    for b in reg["bundles"]:
        if b["bundle_id"] == bundle_id:
            b["channel"] = channel
    _save_registry(cfg, task, reg)
    print(f"Promoted '{bundle_id}' to {task} {channel} channel.")
