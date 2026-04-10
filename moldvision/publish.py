"""ARIA MoldVision — Model Bundle Publishing.

Upload trained model bundles to S3 and update the central catalog.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

CATALOG_SCHEMA_VERSION = "catalog-v1"
CATALOG_KEY = "catalog.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(bundle_path: Path) -> Dict[str, Any]:
    """Read manifest.json from a bundle directory or .mpk/.zip file."""
    if bundle_path.is_dir():
        for name in ("manifest.json", "bundle_manifest.json"):
            p = bundle_path / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No manifest found in {bundle_path}")
    # Archive
    import zipfile
    with zipfile.ZipFile(bundle_path, "r") as zf:
        for name in ("manifest.json", "bundle_manifest.json"):
            if name in zf.namelist():
                return json.loads(zf.read(name).decode("utf-8"))
        raise FileNotFoundError(f"No manifest found in {bundle_path}")


def _ensure_mpk(bundle_path: Path) -> Path:
    """Ensure we have an .mpk (zip) file. If input is a directory, pack it."""
    if bundle_path.is_file() and bundle_path.suffix in (".mpk", ".zip"):
        return bundle_path
    if not bundle_path.is_dir():
        raise ValueError(f"Expected a directory or .mpk/.zip file: {bundle_path}")
    import zipfile
    mpk_path = bundle_path.with_suffix(".mpk")
    with zipfile.ZipFile(mpk_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(bundle_path.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_path))
    _log.info("Packed bundle directory to %s", mpk_path)
    return mpk_path


def _load_publish_config() -> Dict[str, Any]:
    """Load publish config from data_lake_config.json or environment."""
    # Check environment variables first
    bucket = os.environ.get("ARIA_MODEL_CATALOG_BUCKET")
    region = os.environ.get("ARIA_MODEL_CATALOG_REGION", "eu-west-1")
    prefix = os.environ.get("ARIA_MODEL_CATALOG_PREFIX", "")
    if bucket:
        return {"bucket": bucket, "region": region, "prefix": prefix}

    # Try data_lake_config.json in common locations
    for candidate in [
        Path("data_lake_config.json"),
        Path.home() / ".aria" / "data_lake_config.json",
    ]:
        if candidate.exists():
            cfg = json.loads(candidate.read_text(encoding="utf-8"))
            if "model_catalog_bucket" in cfg:
                return {
                    "bucket": cfg["model_catalog_bucket"],
                    "region": cfg.get("model_catalog_region", "eu-west-1"),
                    "prefix": cfg.get("model_catalog_prefix", ""),
                }

    raise RuntimeError(
        "No publish configuration found. Set ARIA_MODEL_CATALOG_BUCKET env var "
        "or add 'model_catalog_bucket' to data_lake_config.json."
    )


def _s3_client(region: str):
    """Create a boto3 S3 client."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for publishing. Install with: pip install boto3")
    return boto3.client("s3", region_name=region)


def _fetch_catalog(s3, bucket: str, prefix: str) -> Dict[str, Any]:
    """Fetch the current catalog.json from S3, or return an empty catalog."""
    key = f"{prefix}{CATALOG_KEY}" if prefix else CATALOG_KEY
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e):
            _log.info("No existing catalog found, creating new one.")
            return {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "updated_at": None,
                "models": [],
            }
        raise


def _upload_catalog(s3, bucket: str, prefix: str, catalog: Dict[str, Any]) -> None:
    """Write catalog.json back to S3."""
    key = f"{prefix}{CATALOG_KEY}" if prefix else CATALOG_KEY
    body = json.dumps(catalog, indent=2, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    _log.info("Updated catalog at s3://%s/%s", bucket, key)


def publish_bundle(
    bundle_path: Path,
    *,
    role: str,
    channel: str = "stable",
    compatible_layouts: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Publish a model bundle to the S3 model catalog.

    Args:
        bundle_path: Path to a bundle directory or .mpk/.zip archive.
        role: Model role (e.g., "defect_detector", "monitor_segmenter").
        channel: Release channel ("stable" or "beta").
        compatible_layouts: HMI layouts this model supports. None = ["*"].
        dry_run: If True, print what would be done without uploading.

    Returns:
        Dict with published bundle metadata.
    """
    bundle_path = Path(bundle_path)
    manifest = _read_manifest(bundle_path)

    bundle_id = manifest.get("bundle_id") or manifest.get("model_name", "unknown")
    model_version = manifest.get("model_version", "0.0.0")
    supersedes = manifest.get("supersedes")
    min_app_version = manifest.get("min_app_version", "0.0.0")

    mpk_path = _ensure_mpk(bundle_path)
    sha256 = _sha256_file(mpk_path)
    size_bytes = mpk_path.stat().st_size

    artifact_key_name = f"bundles/{bundle_id}.mpk"
    layouts = compatible_layouts or ["*"]

    catalog_entry = {
        "bundle_id": bundle_id,
        "model_name": manifest.get("model_name", bundle_id),
        "model_version": model_version,
        "channel": channel,
        "role": role,
        "min_app_version": min_app_version,
        "artifact_key": artifact_key_name,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "compatible_layouts": layouts,
        "supersedes": supersedes,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        _log.info("DRY RUN — would publish:\n%s", json.dumps(catalog_entry, indent=2))
        return catalog_entry

    config = _load_publish_config()
    s3 = _s3_client(config["region"])
    bucket = config["bucket"]
    prefix = config.get("prefix", "")

    # Upload .mpk
    artifact_key = f"{prefix}{artifact_key_name}" if prefix else artifact_key_name
    _log.info("Uploading %s to s3://%s/%s ...", mpk_path.name, bucket, artifact_key)
    s3.upload_file(str(mpk_path), bucket, artifact_key)
    _log.info("Upload complete (%d bytes, sha256=%s)", size_bytes, sha256[:16])

    # Update catalog
    catalog = _fetch_catalog(s3, bucket, prefix)
    # Remove any existing entry with same bundle_id
    catalog["models"] = [
        m for m in catalog.get("models", []) if m.get("bundle_id") != bundle_id
    ]
    catalog["models"].append(catalog_entry)
    catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
    _upload_catalog(s3, bucket, prefix, catalog)

    _log.info("Published %s (v%s) to channel '%s', role '%s'", bundle_id, model_version, channel, role)
    return catalog_entry
