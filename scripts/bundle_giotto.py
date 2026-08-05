#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper utility to bundle the Giotto startup suggestion models.
Converts the scikit-learn Lasso pipelines to ONNX formats, trains
a composite quality score pipeline on synthetic data, creates the suggestion bundle,
and publishes it under the shared folder while updating the index catalog.
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Install-time imports (from .[predictive])
try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("ERROR: skl2onnx and onnxmltools must be installed in the environment.")
    print("Run: uv pip install -e .[predictive] first.")
    sys.exit(1)

def sha256_file(filepath: Path) -> str:
    """Compute the SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def pack_zip_archive(src_dir: Path, output_zip: Path) -> None:
    """Pack a directory contents into a ZIP archive."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(src_dir))

def main():
    # 1. Resolve paths
    src_dir = Path("C:/Users/aria-/Downloads/models_giotto_09_07_26")
    model_info_path = src_dir / "model_info.json"
    
    if not model_info_path.exists():
        print(f"ERROR: model_info.json not found at {model_info_path}")
        sys.exit(1)
        
    print(f"Loading metadata from {model_info_path} ...")
    with open(model_info_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    feature_cols = meta["feature_cols"]
    print(f"Features: {feature_cols}")
    
    # 2. Reconstruct scikit-learn pipelines for each target
    pipelines = {}
    for target_name, target_cfg in meta["targets"].items():
        print(f"Reconstructing pipeline for target: {target_name} ...")
        scaler_path = src_dir / target_cfg["x_scaler_file"].replace("models_giotto_09_07_26\\", "")
        poly_path = src_dir / target_cfg["poly_features_file"].replace("models_giotto_09_07_26\\", "")
        model_path = src_dir / target_cfg["model_file"].replace("models_giotto_09_07_26\\", "")
        
        scaler = joblib.load(scaler_path)
        poly = joblib.load(poly_path)
        model = joblib.load(model_path)
        
        # Build pipeline
        pipe = Pipeline([
            ("scaler", scaler),
            ("poly", poly),
            ("regressor", model)
        ])
        pipelines[target_name] = pipe

    # 3. Create synthetic data and train the composite quality score model
    print("Generating synthetic data to train composite quality score model ...")
    np.random.seed(42)
    n_samples = 2000
    X_synthetic = []
    for col in feature_cols:
        col_min = meta["x_min_max"][col]["min"]
        col_max = meta["x_min_max"][col]["max"]
        # Uniform sampling within range
        X_synthetic.append(np.random.uniform(col_min, col_max, n_samples))
        
    X_synthetic = np.stack(X_synthetic, axis=1) # shape (2000, 4)
    
    # Predict the target severity values
    y_weld = pipelines["weld_line"].predict(X_synthetic)
    y_sink = pipelines["sink_mark"].predict(X_synthetic)
    y_flash = pipelines["flash"].predict(X_synthetic)
    
    # Quality Score formula: 1.0 - (0.15 * weld_line + 0.35 * sink_mark + 0.30 * flash)
    # Clamp results to [0.0, 1.0]
    y_quality = np.clip(1.0 - (0.15 * y_weld + 0.35 * y_sink + 0.30 * y_flash), 0.0, 1.0)
    
    print("Training quality score pipeline on synthetic targets ...")
    qs_scaler = StandardScaler()
    qs_poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    qs_regressor = LinearRegression()
    
    qs_pipeline = Pipeline([
        ("scaler", qs_scaler),
        ("poly", qs_poly),
        ("regressor", qs_regressor)
    ])
    qs_pipeline.fit(X_synthetic, y_quality)
    
    # Check synthetic training accuracy
    r2 = qs_pipeline.score(X_synthetic, y_quality)
    print(f"  Quality score pipeline R^2 on synthetic data: {r2:.6f}")
    pipelines["quality_score"] = qs_pipeline

    # 4. Create output directory for bundle
    bundle_id = "mold-giotto-startup-suggestion-v1.0.0"
    output_dir = Path("C:/Users/aria-/dev/ARIA_MoldVision/runs") / bundle_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX models to local bundle dir {output_dir} ...")
    
    # Convert and write each pipeline to ONNX
    onnx_filenames = {}
    for target_name, pipe in pipelines.items():
        print(f"  Converting {target_name} to ONNX ...")
        # Define float input of shape [None, 4]
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onnx_model = to_onnx(pipe, initial_types=initial_type, target_opset=12)
        
        filename = f"model_quality_score.onnx" if target_name == "quality_score" else f"model_defect_{target_name}.onnx"
        onnx_path = output_dir / filename
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        onnx_filenames[target_name] = filename
        print(f"    Saved: {onnx_path}")

    # 5. Write training_meta.json
    meta_json = {
        "dataset_name": meta["data_source"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_feature_keys": feature_cols,
        "metrics": {
            "weld_line_train_r2": float(pipelines["weld_line"].score(X_synthetic, y_weld)),
            "sink_mark_train_r2": float(pipelines["sink_mark"].score(X_synthetic, y_sink)),
            "flash_train_r2": float(pipelines["flash"].score(X_synthetic, y_flash)),
            "quality_score_r2": float(r2)
        }
    }
    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, ensure_ascii=False)

    # 6. Write manifest.json
    checksums = {
        meta_path.name: sha256_file(meta_path)
    }
    for target_name, filename in onnx_filenames.items():
        checksums[filename] = sha256_file(output_dir / filename)
        
    manifest = {
        "bundle_type": "startup_suggestion",
        "bundle_id": bundle_id,
        "model_name": "Mold Giotto Startup Suggestion",
        "model_version": "1.0.0",
        "schema_version": 3,
        "channel": "stable",
        "supersedes": None,
        "min_app_version": "0.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_keys": feature_cols,
        "context_feature_keys": [],
        "trained_feature_keys": feature_cols,
        "parameter_schema": [
            {
                "parameter_id": "t_melt",
                "display_name": "Melt Temperature",
                "unit": "degC",
                "baseline": 247.5,
                "range_min": 235.0,
                "range_max": 260.0,
                "control_feature_keys": ["t_melt"],
                "family_id": "t_melt",
                "semantic_parameter_id": "melt_temperature",
                "step_mode": "absolute",
                "preferred_step": 2.0,
                "max_delta": 5.0,
                "observed_support_min": 235.0,
                "observed_support_max": 260.0,
                "support_margin_ratio": 0.0,
                "decimal_places": 0
            },
            {
                "parameter_id": "t_mold",
                "display_name": "Mold Temperature",
                "unit": "degC",
                "baseline": 72.5,
                "range_min": 60.0,
                "range_max": 85.0,
                "control_feature_keys": ["t_mold"],
                "family_id": "t_mold",
                "semantic_parameter_id": "mold_temperature",
                "step_mode": "absolute",
                "preferred_step": 2.0,
                "max_delta": 5.0,
                "observed_support_min": 60.0,
                "observed_support_max": 85.0,
                "support_margin_ratio": 0.0,
                "decimal_places": 0
            },
            {
                "parameter_id": "inj_speed",
                "display_name": "Injection Speed",
                "unit": "mm/s",
                "baseline": 55.0,
                "range_min": 35.0,
                "range_max": 75.0,
                "control_feature_keys": ["inj_speed"],
                "family_id": "inj_speed",
                "semantic_parameter_id": "injection_speed",
                "step_mode": "absolute",
                "preferred_step": 5.0,
                "max_delta": 10.0,
                "observed_support_min": 35.0,
                "observed_support_max": 75.0,
                "support_margin_ratio": 0.0,
                "decimal_places": 0
            },
            {
                "parameter_id": "pack_pressure",
                "display_name": "Pack Pressure",
                "unit": "bar",
                "baseline": 812.5,
                "range_min": 650.0,
                "range_max": 975.0,
                "control_feature_keys": ["pack_pressure"],
                "family_id": "pack_pressure",
                "semantic_parameter_id": "hold_pressure",
                "step_mode": "absolute",
                "preferred_step": 50.0,
                "max_delta": 100.0,
                "observed_support_min": 650.0,
                "observed_support_max": 975.0,
                "support_margin_ratio": 0.0,
                "decimal_places": 0
            }
        ],
        "control_families": [
            {
                "family_id": "t_melt",
                "display_name": "Melt Temperature",
                "family_type": "single_slot",
                "semantic_parameter_id": "melt_temperature",
                "ordered_members": [{"parameter_id": "t_melt"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["t_melt"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            },
            {
                "family_id": "t_mold",
                "display_name": "Mold Temperature",
                "family_type": "single_slot",
                "semantic_parameter_id": "mold_temperature",
                "ordered_members": [{"parameter_id": "t_mold"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["t_mold"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            },
            {
                "family_id": "inj_speed",
                "display_name": "Injection Speed",
                "family_type": "single_slot",
                "semantic_parameter_id": "injection_speed",
                "ordered_members": [{"parameter_id": "inj_speed"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["inj_speed"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            },
            {
                "family_id": "pack_pressure",
                "display_name": "Pack Pressure",
                "family_type": "single_slot",
                "semantic_parameter_id": "hold_pressure",
                "ordered_members": [{"parameter_id": "pack_pressure"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["pack_pressure"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            }
        ],
        "deployable_control_families": [
            {
                "family_id": "t_melt",
                "display_name": "Melt Temperature",
                "family_type": "single_slot",
                "semantic_parameter_id": "melt_temperature",
                "ordered_members": [{"parameter_id": "t_melt"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["t_melt"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            },
            {
                "family_id": "t_mold",
                "display_name": "Mold Temperature",
                "family_type": "single_slot",
                "semantic_parameter_id": "mold_temperature",
                "ordered_members": [{"parameter_id": "t_mold"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["t_mold"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            },
            {
                "family_id": "inj_speed",
                "display_name": "Injection Speed",
                "family_type": "single_slot",
                "semantic_parameter_id": "injection_speed",
                "ordered_members": [{"parameter_id": "inj_speed"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["inj_speed"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            },
            {
                "family_id": "pack_pressure",
                "display_name": "Pack Pressure",
                "family_type": "single_slot",
                "semantic_parameter_id": "hold_pressure",
                "ordered_members": [{"parameter_id": "pack_pressure"}],
                "family_constraints": {
                    "ordered_slots": False,
                    "dynamic_activation": False,
                    "activation_semantics": "single",
                    "shape_preserving_only": False,
                    "allowed_recipe_types": ["single_step"],
                    "controllable_member_parameter_ids": ["pack_pressure"]
                },
                "observed_activation_states": ["active"],
                "observed_observability_states": ["observed"]
            }
        ],
        "imputation_values": {
            "t_melt": 247.5,
            "t_mold": 72.5,
            "inj_speed": 55.0,
            "pack_pressure": 812.5
        },
        "null_strategy": "mean_impute",
        "selected_feature_stats": ["setpoint"],
        "target_models": {
            "quality_score": {
                "filename": "model_quality_score.onnx",
                "model_type": "regression",
                "source_target": "quality_score",
                "signal_kind": "quality_score",
                "signal_role": "quality"
            },
            "defect_sink_mark": {
                "filename": "model_defect_sink_mark.onnx",
                "model_type": "regression",
                "source_target": "Sink_Mark_severity",
                "signal_kind": "duration_ratio",
                "signal_role": "optimization",
                "y_min": float(meta["y_min_max"]["sink_mark"]["min"]),
                "y_max": float(meta["y_min_max"]["sink_mark"]["max"])
            },
            "defect_weld_line": {
                "filename": "model_defect_weld_line.onnx",
                "model_type": "regression",
                "source_target": "Weld_Line_severity",
                "signal_kind": "duration_ratio",
                "signal_role": "optimization",
                "y_min": float(meta["y_min_max"]["weld_line"]["min"]),
                "y_max": float(meta["y_min_max"]["weld_line"]["max"])
            },
            "defect_flash": {
                "filename": "model_defect_flash.onnx",
                "model_type": "regression",
                "source_target": "Flash_severity",
                "signal_kind": "duration_ratio",
                "signal_role": "optimization",
                "y_min": float(meta["y_min_max"]["flash"]["min"]),
                "y_max": float(meta["y_min_max"]["flash"]["max"])
            }
        },
        "quality_weights": {
            "burn_mark": 0.2,
            "flash": 0.3,
            "sink_mark": 0.35,
            "weld_line": 0.15
        },
        "defect_display_labels": {
            "sink_mark": "Sink Mark",
            "weld_line": "Weld Line",
            "flash": "Flash"
        },
        "default_threshold_by_defect": {
            "sink_mark": 0.1,
            "weld_line": 0.1,
            "flash": 0.1
        },
        "checksums": checksums,
        "scope": {
            "machine_id": "wittman_plast",
            "material_id": "abs",
            "mold_id": "giotto"
        },
        "machine_name": "wittman_plast",
        "material_name": "abs",
        "mold_name": "giotto",
        "compatible_layouts": ["*"],
        "source_system": "TestWittmann"
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Written manifest to: {manifest_path}")

    # 7. Create .sugbundle file in local runs folder
    archive_path = output_dir.parent / f"{bundle_id}.sugbundle"
    print(f"Packing ZIP archive to: {archive_path} ...")
    pack_zip_archive(output_dir, archive_path)
    print("Archive pack completed successfully.")

    # 8. Copy/Publish bundle directory and .sugbundle file to shared folder
    shared_bundles_dir = Path("C:/Users/aria-/dev/shared/published/moldpilot/suggestions/bundles")
    shared_dest_dir = shared_bundles_dir / bundle_id
    shared_dest_archive = shared_bundles_dir / f"{bundle_id}.sugbundle"
    
    print(f"Publishing/Copying bundle to shared directory: {shared_dest_dir} ...")
    if shared_dest_dir.exists():
        print(f"  Destination {shared_dest_dir} exists, removing it first...")
        shutil.rmtree(shared_dest_dir)
    shared_dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, shared_dest_dir)
    
    print(f"Copying archive to: {shared_dest_archive} ...")
    shutil.copy2(archive_path, shared_dest_archive)
    print("Copy completed successfully.")
    
    # 9. Update the suggestions index.json file
    index_path = Path("C:/Users/aria-/dev/shared/published/moldpilot/suggestions/index.json")
    if index_path.exists():
        print(f"Updating index file: {index_path} ...")
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            
        # Build the new catalog entry
        catalog_entry = {
            "bundle_id": bundle_id,
            "model_name": "Mold Giotto Startup Suggestion",
            "model_version": "1.0.0",
            "channel": "stable",
            "role": "startup_suggestion",
            "min_app_version": "0.0.0",
            "artifact_key": f"bundles/{bundle_id}/",
            "sha256": sha256_file(shared_dest_archive),
            "size_bytes": shared_dest_archive.stat().st_size,
            "compatible_layouts": ["*"],
            "supersedes": None,
            "published_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Replace the existing entry if one exists with the same bundle_id
        bundles = index_data.get("bundles", [])
        updated_bundles = [b for b in bundles if b.get("bundle_id") != bundle_id]
        updated_bundles.append(catalog_entry)
        index_data["bundles"] = updated_bundles
        
        # Set this bundle as active for stable channel
        active_by_channel = index_data.get("active_by_channel", {})
        active_by_channel["stable"] = bundle_id
        index_data["active_by_channel"] = active_by_channel
        index_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print("index.json updated successfully.")
    else:
        print(f"WARNING: suggestions index.json not found at {index_path}. Skipping catalog index update.")

if __name__ == "__main__":
    main()
