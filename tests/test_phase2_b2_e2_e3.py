"""Tests for Phase 2 tasks B2, E2, and E3."""
import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── B2: compatible_layouts in manifest ────────────────────────────────────────


class TestBundleCompatibleLayouts:
    """create_bundle() should write compatible_layouts into manifest.json."""

    def _run_create_bundle(self, tmp_path, compatible_layouts=None):
        """Call create_bundle with heavy ops mocked; return the written manifest."""
        from moldvision.bundle import create_bundle
        from moldvision.export import ExportResult

        ds = tmp_path / "ds"
        ds.mkdir()
        (ds / "metadata.json").write_text(json.dumps({
            "name": "test-ds",
            "class_names": ["A"],
        }))
        (ds / "models").mkdir()

        wt = tmp_path / "weights.pth"
        wt.write_bytes(b"\x00" * 16)

        out = tmp_path / "bundle_out"

        def fake_export_onnx(**kw):
            # Write a dummy ONNX file so the bundle sees it
            p = Path(kw["output"])
            p.write_bytes(b"ONNX")
            return ExportResult(ok=True, output_path=p, message="ok")

        with patch("moldvision.bundle.export_onnx", side_effect=fake_export_onnx):
            kwargs = dict(
                dataset_dir=ds,
                weights=wt,
                task="detect",
                size="nano",
                output_dir=out,
                height=640,
                width=640,
                exports=[],
                device="cpu",
                opset=18,
                dynamic_onnx=False,
                use_checkpoint_model=False,
                checkpoint_key=None,
                strict=False,
                fp16=False,
                workspace_mb=512,
                portable_checkpoint=False,
                allow_raw_checkpoint_fallback=True,
                include_raw_checkpoint=False,
                make_zip=False,
                make_mpk=False,
                overwrite=True,
            )
            if compatible_layouts is not None:
                kwargs["compatible_layouts"] = compatible_layouts

            res = create_bundle(**kwargs)
            assert res.ok, res.message

        manifest = json.loads((res.bundle_dir / "manifest.json").read_text())
        return manifest

    def test_default_layouts(self, tmp_path):
        manifest = self._run_create_bundle(tmp_path)
        assert manifest["compatible_layouts"] == ["*"]

    def test_custom_layouts(self, tmp_path):
        manifest = self._run_create_bundle(tmp_path, compatible_layouts=["layout_A", "layout_B"])
        assert manifest["compatible_layouts"] == ["layout_A", "layout_B"]


# ── E2: Model role resolution ────────────────────────────────────────────────


class TestModelRoles:
    def test_known_roles(self):
        from moldvision.lake import MODEL_ROLES, resolve_role_directory

        assert resolve_role_directory("defect_detector") == "models/defect_detection"
        assert resolve_role_directory("monitor_segmenter") == "models/monitor_segmentation"
        assert len(MODEL_ROLES) >= 2

    def test_unknown_role_raises(self):
        from moldvision.lake import resolve_role_directory

        with pytest.raises(ValueError, match="Unknown model role"):
            resolve_role_directory("nonexistent_role")


# ── E3: --publish flag triggers publish after bundle creation ─────────────────


class TestPublishAfterBundle:
    """handle_bundle() with --publish should call publish_bundle."""

    @patch("moldvision.cli_handlers.create_bundle")
    @patch("moldvision.publish.publish_bundle")
    def test_publish_flag_triggers_publish(self, mock_publish, mock_create, tmp_path):
        from moldvision.cli_handlers import handle_bundle
        from moldvision.bundle import BundleResult

        bundle_dir = tmp_path / "mybundle"
        bundle_dir.mkdir()
        mock_create.return_value = BundleResult(ok=True, bundle_dir=bundle_dir, message="ok")
        mock_publish.return_value = {"bundle_id": "test-v1", "role": "defect_detector"}

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            weights=str(tmp_path / "w.pth"),
            output_dir=None,
            export=None,
            task="detect",
            size="nano",
            height=640,
            width=640,
            device="cpu",
            opset=18,
            dynamic=False,
            use_checkpoint_model=False,
            checkpoint_key=None,
            strict=False,
            fp16=False,
            workspace_mb=512,
            portable_checkpoint=False,
            allow_raw_checkpoint_fallback=False,
            include_raw_checkpoint=False,
            zip=False,
            mpk=False,
            overwrite=False,
            quantize=False,
            calibration_split="valid",
            calibration_count=100,
            bundle_id=None,
            model_name=None,
            model_version="1.0.0",
            channel="stable",
            supersedes=None,
            min_app_version="0.0.0",
            standalone=False,
            publish=True,
            publish_role="defect_detector",
            publish_dry_run=False,
        )

        rc = handle_bundle(args)
        assert rc == 0
        mock_publish.assert_called_once_with(
            bundle_dir,
            role="defect_detector",
            channel="stable",
            dry_run=False,
        )

    @patch("moldvision.cli_handlers.create_bundle")
    def test_no_publish_flag_skips_publish(self, mock_create, tmp_path):
        from moldvision.cli_handlers import handle_bundle
        from moldvision.bundle import BundleResult

        mock_create.return_value = BundleResult(ok=True, bundle_dir=tmp_path, message="ok")

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            weights=str(tmp_path / "w.pth"),
            output_dir=None,
            export=None,
            task="detect",
            size="nano",
            height=640,
            width=640,
            device="cpu",
            opset=18,
            dynamic=False,
            use_checkpoint_model=False,
            checkpoint_key=None,
            strict=False,
            fp16=False,
            workspace_mb=512,
            portable_checkpoint=False,
            allow_raw_checkpoint_fallback=False,
            include_raw_checkpoint=False,
            zip=False,
            mpk=False,
            overwrite=False,
            quantize=False,
            calibration_split="valid",
            calibration_count=100,
            bundle_id=None,
            model_name=None,
            model_version="1.0.0",
            channel="stable",
            supersedes=None,
            min_app_version="0.0.0",
            standalone=False,
            publish=False,
        )

        rc = handle_bundle(args)
        assert rc == 0
        # publish module should never have been invoked

    @patch("moldvision.cli_handlers.create_bundle")
    def test_publish_without_role_fails(self, mock_create, tmp_path):
        from moldvision.cli_handlers import handle_bundle
        from moldvision.bundle import BundleResult

        mock_create.return_value = BundleResult(ok=True, bundle_dir=tmp_path, message="ok")

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            weights=str(tmp_path / "w.pth"),
            output_dir=None,
            export=None,
            task="detect",
            size="nano",
            height=640,
            width=640,
            device="cpu",
            opset=18,
            dynamic=False,
            use_checkpoint_model=False,
            checkpoint_key=None,
            strict=False,
            fp16=False,
            workspace_mb=512,
            portable_checkpoint=False,
            allow_raw_checkpoint_fallback=False,
            include_raw_checkpoint=False,
            zip=False,
            mpk=False,
            overwrite=False,
            quantize=False,
            calibration_split="valid",
            calibration_count=100,
            bundle_id=None,
            model_name=None,
            model_version="1.0.0",
            channel="stable",
            supersedes=None,
            min_app_version="0.0.0",
            standalone=False,
            publish=True,
            publish_role=None,
            publish_dry_run=False,
        )

        rc = handle_bundle(args)
        assert rc == 2  # should fail because role is required
