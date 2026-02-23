from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .checkpoints import load_checkpoint_weights
from .datasets import load_metadata


@dataclass(frozen=True)
class ExportResult:
    ok: bool
    output_path: Optional[Path] = None
    message: str = ""


def _instantiate_model(task: str, size: str, num_classes: Optional[int], pretrain_weights: Optional[str] = None):
    task = (task or "detect").lower().strip()
    if task == "seg":
        from rfdetr import RFDETRSegPreview  # type: ignore

        kwargs: Dict[str, Any] = {}
        if pretrain_weights:
            kwargs["pretrain_weights"] = pretrain_weights
        try:
            if num_classes is not None:
                kwargs["num_classes"] = int(num_classes)
            return RFDETRSegPreview(**kwargs) if kwargs else RFDETRSegPreview()
        except TypeError:
            return RFDETRSegPreview()

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium  # type: ignore

    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase, "medium": RFDETRMedium}.get(size)
    if cls is None:
        raise ValueError(f"Unknown model size: {size}")

    kwargs: Dict[str, Any] = {}
    if pretrain_weights:
        kwargs["pretrain_weights"] = pretrain_weights
    if num_classes is not None:
        kwargs["num_classes"] = int(num_classes)
    try:
        return cls(**kwargs) if kwargs else cls()
    except TypeError:
        return cls()


def _find_torch_module(model: object):
    import torch
    import torch.nn as nn

    if isinstance(model, nn.Module):
        return model

    for attr in ("model", "net", "network", "module", "detector", "backbone", "transformer"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        if isinstance(inner, nn.Module):
            return inner

    # best-effort: search one level deep through __dict__
    try:
        for _, val in vars(model).items():
            if isinstance(val, nn.Module):
                return val
    except Exception:
        pass

    raise TypeError(f"Could not find a torch.nn.Module inside model object of type {type(model)}")


def _extract_outputs(out: object, *, want_masks: bool) -> Tuple[Any, ...]:
    """Convert model forward outputs into a tuple of tensors suitable for ONNX export."""
    import torch

    def _as_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return None

    # dict-like
    if isinstance(out, dict):
        logits = out.get("pred_logits") or out.get("logits")
        boxes = out.get("pred_boxes") or out.get("boxes")
        masks = out.get("pred_masks") or out.get("masks") or out.get("mask")
        t_logits = _as_tensor(logits)
        t_boxes = _as_tensor(boxes)
        t_masks = _as_tensor(masks)
        if t_logits is not None and t_boxes is not None:
            if want_masks and t_masks is not None:
                return (t_logits, t_boxes, t_masks)
            return (t_logits, t_boxes)

    # object with attributes
    for logits_attr, boxes_attr, masks_attr in (
        ("pred_logits", "pred_boxes", "pred_masks"),
        ("logits", "boxes", "masks"),
    ):
        if hasattr(out, logits_attr) and hasattr(out, boxes_attr):
            t_logits = _as_tensor(getattr(out, logits_attr))
            t_boxes = _as_tensor(getattr(out, boxes_attr))
            if t_logits is not None and t_boxes is not None:
                if want_masks and hasattr(out, masks_attr):
                    t_masks = _as_tensor(getattr(out, masks_attr))
                    if t_masks is not None:
                        return (t_logits, t_boxes, t_masks)
                return (t_logits, t_boxes)

    # tuple/list of tensors
    if isinstance(out, (tuple, list)):
        tensors = tuple(x for x in out if _as_tensor(x) is not None)
        if len(tensors) >= (3 if want_masks else 2):
            return tensors[: (3 if want_masks else 2)]

    raise TypeError(
        "Model forward output is not exportable to ONNX (expected tensors or a dict/object with pred_logits/pred_boxes[/pred_masks]). "
        f"Got: {type(out)}"
    )


def export_onnx(
    *,
    dataset_dir: Path,
    weights: Path,
    task: str,
    size: str,
    output: Optional[Path],
    device: Optional[str],
    height: int,
    width: int,
    opset: int,
    dynamic: bool,
    use_checkpoint_model: bool,
    checkpoint_key: Optional[str],
    strict: bool,
) -> ExportResult:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        return ExportResult(False, None, f"PyTorch not available: {e}")

    dataset_dir = dataset_dir.expanduser().resolve()
    weights = weights.expanduser().resolve()

    md = load_metadata(dataset_dir)
    class_names = md.get("class_names", []) or []
    num_classes = len(class_names) if class_names else None

    try:
        model = _instantiate_model(task, size, num_classes)
    except Exception as e:
        return ExportResult(False, None, f"Failed to instantiate model: {e}")

    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    lr = load_checkpoint_weights(
        model,
        str(weights),
        torch_device,
        checkpoint_key=checkpoint_key,
        allow_replace_model=bool(use_checkpoint_model),
        strict=bool(strict),
        verbose=True,
    )
    if not lr.ok and lr.replacement_model is None:
        return ExportResult(False, None, f"Failed to load weights: {lr.message}")
    if lr.replacement_model is not None:
        model = lr.replacement_model

    module = _find_torch_module(model)
    try:
        module.to(torch_device)
    except Exception:
        pass
    module.eval()

    want_masks = (task or "").lower().strip() == "seg"

    class OnnxWrapper(nn.Module):
        def __init__(self, inner: nn.Module, want_masks: bool):
            super().__init__()
            self.inner = inner
            self.want_masks = want_masks

        def forward(self, images: torch.Tensor):  # type: ignore[override]
            out = self.inner(images)
            return _extract_outputs(out, want_masks=self.want_masks)

    wrapper = OnnxWrapper(module, want_masks=want_masks).to(torch_device).eval()

    dummy = torch.randn(1, 3, int(height), int(width), device=torch_device)

    # output paths
    if output is None:
        out_dir = dataset_dir / "exports" / "onnx"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / ("model_seg.onnx" if want_masks else "model_detect.onnx")
    else:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["images"]
    output_names = ["pred_logits", "pred_boxes"] + (["pred_masks"] if want_masks else [])

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "pred_logits": {0: "batch", 1: "num_queries"},
            "pred_boxes": {0: "batch", 1: "num_queries"},
        }
        if want_masks:
            dynamic_axes["pred_masks"] = {0: "batch", 1: "num_queries", 2: "mask_h", 3: "mask_w"}

    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            export_params=True,
            opset_version=int(opset),
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except Exception as e:
        return ExportResult(False, None, f"torch.onnx.export failed: {e}")

    return ExportResult(True, output, f"Wrote ONNX: {output}")


def export_tensorrt_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Optional[Path],
    height: int,
    width: int,
    fp16: bool,
    workspace_mb: int,
) -> ExportResult:
    """Build a TensorRT engine by shelling out to `trtexec` if available.

    This keeps TensorRT as an optional deployment tool (no hard dependency).
    """
    onnx_path = onnx_path.expanduser().resolve()
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    else:
        engine_path = engine_path.expanduser().resolve()
        engine_path.parent.mkdir(parents=True, exist_ok=True)

    trtexec = shutil.which("trtexec")
    if not trtexec:
        return ExportResult(
            False,
            None,
            "TensorRT export requires `trtexec` on PATH. Install TensorRT and ensure `trtexec` is available.",
        )

    shapes = f"images:1x3x{int(height)}x{int(width)}"
    cmd = [
        trtexec,
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(engine_path)}",
        "--explicitBatch",
        f"--shapes={shapes}",
        f"--workspace={int(workspace_mb)}",
    ]
    if fp16:
        cmd.append("--fp16")

    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as e:
        return ExportResult(False, None, f"Failed to run trtexec: {e}")

    if proc.returncode != 0:
        return ExportResult(False, None, f"trtexec failed with exit code {proc.returncode}")

    if not engine_path.exists():
        return ExportResult(False, None, f"Engine was not created: {engine_path}")

    return ExportResult(True, engine_path, f"Wrote TensorRT engine: {engine_path}")
