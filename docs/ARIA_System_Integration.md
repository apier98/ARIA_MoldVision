# ARIA System Integration — Cross-Project Context Document

> **Scope**: This document is the shared reference for the three software systems built by the ARIA startup.
> It is identical across `ARIA_MoldPilot/docs/`, `ARIA_MoldTrace/docs/`, and `ARIA_MoldVision/docs/`.
> Keep it in sync when any integration contract changes.

---

## 1. Mission & Product Vision

ARIA is building a suite of industrial AI tools to support **injection molding** process quality.
The core insight is that experienced (senior) operators possess a **mental mapping** between
machine process parameters (barrel temperature, injection pressure, holding time, etc.) and the
surface defects that appear on produced components (sink marks, weld lines, flash, burn marks).
ARIA captures, formalises, and operationalises that mapping so that **junior operators** can
benefit from it in real time.

The product is structured in three layers:

| Layer | Software | Role |
|-------|----------|------|
| **Edge — operator tool** | ARIA_MoldPilot | Desktop app on the shop-floor PC, drives the camera station |
| **Cloud — analysis pipeline** | ARIA_MoldTrace | AWS-hosted pipeline that processes recordings and builds labeled datasets |
| **Internal — model factory** | ARIA_MoldVision | Internal toolchain to train and export CV models consumed by the other two |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SHOP FLOOR                                                                 │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ARIA_MoldPilot  (Windows Desktop — PySide6)                         │  │
│  │                                                                      │  │
│  │  ┌─────────────────────┐   ┌──────────────────────────────────────┐  │  │
│  │  │  Qualification Mode │   │  Quality Monitoring Mode             │  │  │
│  │  │                     │   │                                      │  │  │
│  │  │  • Record component │   │  • Live ONNX inference on each frame │  │  │
│  │  │    video (H.264 MP4)│   │  • IoU tracking (components+defects) │  │  │
│  │  │  • Record HMI video │   │  • Severity metrics in real time     │  │  │
│  │  │  • Capture operator │   │  • Startup Assistant (ML suggestions)│  │  │
│  │  │    metadata & notes │   │  • Operator guidance via threshold   │  │  │
│  │  │  • Seal session     │   │    bars                              │  │  │
│  │  │    manifest (JSON)  │   └──────────────────────────────────────┘  │  │
│  │  └─────────────────────┘                                             │  │
│  │                                                                      │  │
│  │  Hardware: Baumer VCXG.2-32C (GigE) + Arduino/FUYU FSK40 Z-axis     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                      │ MP4 videos + session manifest                        │
│                      │ (S3 upload — not wired yet)                          │
└──────────────────────┼──────────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AWS CLOUD                                                                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ARIA_MoldTrace  (Python CLI / AWS service)                          │  │
│  │                                                                      │  │
│  │  ① extract_frames  ─── FFmpeg → JPEG frames                         │  │
│  │  ② extract_monitor ─── RF-DETR seg → HMI screen geometry            │  │
│  │  ③ monitor_quality ─── blur/overlay/occlusion flags                 │  │
│  │  ④ extract_params  ─── RapidOCR on ROIs → parameter JSONL           │  │
│  │  ⑤ audio_extract   ─── FLAC/WAV from process video                  │  │
│  │  ⑥ components      ─── RF-DETR det → component boxes               │  │
│  │  ⑦ defects         ─── RF-DETR det → defect boxes + class          │  │
│  │  ⑧ merge_components─── multi-frame IoU tracking                     │  │
│  │  ⑨ coupling        ─── align defects ↔ parameters → labeled dataset │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                      │ labeled datasets                                     │
│                      │ (component + defects + process params)               │
└──────────────────────┼──────────────────────────────────────────────────────┘
                       │
                       ▼  (used to build suggestion logic — future)
┌─────────────────────────────────────────────────────────────────────────────┐
│  INTERNAL TOOLCHAIN                                                         │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ARIA_MoldVision  (Python CLI — runs on dev/GPU workstation)         │  │
│  │                                                                      │  │
│  │  • Dataset management (UUID folders, COCO/YOLO ingestion)           │  │
│  │  • RF-DETR training (detect + seg, nano → 2xlarge)                  │  │
│  │  • ONNX / TensorRT / INT8-quantized export                          │  │
│  │  • Deployment bundle creation (.mpk format)                          │  │
│  │  • Label Studio ML backend for active learning                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│          │ .mpk bundles for MoldPilot      │ .mpk bundles for MoldTrace     │
└──────────┼─────────────────────────────────┼───────────────────────────────┘
           ▼                                 ▼
    MoldPilot model registry         MoldTrace model registry
    (monitoring mode inference)      (monitor_segmenter, components,
                                      ocr_recognizer roles)
```

---

## 3. ARIA_MoldPilot

### 3.1 Purpose

Desktop application installed on the shop-floor Windows PC. It is the **operator's primary interface**
for two tasks:

1. **Qualification Mode** — recording timed sessions that capture both a component-view video
   (showing the part coming out of the mold) and a process-view video (showing the machine HMI
   screen), together with operator-supplied metadata. Sessions are sealed as local manifests and
   enqueued for upload to S3 (upload not yet wired).

2. **Quality Monitoring Mode** — continuous live defect detection using a loaded ONNX model bundle.
   Components are tracked frame-by-frame and severity metrics (weld lines, sink marks, flash, burn
   marks) are charted in real time. The **Startup Assistant** sub-screen displays ML-suggested
   machine parameters to help junior operators tune a startup.

### 3.2 Tech Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.10+ |
| UI | PySide6 6.8+ (Qt6 Widgets) |
| Video encoding | PyAV (libx264 H.264, CRF-20) |
| Frame capture | Baumer neoAPI (proprietary GigE SDK) |
| Image processing | OpenCV 4.8+ |
| Inference runtime | ONNX Runtime 1.17 (CUDA / DirectML / CPU) |
| Live charting | PyQtGraph 0.13+ |
| Motion control | pyserial → Arduino ASCII protocol |
| Linting / tests | Ruff, pytest |

### 3.3 Architecture

Strict five-layer stack — UI never touches infrastructure directly:

```
ui/              ← PySide6 screens & widgets
application/     ← QObject controllers, workflow orchestration
services/        ← Protocol interfaces only (no implementations)
infrastructure/  ← Concrete: Baumer camera, ONNX, IoU tracker, local store…
domain/          ← Immutable frozen dataclasses, zero external dependencies
```

All wiring happens in `app.py` via explicit dependency injection. Mock services are available for
all hardware interfaces (`ARIA_MOTION_MOCK=1`, etc.), enabling full off-machine development.

### 3.4 Key Data Produced

| Artifact | Format | Location | Consumer |
|----------|--------|----------|----------|
| Component-view video | MP4 H.264, 60 s chunks | `sessions/<id>/*.mp4` | MoldTrace |
| Process-view (HMI) video | MP4 H.264, 60 s chunks | `sessions/<id>/*.mp4` | MoldTrace |
| Session manifest | JSON | `sessions/<id>.json` | MoldTrace |
| Monitoring component record | JSONL | `monitoring/<id>/components.jsonl` | Future analytics |
| Monitoring timeseries | JSONL | `monitoring/<id>/timeseries.jsonl` | Future analytics |

### 3.5 Session Manifest Schema (key fields)

```json
{
  "session_id":            "qual_20260324T093000Z_a1b2c3d4",
  "configuration_id":      "cfg_moldpilot_2026_001",
  "machine_id":            "machine_01",
  "mold_id":               "mold_a12",
  "part_id":               "part_cap_32",
  "started_at":            "2026-03-24T09:30:00+00:00",
  "ended_at":              "2026-03-24T09:45:12+00:00",
  "status":                "completed",
  "operator_name":         "Andrea Rossi",
  "batch_number":          "LOT-2026-0324-A",
  "operator_notes":        "Ambient temp 22°C. New mold insert.",
  "markers":               ["setup changed", "visible flash on shot 12"],
  "video_chunks":          ["component_view_chunk_001.mp4", "component_view_chunk_002.mp4"]
}
```

### 3.6 Model Bundle Contract (consumed by MoldPilot)

MoldPilot expects a directory (or `.mpk` = renamed `.zip`) with:

```
manifest.json        ← bundle_id, model_name/version, classes map, checksums, providers
model.onnx           ← primary ONNX model (FP32)
model_fp16.onnx      ← optional FP16 variant
preprocess.json      ← resize policy, input shape, normalization params
postprocess.json     ← score_threshold, nms_iou_threshold, topk
```

**ONNX input contract**: `[1, 3, 560, 560]` RGB float32, letterboxed, ImageNet-normalised.  
**ONNX output contract**: `boxes` (N×4 xyxy normalised), `scores` (N), `labels` (N int64).

**Class IDs used in monitoring mode**:

| ID | Label |
|----|-------|
| 0 | Component_Base |
| 1 | Weld_Line |
| 2 | Sink_Mark |
| 3 | Flash |
| 4 | Burn_Mark |

### 3.7 S3 Upload (not yet wired)

Architecture is ready: `get_upload_state()` stubs exist in recording/qualification services.
Planned flow: session sealed locally → enqueued → background thread uploads chunks + manifest to
`s3://<bucket>/sessions/<machine_id>/<session_id>/`.

### 3.8 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIA_MOLDPILOT_HOME` | `%LOCALAPPDATA%\ARIA\MoldPilot` | App data root |
| `ARIA_MOTION_MOCK` | `0` | `1` → use mock motion service |
| `ARIA_MOTION_PORT` | `COM6` | Arduino serial port |
| `ARIA_MOTION_BAUD` | `9600` | Serial baud rate |
| `ARIA_MOTION_STEPS_PER_MM` | `160.0` | Axis calibration |

---

## 4. ARIA_MoldTrace

### 4.1 Purpose

Cloud-side analysis pipeline (target: AWS). Takes the raw session recordings produced by MoldPilot
and runs a multi-stage pipeline to:

1. Extract machine **process parameters** from the HMI video stream via OCR.
2. Detect and track **surface defects** on components via deep learning on the inspection video.
3. **Couple** both data streams into labeled records that link process state to component quality —
   formalising the senior operator's mental map into a structured dataset.

These labeled datasets are the training material for the **suggestion logic** that will eventually
power the Startup Assistant in MoldPilot.

### 4.2 Tech Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.10 / 3.11 |
| Video decoding | FFmpeg (via subprocess) |
| Object detection | RF-DETR (PyTorch → ONNX at inference time) |
| OCR | RapidOCR (recognition-only, no detector) |
| Inference runtime | ONNX Runtime (CPU / GPU) |
| Storage abstraction | `IStorageBackend` (`LocalStorageBackend` active, `S3StorageBackend` stub) |
| CI | GitHub Actions (Windows + Ubuntu, Python 3.10 & 3.11) |

### 4.3 Pipeline Stages

The main entry point is `python -m moldtrace run --session <uuid>`. Stages are independently
cacheable; `--force` recomputes from scratch.

```
Stage                   Input                           Output
─────────────────────────────────────────────────────────────────────────────────
① extract_frames        raw MP4 videos                  JPEG frames (5 fps HMI, activity-gated inspection)
② extract_monitor       process video frames            monitor_geometry_<vid>.json + warped frames
③ monitor_quality       warped monitor frames           per-frame usability flags
④ extract_params        warped frames + HMI layout      process_params_<vid>.jsonl (raw OCR)
   clean_timeseries     raw OCR JSONL                   process_params_clean_<vid>.jsonl
   reconstruct_state    clean JSONL                     process_params_statefull_<vid>.jsonl
⑤ extract_audio         raw MP4                         FLAC/WAV audio
⑥ components_from_clips inspection frames               components_<vid>.jsonl (bbox + class)
⑦ defects_from_comps    component crops                 defects_<vid>.jsonl (type + severity)
⑧ merge_components      per-frame detections            components_merged_<vid>.jsonl (stable tracks)
⑨ coupling              merged components + stateful    labeled dataset records (component ↔ params ↔ defects)
                        process params
```

### 4.4 Session Folder Layout

```
sessions/<uuid>/
├── meta/
│   ├── session.json               ← provenance (machine_id, plant, operator, recording_mode)
│   └── run_<timestamp>.json       ← per-execution record
├── inputs/
│   ├── process_video/raw/         ← HMI MP4 files
│   │   └── frames/<video_id>/     ← extracted JPEG frames
│   └── inspection_video/raw/      ← component-view MP4 files
│       └── frames/<video_id>/     ← extracted JPEG frames
└── artifacts/
    ├── process_monitoring/
    │   ├── layout/                ← monitor_geometry_<vid>.json
    │   ├── quality/               ← monitor_quality_<vid>.json
    │   └── ocr/raw/               ← process_params_<vid>.jsonl
    ├── audio/extracted/           ← FLAC/WAV files
    └── inspection/
        ├── detection/             ← components_<vid>.jsonl, components_merged_<vid>.jsonl
        └── defects/               ← defects_<vid>.jsonl
```

### 4.5 Key Schemas

**Process Parameter Record (JSONL)**
```json
{
  "frame_index": 1000,
  "timestamp_sec": 200.0,
  "video_id": "hmi_01",
  "page_id": "main",
  "values": {
    "temp_barrel": {
      "slots": {
        "actual":   { "value": 220.5, "unit": "°C",  "accepted": true },
        "setpoint": { "value": 220.0, "unit": "°C",  "accepted": true }
      }
    },
    "pressure_injection": {
      "slots": {
        "actual": { "value": 1200.0, "unit": "bar", "accepted": true }
      }
    }
  }
}
```

**Component Detection Record (JSONL)**
```json
{
  "frame_idx": 45,
  "timestamp_sec": 1.5,
  "video_id": "inspection_01",
  "class_name": "Component_Base",
  "bbox_xyxy": [100, 200, 400, 500],
  "score": 0.92,
  "centroid": [250, 350]
}
```

**Labeled Dataset Record (coupled output)**
```json
{
  "component_id": "comp_001",
  "production_window": { "start_sec": 45.0, "end_sec": 48.0 },
  "process_parameters": {
    "temp_barrel":         220.5,
    "temp_mold":           85.0,
    "pressure_injection":  1200.0
  },
  "defects": [
    { "type": "Sink_Mark", "severity": "medium", "area_pct": 3.2 }
  ],
  "surface_quality_score": 0.85
}
```

### 4.6 HMI Layout Files

Layouts define the OCR regions of interest for each machine model. They are JSON files stored
under `%LOCALAPPDATA%\ARIA\MoldTrace\layouts\<company>\<machine_family>\<version>.json`.

```json
{
  "pages": [{
    "page_id": "main",
    "parameters": [{
      "parameter_id": "temp_barrel",
      "parameter_name": "Barrel Temperature",
      "slots": [{
        "slot_id": "actual",
        "roi": { "x": 120, "y": 45, "w": 80, "h": 20 },
        "type": "numeric",
        "unit": "°C",
        "range": [150, 320]
      }]
    }]
  }]
}
```

Layouts are **one-time authoring** per machine family. The `layout init` wizard and
`layout edit` CLI provide interactive tooling.

### 4.7 Model Roles

MoldTrace uses its own model registry with named **roles**:

| Role | Model task | Trained in MoldVision? |
|------|-----------|----------------------|
| `monitor_segmenter` | Segmentation — detect HMI screen quad | ✅ yes |
| `components` | Detection — Component_Base + defect classes | ✅ yes |
| `ocr_recognizer` | OCR recognition (RapidOCR model) | external |

### 4.8 AWS Integration Status

`S3StorageBackend` stub exists; all session I/O flows through the `IStorageBackend` abstraction
making the cloud swap code-compatible. The swap requires:

1. Implement `S3StorageBackend` with `boto3`.
2. Set `ARIA_STORAGE_BACKEND=s3` + S3 bucket config.
3. Add Lambda trigger on `sessions/<uuid>/meta/session.json` creation to fire pipeline.

### 4.9 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIA_SESSIONS_ROOT` | `%LOCALAPPDATA%\ARIA\MoldTrace\sessions` | Session storage root |
| `ARIA_MODELS_ROOT` | `%LOCALAPPDATA%\ARIA\MoldTrace\models` | Model bundle root |
| `ARIA_STORAGE_BACKEND` | `local` | `local` or `s3` |

---

## 5. ARIA_MoldVision

### 5.1 Purpose

Internal toolchain for the ARIA team to **train, validate, and package Computer Vision models**
that are deployed into MoldPilot (monitoring mode) and MoldTrace (all vision-based pipeline stages).

Key capabilities:
- UUID-based dataset management with COCO/YOLO ingestion
- RF-DETR fine-tuning (object detection and instance segmentation)
- ONNX / TensorRT / INT8-quantized model export
- Deployment bundle creation (`.mpk` format — the shared model exchange format)
- Label Studio ML backend for active-learning annotation loops

### 5.2 Tech Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.9+ |
| Training framework | PyTorch 2.6+, RF-DETR 1.5 |
| Data augmentation | Albumentations 1.4 |
| Annotation format | COCO (primary), YOLO (ingestion) |
| COCO tooling | pycocotools, faster-coco-eval |
| Export | ONNX 1.17, ONNX Runtime GPU 1.20, TensorRT (optional) |
| Active learning | Label Studio ML backend |
| Detection utilities | Supervision 0.27 |

### 5.3 Typical Workflow

```
1. moldvision dataset create --name <name> -c Component_Base -c Sink_Mark …
   └─ Creates datasets/<UUID>/ with METADATA.json

2. Place raw images in datasets/<UUID>/raw/
   Place labels in datasets/<UUID>/labels_inbox/yolo/ or labels_inbox/coco/

3. moldvision dataset ingest -d <UUID> --train-ratio 0.8
   └─ Stratified COCO split → coco/train/, coco/valid/, coco/test/

4. moldvision dataset validate -d <UUID> --task seg

5. moldvision train -d <UUID> --task seg --epochs 50 --batch-size 4
   └─ Saves checkpoint_best_*.pth + model_config.json

6. moldvision export -d <UUID> -w checkpoint_best_total.pth --format onnx_fp16

7. moldvision bundle -d <UUID> -w checkpoint_best_total.pth \
     --model-name "mold-defect" --model-version 1.0.0 --mpk
   └─ Produces datasets/<UUID>/deploy/mold-defect-v1.0.0.mpk
```

### 5.4 Dataset Folder Layout

```
datasets/<UUID>/
├── METADATA.json               ← { uuid, name, class_names, created_at }
├── raw/                        ← original unlabeled images
├── labels_inbox/
│   ├── yolo/                   ← external YOLO .txt labels staging
│   ├── coco/                   ← external COCO JSON staging
│   └── quarantine/             ← conflicting/rejected labels
├── coco/
│   ├── train/_annotations.coco.json + *.jpg
│   ├── valid/_annotations.coco.json + *.jpg
│   └── test/_annotations.coco.json  + *.jpg
├── models/
│   ├── checkpoint_best_total.pth
│   ├── checkpoint_portable.pth ← weights-only, PyTorch 2.6+ compatible
│   └── model_config.json
├── exports/
│   └── model.onnx, model_fp16.onnx, model_quantized.onnx
└── deploy/
    └── <bundle-id>/
        ├── manifest.json
        ├── model.onnx
        ├── model_fp16.onnx
        ├── preprocess.json
        ├── postprocess.json
        ├── classes.json
        └── checkpoint.pth
```

### 5.5 Bundle Manifest Schema

```json
{
  "bundle_id":       "mold-defect-v1.0.0",
  "model_name":      "mold-defect",
  "model_version":   "1.0.0",
  "channel":         "stable",
  "supersedes":      null,
  "min_app_version": "0.0.0",
  "classes":         { "0": "Component_Base", "1": "Weld_Line", "2": "Sink_Mark", "3": "Flash", "4": "Burn_Mark" },
  "format_version":  1,
  "created_at":      "2026-03-17T10:00:00Z",
  "primary_artifact":"model.onnx",
  "artifacts":       ["model.onnx", "model_fp16.onnx", "preprocess.json", "postprocess.json"],
  "runtime": {
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "checksums": {
    "model.onnx":      "sha256:<hex>",
    "preprocess.json": "sha256:<hex>"
  }
}
```

### 5.6 Preprocessing Contract (preprocess.json)

All three projects share this inference preprocessing contract. MoldVision writes it,
MoldPilot and MoldTrace consume it.

```json
{
  "resize_policy": "letterbox",
  "target_h": 640,
  "target_w": 640,
  "input_color": "RGB",
  "input_layout": "NCHW",
  "input_dtype": "float32",
  "input_range": "0..1",
  "normalize": {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  }
}
```

> **Note**: MoldPilot's monitoring mode currently uses a fixed 560×560 input shape. Align
> `target_h` / `target_w` in `preprocess.json` with the shape the model was exported at.

### 5.7 Active Learning Loop (Label Studio)

```
1. Run moldvision label_studio_backend --port 9090
2. In Label Studio: Settings → Machine Learning → http://localhost:9090
3. Import new raw images → pre-labels appear automatically
4. Reviewer corrects annotations → export as COCO JSON
5. moldvision dataset import-coco -d <UUID> --split train --coco-json <export.json> --images-dir <dir>
6. moldvision dataset ingest -d <UUID>   (re-ingest with new labels)
7. moldvision train ...                  (retrain)
8. moldvision bundle ... --mpk           (new bundle → deploy to MoldPilot/MoldTrace)
```

---

## 6. Integration Data Flows

### 6.1 Flow A — Qualification Recording → Dataset Building

```
[Operator @ MoldPilot]
  │  presses Start in Qualification Mode
  │  MoldPilot records:
  │    • component_view_chunk_*.mp4   (Baumer camera, H.264, 60 s chunks)
  │    • process_view_chunk_*.mp4     (second camera or same + notes)
  │    • session manifest JSON        (machine_id, mold_id, operator, markers…)
  │
  │  [NOT YET] S3 upload
  ▼
[MoldTrace — create session]
  python tools/create_session.py
    --process-video  <hmi_mp4>
    --inspection-video <component_mp4>
    --recording-mode qualification
  → copies videos to sessions/<uuid>/inputs/
  → writes meta/session.json with MoldPilot manifest fields

[MoldTrace — run pipeline]
  python -m moldtrace run --session <uuid> \
    --extract-frames --extract-monitor --extract-parameters \
    --extract-components --extract-defects --merge-components
  → produces labeled JSONL artifacts
  → coupling stage → labeled dataset records
```

### 6.2 Flow B — Model Training → Bundle Deployment to MoldPilot

```
[MoldVision]
  moldvision dataset create + ingest + train + export + bundle --mpk
  → produces mold-defect-v1.0.0.mpk

[Transfer]
  Copy .mpk to MoldPilot machine (USB / S3 download / shared drive)

[MoldPilot — model registry]
  LocalModelRegistryService.install_bundle("mold-defect-v1.0.0.mpk")
  → validates manifest + checksums
  → extracts to models/bundles/mold-defect-v1.0.0/
  → updates models/registry.json (active_bundle_id)

[MoldPilot — monitoring mode]
  OnnxInferenceService.load_bundle("mold-defect-v1.0.0")
  → reads preprocess.json, postprocess.json
  → loads model_fp16.onnx (CUDA) or model.onnx (CPU)
  → starts inference on live frames
```

### 6.3 Flow C — Model Training → Bundle Deployment to MoldTrace

```
[MoldVision]
  moldvision train --task seg   ← for monitor_segmenter role
  moldvision bundle --mpk

[MoldTrace — model registry]
  python -m moldtrace models install <bundle_dir> --role monitor_segmenter
  python -m moldtrace models activate <bundle_name> --role monitor_segmenter

[MoldTrace — pipeline]
  extract_monitor_stage uses monitor_segmenter bundle
  components_from_clips uses components bundle
```

### 6.4 Flow D — Labeled Dataset → Suggestion Logic (future)

```
[MoldTrace output]
  labeled JSONL: { component, process_parameters, defects }
  accumulated across many qualification sessions

[Future: Suggestion Model Training]
  Input features: process_parameters (barrel temp, mold temp, injection pressure,
                  holding pressure, cooling time, …)
  Target: surface quality score / defect occurrence probability

[Future: Startup Assistant in MoldPilot]
  StartupAssistantService.get_suggestion(current_params) → StartupSuggestion
  → displayed on Startup Assistant screen as threshold bars +
    recommended parameter adjustments
```

---

## 7. Shared Contracts & Compatibility Matrix

### 7.1 Model Bundle Format Compatibility

| Field | MoldPilot consumer | MoldTrace consumer | MoldVision producer |
|-------|-------------------|--------------------|---------------------|
| `manifest.json` format_version | 1 | 2 (superset) | writes both |
| Primary ONNX input shape | 1×3×560×560 | variable per role | must match training export |
| Output keys | `boxes`, `scores`, `labels` | `boxes`, `scores`, `labels`, `masks` (seg) | RF-DETR standard |
| Checksum algorithm | SHA-256 | SHA-256 | SHA-256 |
| Bundle file extension | `.mpk` (zip) | directory or `.mpk` | both |

> **Action item**: Align `format_version` between MoldPilot (v1) and MoldTrace (v2).
> MoldVision should write v2 manifests as the canonical standard.

### 7.2 Class ID Alignment

The following class IDs are used across all three systems. Any retrained model must preserve this mapping.

| ID | Class name | Used in MoldPilot | Used in MoldTrace |
|----|-----------|:-----------------:|:-----------------:|
| 0 | `Component_Base` | ✅ monitoring | ✅ component localization |
| 1 | `Weld_Line` | ✅ monitoring | ✅ defect detection |
| 2 | `Sink_Mark` | ✅ monitoring | ✅ defect detection |
| 3 | `Flash` | ✅ monitoring | ✅ defect detection |
| 4 | `Burn_Mark` | ✅ monitoring | ✅ defect detection |

### 7.3 Session Identity Fields

These fields originate in MoldPilot and must be preserved all the way through MoldTrace
to the final labeled dataset, ensuring full traceability.

| Field | Source | Used by |
|-------|--------|---------|
| `machine_id` | MoldPilot config | MoldTrace session.json, labeled dataset |
| `mold_id` | MoldPilot config | MoldTrace session.json, labeled dataset |
| `part_id` | MoldPilot config | MoldTrace session.json |
| `session_id` | MoldPilot runtime | S3 path, MoldTrace session UUID |
| `operator_name` | Qualification form | MoldTrace session.json |
| `batch_number` | Qualification form | MoldTrace session.json |
| `markers` | Qualification form | MoldTrace session.json (for annotation events) |

### 7.4 Storage Root Conventions

| Project | Windows default | Override env var |
|---------|----------------|-----------------|
| MoldPilot | `%LOCALAPPDATA%\ARIA\MoldPilot` | `ARIA_MOLDPILOT_HOME` |
| MoldTrace | `%LOCALAPPDATA%\ARIA\MoldTrace` | `ARIA_SESSIONS_ROOT`, `ARIA_MODELS_ROOT` |
| MoldVision | `%LOCALAPPDATA%\MoldVision` (config) | `MOLDVISION_DATASETS` |

---

## 8. Current Integration Gaps & Open Work Items

The following items are known gaps as of **April 2026**. They represent concrete engineering tasks
needed to close the end-to-end loop.

### 8.1 S3 Upload (MoldPilot → MoldTrace)

**Status**: Architecture ready on both sides; wire-up not implemented.

**What's needed**:
- Implement `S3UploadService` in MoldPilot (`infrastructure/`), wiring it to `get_upload_state()`.
- Implement `S3StorageBackend` in MoldTrace (`storage.py`), replacing `NotImplementedError`.
- Agree on the S3 bucket structure (suggested: `s3://aria-sessions/<machine_id>/<session_id>/`).
- Add a trigger (Lambda or polling) in MoldTrace to auto-create a session when a manifest lands in S3.
- Add progress reporting UI in MoldPilot (upload state badge on Sessions Browser screen).

### 8.2 MoldTrace REST API / Async Trigger

**Status**: MoldTrace is currently CLI-only.

**What's needed**:
- Design a thin HTTP API (FastAPI recommended) wrapping `run_pipeline()`.
- Endpoint: `POST /sessions` (create from S3 keys) → returns `session_uuid`.
- Endpoint: `POST /sessions/{uuid}/run` → trigger pipeline asynchronously.
- Endpoint: `GET /sessions/{uuid}/status` → poll status.
- Lambda handler can call this or invoke `run_pipeline()` directly.

### 8.3 Coupling Stage Completion (MoldTrace)

**Status**: `coupling.py` architecture exists; temporal alignment logic is pending.

**What's needed**:
- Define the coupling window strategy (e.g., use the process parameters active during the
  ±N seconds around each component's production window).
- Implement the labeled dataset writer (JSONL per session).
- Define the final schema for the labeled record (used as training data for suggestion logic).

### 8.4 Labeled Dataset → Suggestion Model (MoldTrace → MoldPilot)

**Status**: No implementation yet. `MockStartupAssistantService` provides synthetic data.

**What's needed**:
- Decide on the model type for suggestion logic (e.g., gradient boosted trees, simple MLP,
  rule-based lookup). Input: process parameters. Output: defect risk scores or parameter adjustments.
- Train on accumulated labeled datasets from MoldTrace.
- Export to ONNX or a lightweight JSON rule-set.
- Wire into `StartupAssistantService` in MoldPilot (replace the mock).

### 8.5 MoldVision Dataset Schema for Labeled Images

**Status**: Dataset schema is well-defined for model training (COCO). A schema for the
**labeled process dataset** (coupling images + defect labels + parameter metadata) has not been
formalised yet.

**What's needed**:
- Define a COCO extension or side-car JSON schema that attaches process parameters to each image.
- MoldTrace should produce this extended format as output of the coupling stage.
- MoldVision should be able to ingest it (or the standard COCO part of it) for future model training
  that incorporates process context.

### 8.6 Model Bundle Format Alignment

**Status**: MoldPilot uses `format_version: 1`; MoldTrace expects `format_version: 2`.

**What's needed**:
- Update MoldVision's `bundle.py` to write `format_version: 2` as default.
- Update MoldPilot's `model_registry.py` to accept v2 manifests.
- Document the v1 → v2 diff (v2 adds `masks` output key support for segmentation models).

---

## 9. Development & Tooling Notes

### 9.1 Running Without Hardware (MoldPilot)

```powershell
# All hardware mocked
$env:ARIA_MOTION_MOCK = "1"
.\.venv\Scripts\python.exe -m aria_moldpilot
```

The Baumer camera service also has a software mock path; set camera discovery to
return `MockCameraService` via the DI container in `app.py`.

### 9.2 Running MoldTrace Locally (without S3)

```powershell
# Set sessions root to a local path
$env:ARIA_SESSIONS_ROOT = "C:\dev\aria\sessions"
$env:ARIA_MODELS_ROOT   = "C:\dev\aria\models"

python tools/create_session.py `
  --process-video    C:\recordings\hmi.mp4 `
  --inspection-video C:\recordings\component.mp4 `
  --recording-mode   qualification

python -m moldtrace run --session <uuid> --extract-frames --process-fps 5
```

### 9.3 Training a New Bundle with MoldVision

```powershell
# One-time setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install rfdetr==1.5.0
pip install -e .

# Workflow
moldvision dataset create --name mold-defect-v2 -c Component_Base -c Weld_Line -c Sink_Mark -c Flash -c Burn_Mark
# … place images in raw/, labels in labels_inbox/ …
moldvision dataset ingest -d <UUID> --train-ratio 0.8
moldvision train -d <UUID> --task detect --epochs 50 --batch-size 4 --grad-accum 4
moldvision export -d <UUID> -w checkpoint_best_total.pth --format onnx_fp16
moldvision bundle -d <UUID> -w checkpoint_best_total.pth `
  --model-name "mold-defect" --model-version 2.0.0 `
  --channel stable --supersedes mold-defect-v1.0.0 --mpk
```

### 9.4 Test Commands

| Project | Command |
|---------|---------|
| MoldPilot | `.\.venv\Scripts\python.exe -m pytest tests/ -v` |
| MoldTrace | `pytest tests/ -v` (GitHub Actions: Windows + Ubuntu, Py 3.10 & 3.11) |
| MoldVision | `pytest` (see `AGENTS.md`) |

### 9.5 Linting

| Project | Command |
|---------|---------|
| MoldPilot | `.\.venv\Scripts\python.exe -m ruff check src/ tests/` |
| MoldTrace | (ruff or equivalent — check `pyproject.toml` / `requirements-dev.txt`) |
| MoldVision | (ruff or equivalent — check `pyproject.toml`) |

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **Qualification Mode** | MoldPilot operating mode that records session videos during mold setup/tuning |
| **Monitoring Mode** | MoldPilot operating mode that runs live ONNX inference and tracks defect severity |
| **Startup Assistant** | MoldPilot screen showing ML-suggested process parameters for junior operators |
| **Session** | A bounded recording window: one mold startup or production run, identified by UUID |
| **Session Manifest** | JSON file sealing a qualification session: operator, timestamps, video chunks, markers |
| **Bundle / .mpk** | Model deployment package: ONNX model + manifest + preprocessing contracts |
| **HMI Layout** | JSON definition of screen ROIs used by MoldTrace to extract process parameters |
| **Coupling** | MoldTrace step that aligns component defects with contemporaneous process parameter values |
| **Labeled Dataset** | Output of coupling: structured records linking process state to surface quality |
| **Suggestion Logic** | Future ML model in MoldPilot that turns labeled datasets into operator recommendations |
| **RF-DETR** | Real-time Fast Detection Transformer — the object detection / segmentation architecture used |
| **Component_Base** | Class ID 0: the physical part being inspected (non-defect anchor box) |
| **IoU Tracker** | Simple SORT-like bounding-box tracker using intersection-over-union assignment |
| **IStorageBackend** | MoldTrace abstraction layer over local filesystem / S3 |
| **format_version** | Bundle manifest version. v1 = MoldPilot (detection only). v2 = MoldTrace (+ segmentation) |
