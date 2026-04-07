# ARIA Data Lake — Design & Implementation Plan

> **Status**: Design approved. Implementation pending.  
> **Scope**: `ARIA_MoldVision` — `moldvision lake` CLI surface, local folder layout, and
> migration path toward low-cost remote storage.  
> Keep this document in sync with `ARIA_System_Integration.md` when integration contracts change.

---

## 1. Problem Statement

MoldPilot produces qualification sessions (MP4 videos + manifest JSON) that MoldTrace
processes into per-frame JPEG images and labeled artifacts. Today there is no structured
place for those images to live, no way to track which images have been annotated, and no
tooling to build a training-ready dataset from a set of sessions with proper class-distribution
rules.

This document designs the **ARIA Data Lake**: a local-first, folder-based image and annotation
store with a CLI for pulling balanced training datasets, tracking annotation status, and storing
trained model bundles — with a clear, low-cost migration path to remote object storage when the
team is ready.

---

## 2. Guiding Principles

1. **Local-first, zero cost to start.** Everything runs on a dev workstation or NAS.
   No cloud account is required to start labeling and training.
2. **Session-centric.** Images belong to sessions. Session metadata (machine, mold, operator,
   markers) is the primary axis for filtering and auditing.
3. **Annotation-status aware.** Every image is `unlabeled`, `labeled`, `hard_negative`, or
   `background`. The pull command only uses what you tell it to.
4. **Distribution rules are first-class.** Sessions have wildly different frame counts.
   The pull engine enforces per-session caps and class balancing so training sets are
   always well-conditioned regardless of how uneven the raw data is.
5. **Compatible with the existing `moldvision` pipeline.** `lake pull` writes into the
   standard `datasets/<UUID>/` layout. Nothing downstream changes.
6. **Cloud-agnostic abstraction.** The storage backend is pluggable (local → B2/R2/S3)
   without changing any annotation or training commands.

---

## 3. Available Session Metadata

Qual-session metadata originates in MoldPilot and propagates through MoldTrace to the data lake.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | e.g. `qual_20260324T093000Z_a1b2c3d4` |
| `machine_id` | string | e.g. `machine_01` |
| `mold_id` | string | e.g. `mold_a12` |
| `part_id` | string | e.g. `part_cap_32` |
| `started_at` | ISO-8601 UTC | Recording start |
| `ended_at` | ISO-8601 UTC | Recording end |
| `status` | string | `completed`, `aborted`, … |
| `operator_name` | string | |
| `batch_number` | string | e.g. `LOT-2026-0324-A` |
| `operator_notes` | string | freeform notes |
| `markers` | list[string] | Operator-flagged events: `["setup changed", "visible flash on shot 12"]` |
| `video_chunks` | list[string] | MP4 filenames that were recorded |

These fields are written verbatim into `sessions/<session_id>/session_meta.json` and indexed
in `image_index.jsonl` for per-image filtering.

---

## 4. Folder Layout

```
aria_data_lake/                              ← root; set via ARIA_DATA_LAKE env var
│                                              default: %LOCALAPPDATA%\ARIA\DataLake
│
├── data_lake_config.json                    ← created by `lake init`
│
├── sessions/                                ← one folder per qualification session
│   └── <session_id>/
│       ├── session_meta.json                ← full MoldPilot manifest (all fields above)
│       ├── inspection_frames/               ← component-view JPEGs
│       │   └── frame_<N>.jpg
│       ├── monitor_frames/                  ← HMI-view JPEGs (monitor segmentation task)
│       │   └── frame_<N>.jpg
│       └── annotations/
│           ├── detect/
│           │   ├── _annotations.coco.json   ← COCO bounding-box labels (defect detection)
│           │   └── yolo/                    ← optional YOLO .txt drop-zone before ingest
│           └── seg/
│               └── _annotations.coco.json   ← COCO polygon labels (monitor segmentation)
│
├── image_index.jsonl                        ← flat catalogue; one JSON record per image
│                                              rebuilt by `lake index --rebuild`
│                                              appended by `lake session import`
│
├── pools/                                   ← special-purpose image sets
│   ├── hard_negatives/
│   │   └── manifest.jsonl                   ← refs to images: model predicted defect,
│   │                                          human confirmed none
│   └── backgrounds/
│       └── manifest.jsonl                   ← images with no component visible
│
├── datasets/                                ← standard MoldVision training datasets
│   └── <UUID>/                              ← created by `lake pull` or manually
│       ├── METADATA.json
│       ├── raw/
│       ├── coco/ train/ valid/ test/
│       ├── models/
│       ├── exports/
│       └── deploy/
│
└── models/                                  ← trained-model registry (bundles)
    ├── defect_detection/
    │   ├── registry.json
    │   └── <bundle_id>/                     ← extracted .mpk
    │       ├── manifest.json
    │       ├── model.onnx
    │       ├── model_fp16.onnx
    │       ├── preprocess.json
    │       └── postprocess.json
    └── monitor_segmentation/
        ├── registry.json
        └── <bundle_id>/
```

### 4.1 `image_index.jsonl` record schema

One JSON object per line. The index is the queryable backbone of `lake pull`.

```json
{
  "rel_path":     "sessions/qual_20260324T093000Z_a1b2c3d4/inspection_frames/frame_000100.jpg",
  "session_id":   "qual_20260324T093000Z_a1b2c3d4",
  "machine_id":   "machine_01",
  "mold_id":      "mold_a12",
  "part_id":      "part_cap_32",
  "operator_name":"Andrea Rossi",
  "batch_number": "LOT-2026-0324-A",
  "started_at":   "2026-03-24T09:30:00Z",
  "markers":      ["setup changed"],
  "task":         "detect",
  "label_status": "labeled",
  "frame_idx":    100
}
```

`label_status` values:

| Value | Meaning |
|-------|---------|
| `unlabeled` | No annotation exists yet |
| `labeled` | COCO annotation present and reviewed |
| `hard_negative` | Image in `pools/hard_negatives/` |
| `background` | Image in `pools/backgrounds/` |

### 4.2 `models/*/registry.json` schema

```json
{
  "task": "defect_detection",
  "bundles": [
    {
      "bundle_id":    "mold-defect-v2.0.0",
      "version":      "2.0.0",
      "channel":      "stable",
      "dataset_uuid": "3f9a1c2b-...",
      "created_at":   "2026-04-07T10:00:00Z",
      "path":         "models/defect_detection/mold-defect-v2.0.0/"
    }
  ],
  "active": {
    "stable": "mold-defect-v2.0.0",
    "dev":    "mold-defect-v2.1.0-dev"
  }
}
```

---

## 5. Two Model Tasks

| | `detect` | `seg` |
|---|---|---|
| **Purpose** | Defect detection on component frames | Monitor screen quad localization |
| **Source frames** | `inspection_frames/` | `monitor_frames/` |
| **Annotation type** | Bounding boxes | Polygons |
| **Annotation tool** | Label Studio (bbox) | Label Studio (polygons) |
| **Lake annotation path** | `annotations/detect/` | `annotations/seg/` |
| **Class schema** | **Fixed** — see §5.1 | `HMI_Screen` (1 class) |

### 5.1 Defect Detection — Fixed Class IDs (must never change)

These IDs are a hard contract with MoldPilot and MoldTrace. Any model must preserve this mapping.

| ID | Class name |
|----|-----------|
| 0 | `Component_Base` |
| 1 | `Weld_Line` |
| 2 | `Sink_Mark` |
| 3 | `Flash` |
| 4 | `Burn_Mark` |

> ⚠️ Changing class IDs requires a coordinated update in MoldPilot (`OnnxInferenceService`),
> MoldTrace (`defects` stage), and all deployed bundles.

---

## 6. `moldvision lake` CLI Surface

### 6.1 `lake init`

```
moldvision lake init [--root PATH]
```

Creates the full folder skeleton and writes `data_lake_config.json`.
Sets `ARIA_DATA_LAKE` in the config; reads it back on every subsequent command.

---

### 6.2 `lake session import`

```
moldvision lake session import
  --session-meta  <session.json>         # MoldPilot manifest JSON
  --inspection-frames <dir>             # directory of JPEGs (component view)
  [--monitor-frames   <dir>]            # directory of JPEGs (HMI view)
  [--annotations-detect <coco.json>]    # pre-existing COCO labels for detect
  [--annotations-seg    <coco.json>]    # pre-existing COCO labels for seg
  [--overwrite]                         # re-import if session already exists
```

Actions:
1. Copies frames into `sessions/<session_id>/inspection_frames/` (or `monitor_frames/`).
2. Writes `session_meta.json`.
3. Copies annotation JSONs if provided, setting `label_status = "labeled"` for covered images.
4. Appends new records to `image_index.jsonl`.

**Integration note**: MoldTrace already extracts frames into
`sessions/<uuid>/inputs/inspection_video/frames/`. A thin wrapper script
(`scripts/import_from_moldtrace.py`) can call this command after the MoldTrace pipeline
completes, mapping MoldTrace paths to data lake paths automatically.

---

### 6.3 `lake session list`

```
moldvision lake session list
  [--machine-id X] [--mold-id Y] [--part-id Z]
  [--from 2026-01-01] [--to 2026-06-01]
  [--task detect|seg]
  [--label-status labeled|unlabeled|any]
  [--marker "visible flash"]
  [--min-frames N]
```

Prints a table:

```
session_id                            machine  mold      frames  labeled  coverage
qual_20260324T093000Z_a1b2c3d4        mach_01  mold_a12   1240     980     79%
qual_20260401T143000Z_f8e9d3c1        mach_01  mold_a12    540     540    100%
```

---

### 6.4 `lake pull` — the core command

Builds a training-ready `datasets/<UUID>/` from selected sessions.

```
moldvision lake pull
  --task detect|seg

  # Session selection (choose one approach)
  --sessions s1,s2,s3            # explicit session IDs
  --all                          # all sessions in the lake
  --machine-id X                 # filter by machine
  --mold-id Y                    # filter by mold
  --part-id Z                    # filter by part
  --from 2026-01-01              # filter by date range
  --to   2026-06-01
  --marker "setup changed"       # only sessions that have this marker

  # Annotation filter
  --label-status labeled         # default: labeled only
  --include-hard-negatives       # append hard_negatives pool (empty annotations)
  --include-backgrounds          # append backgrounds pool (empty annotations)

  # Distribution rules
  --max-per-session N            # cap: at most N images from any single session
  --min-per-session N            # skip sessions with fewer than N labeled images
  --balance-classes              # undersample to equalise per-class annotation counts
  --min-per-class N              # abort if any class has fewer than N annotations
  --train-ratio 0.8              # train/valid split ratio (default: 0.8)
  --seed 42                      # random seed for reproducibility

  # Output
  --dataset-uuid UUID            # write into existing dataset, or auto-generate
  --dataset-name "my-dataset"    # human-readable name for METADATA.json
  --dataset-root PATH            # default: data_lake/datasets/
  --dry-run                      # print the distribution report, create nothing
```

#### Pull algorithm

```
1. Read image_index.jsonl — filter by task, label_status, and all --filter flags
2. Group images by session_id
3. Apply --min-per-session: drop sessions below threshold
4. Apply --max-per-session: random-sample within each session (seeded)
5. Collect COCO annotations from sessions/<id>/annotations/<task>/_annotations.coco.json
6. Merge annotations across sessions into a single COCO dict
7. If --include-hard-negatives: load pools/hard_negatives/manifest.jsonl,
   append images with empty annotation lists
8. If --include-backgrounds: same for pools/backgrounds/manifest.jsonl
9. If --balance-classes: count annotations per category, undersample images
   containing over-represented classes until counts are within a 2× ratio
   (keeps a 2× tolerance to avoid discarding too many Component_Base images)
10. Shuffle and split into train/valid (--train-ratio, --seed)
11. Call create_dataset() + copy images into datasets/<UUID>/raw/
12. Write merged COCO JSONs into datasets/<UUID>/coco/train/ and valid/
13. Print distribution report: images per session, annotations per class, split counts
```

#### Distribution report (--dry-run output)

```
lake pull DRY RUN — task=detect
─────────────────────────────────────────────────────────
Sessions included (after filters):
  qual_20260324...  machine_01  mold_a12   980 labeled → capped at 300 (--max-per-session)
  qual_20260401...  machine_01  mold_a12   540 labeled → all 540 included

Class distribution (before balance):
  Component_Base  1623  ████████████████████
  Weld_Line        312  ████
  Sink_Mark        198  ███
  Flash             87  █
  Burn_Mark         44  ▌
  hard_negatives    60  ▌

After --balance-classes (2× tolerance on rarest class = Burn_Mark × 2 = 88):
  Component_Base   88
  Weld_Line        88
  Sink_Mark        88
  Flash            87
  Burn_Mark        44

Train: 340 images / 395 annotations
Valid:  86 images /  95 annotations
─────────────────────────────────────────────────────────
Run without --dry-run to create dataset.
```

---

### 6.5 `lake index`

```
moldvision lake index --rebuild          # full scan → rewrite image_index.jsonl
moldvision lake index --stats [--task detect|seg]  # print session and class stats
```

`--rebuild` walks all `sessions/*/` folders, reads every `session_meta.json` and
`annotations/<task>/_annotations.coco.json`, and regenerates `image_index.jsonl` from scratch.
Run this after manually copying frames or after updating annotations outside the CLI.

---

### 6.6 `lake models install / list / promote`

```
moldvision lake models install <bundle.mpk> --task detect|seg
moldvision lake models list    --task detect|seg
moldvision lake models promote <bundle_id> --channel stable|dev
```

`install` extracts the `.mpk` bundle into `models/<task>/<bundle_id>/` and appends an
entry to `registry.json`. `promote` updates the `active.<channel>` pointer.

---

## 7. Hard Negatives and Backgrounds

### 7.1 Hard Negatives

Images where the deployed model predicted a defect that a human reviewer confirmed was a
false positive. These are the most impactful images for reducing false-positive rates.

**Workflow:**
1. Run inference with the current model on unlabeled session frames.
2. Review detections in Label Studio (or via `moldvision infer`).
3. For frames where all detections are wrong: export image path to `pools/hard_negatives/manifest.jsonl`.
4. Include in the next training run with `lake pull --include-hard-negatives`.

`manifest.jsonl` record:
```json
{"rel_path": "sessions/.../inspection_frames/frame_000842.jpg",
 "session_id": "qual_...", "reason": "model_false_positive", "added_at": "2026-04-07T10:00:00Z"}
```

### 7.2 Backgrounds

Images where no injection-molded component is visible (empty conveyor, setup shots, occlusions).
Teaching the model to output no detections on these is important to reduce spurious alarms
in the MoldPilot monitoring mode.

Both pools are ingested as empty-annotation COCO images. They receive a separate
`label_status = "hard_negative"` / `"background"` in the index so they can be
independently controlled by the pull command.

---

## 8. Typical End-to-End Workflow

```
① MoldPilot records a qualification session
   → local: sessions/<session_id>/

② MoldTrace processes the session (extract_frames, extract_monitor, …)
   → produces: inspection_frames/*.jpg + monitor_frames/*.jpg

③ Import into the data lake:
   moldvision lake session import \
     --session-meta sessions/<id>/meta/session.json \
     --inspection-frames sessions/<id>/inputs/inspection_video/frames/vid_01/

④ Label images in Label Studio (ML backend pre-labels via active learning)
   → export COCO JSON to sessions/<session_id>/annotations/detect/_annotations.coco.json

⑤ Rebuild index (or it was updated incrementally in step ③):
   moldvision lake index --stats

⑥ Pull a training dataset:
   moldvision lake pull \
     --task detect \
     --mold-id mold_a12 \
     --label-status labeled \
     --include-hard-negatives \
     --max-per-session 300 \
     --balance-classes \
     --min-per-class 40 \
     --train-ratio 0.85 \
     --dataset-name "mold-defect-2026-q2"

⑦ Validate and train (unchanged workflow):
   moldvision dataset validate -d <UUID> --task detect
   moldvision train -d <UUID> --task detect --epochs 60 --batch-size 4

⑧ Export and bundle:
   moldvision export -d <UUID> -w checkpoint_best_total.pth --format onnx_fp16
   moldvision bundle -d <UUID> -w checkpoint_best_total.pth \
     --model-name mold-defect --model-version 2.1.0 --mpk

⑨ Register bundle:
   moldvision lake models install datasets/<UUID>/deploy/mold-defect-v2.1.0.mpk --task detect
   moldvision lake models promote mold-defect-v2.1.0 --channel stable
```

---

## 9. Remote Storage — Future Migration Path

The data lake is designed so that the **storage backend is the only thing that changes**
when moving to the cloud. All CLI commands, annotation workflows, and training pipelines
stay identical.

### 9.1 Why Move to Remote Storage?

| Trigger | Reason |
|---------|--------|
| Second annotator or team member | Shared access to the same image pool |
| Backup & disaster recovery | Local NAS is not enough for labeling work |
| MoldTrace running on AWS | Frames land in S3; importing to local adds a round-trip |
| Large datasets (>50 GB) | Exceeds practical local storage |

### 9.2 Backend Options Compared (Cost-First)

| Option | Storage cost | Egress cost | S3-compatible | Notes |
|--------|-------------|-------------|:---:|-------|
| **Local / NAS** | HW only | Free | — | Current phase. Zero recurring cost. |
| **Backblaze B2** | $0.006/GB/mo | Free to CDN; $0.01/GB elsewhere | ✅ | ~4× cheaper than S3. Good for frames+labels. |
| **Cloudflare R2** | $0.015/GB/mo | **Zero egress** | ✅ | Best choice when models are downloaded frequently by MoldPilot/MoldTrace. |
| **MinIO (self-hosted)** | HW only | Free | ✅ | Run on a NAS or internal server. Free forever, S3 API, full control. |
| **AWS S3** | $0.023/GB/mo | $0.09/GB | ✅ | Most expensive. Only justifiable if already deep in AWS (Lambda, EC2). |
| **DagsHub (free tier)** | 5 GB free | Free | via DVC | Good for prototyping DVC workflows before committing to a paid backend. |

**Recommendation for the next step after local**: use **Cloudflare R2** for frames and
annotations (zero egress means no cost when MoldPilot downloads new bundles or MoldTrace
uploads extracted frames), with **MinIO** as a local mirror/cache during development.

### 9.3 Abstraction Layer Design

The data lake code will use a `StorageBackend` protocol matching the pattern already
established in MoldTrace:

```python
class ILakeStorage(Protocol):
    def exists(self, rel_path: str) -> bool: ...
    def read_bytes(self, rel_path: str) -> bytes: ...
    def write_bytes(self, rel_path: str, data: bytes) -> None: ...
    def list_prefix(self, prefix: str) -> list[str]: ...
    def delete(self, rel_path: str) -> None: ...
```

Implementations:
- `LocalLakeStorage(root: Path)` — active now, trivial filesystem wrapper
- `S3LakeStorage(bucket: str, prefix: str, client: boto3.S3Client)` — compatible with
  AWS S3, Cloudflare R2, Backblaze B2, and MinIO (all speak S3 API)

Switching backend = one line change in `data_lake_config.json`:
```json
{
  "backend": "s3",
  "s3_bucket": "aria-data-lake",
  "s3_prefix": "v1/",
  "s3_endpoint_url": "https://<account>.r2.cloudflarestorage.com"
}
```

`image_index.jsonl`, session frames, annotations, and model bundles all use relative
paths (`rel_path`) so they are portable between backends without any path rewriting.

### 9.4 DVC as an Optional Versioning Layer

If the team wants reproducible dataset snapshots (e.g., `dataset_v3 = sessions A+B+C at
these label versions`), [DVC](https://dvc.org) can sit on top of any S3-compatible backend:

```bash
dvc init
dvc remote add -d lake s3://aria-data-lake/dvc  # or R2 / B2 endpoint
dvc add aria_data_lake/sessions/
dvc push
```

This gives git-tracked dataset versions for free, without DagsHub's platform dependency.
DVC is added incrementally — nothing in the local lake design blocks it.

### 9.5 Migration Checklist (Local → R2)

1. Create Cloudflare R2 bucket `aria-data-lake`.
2. Generate R2 API token (S3-compatible credentials).
3. Update `data_lake_config.json`: `"backend": "s3"`, set `s3_endpoint_url`.
4. Run `moldvision lake sync --upload` to push existing sessions + annotations.
5. Update MoldTrace import script to push new sessions directly to R2 instead of local copy.
6. All `lake pull`, `lake session list`, etc. continue to work unchanged.

---

## 10. Implementation Phases

### Phase 1 — Local lake skeleton & index

Files to create / modify:
- `moldvision/lake.py` — `LakeConfig`, `LocalLakeStorage`, `image_index` helpers
- `moldvision/cli.py` — `lake` subcommand with `init`, `session import`, `index`
- `moldvision/cli_handlers.py` — `handle_lake_*` functions

Commands delivered: `lake init`, `lake session import`, `lake index --rebuild / --stats`

Acceptance test:
```
moldvision lake init --root C:\dev\aria_data_lake
moldvision lake session import --session-meta ... --inspection-frames ...
moldvision lake index --stats
```

### Phase 2 — Pull engine

Files to create / modify:
- `moldvision/lake_pull.py` — session filtering, per-session cap, class balancing, COCO merge

Commands delivered: `lake pull` (all flags), `lake session list`

Acceptance test:
```
moldvision lake pull --task detect --all --max-per-session 200 --balance-classes --dry-run
moldvision lake pull --task detect --all --max-per-session 200 --balance-classes \
  --dataset-name "smoke-test"
moldvision dataset validate -d <generated-UUID> --task detect
```

### Phase 3 — Model registry

Files to create / modify:
- `moldvision/lake_models.py` — `registry.json` read/write, `install`, `list`, `promote`

Commands delivered: `lake models install / list / promote`

### Phase 4 — Storage backend abstraction

Files to create / modify:
- `moldvision/lake_storage.py` — `ILakeStorage` protocol, `LocalLakeStorage`, `S3LakeStorage`
- `moldvision/lake.py` — wire `LakeConfig.backend` to the right implementation

Commands delivered: `lake sync --upload / --download` (push/pull full lake to/from S3)

### Phase 5 — MoldTrace integration

Files to create (in `ARIA_MoldTrace`):
- `scripts/import_session_to_lake.py` — runs after MoldTrace pipeline, calls
  `moldvision lake session import` automatically

No changes to MoldVision needed; Phase 5 is purely a MoldTrace-side script.

---

## 11. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIA_DATA_LAKE` | `%LOCALAPPDATA%\ARIA\DataLake` | Data lake root directory |
| `ARIA_LAKE_BACKEND` | `local` | `local` or `s3`; overrides `data_lake_config.json` |
| `ARIA_LAKE_S3_BUCKET` | — | S3 / R2 / B2 bucket name |
| `ARIA_LAKE_S3_ENDPOINT` | — | Custom endpoint URL for R2/B2/MinIO |
| `ARIA_LAKE_S3_PREFIX` | `v1/` | Key prefix inside bucket |

---

## 12. Open Questions

| # | Question | Recommendation |
|---|----------|---------------|
| 1 | Should `sessions/` and `datasets/` share the same `aria_data_lake/` root, or be separate directories? | Same root — one env var, simpler config. |
| 2 | Monitor frames: all frames, or only quality-passed frames from MoldTrace `monitor_quality` stage? | Default: quality-passed only. `--include-all-frames` flag for `lake session import` to override. |
| 3 | `--balance-classes`: hard undersample at pull time, or training-time class weights? | Hard undersample at pull time — deterministic and reproducible. Training-time weights as an additional `--class-weights` argument is a future option. |
| 4 | Where do training `runs/` logs go? | `aria_data_lake/runs/<dataset_uuid>/` — keeps dataset and its training logs together. |
| 5 | Should `image_index.jsonl` be rebuilt lazily on every `lake pull`, or maintained incrementally? | Incremental (appended by `session import`); full rebuild on `lake index --rebuild`. Lazy rebuild is expensive at scale. |
| 6 | Do we want a `lake annotate` command that auto-opens Label Studio and pre-configures it for a session? | Desirable in Phase 2+, not blocking Phase 1. |
