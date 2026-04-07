# ARIA System Integration: MLOps Backend Evaluation

> **Context**: This document evaluates three distinct architectures to replace local/Google Drive storage for ARIA's machine learning datasets and models. It details how each platform integrates with `ARIA_MoldPilot`, `ARIA_MoldTrace`, and `ARIA_MoldVision`.

---

## 1. DagsHub + DVC + MLflow (The "Git for Data" Approach)

DagsHub acts as a centralized AI development platform that leverages open-source formats—Git for code, DVC for data file versioning, MLflow for experiment tracking, and Label Studio for annotations. 

### Pros & Cons
| Pros | Cons |
|------|------|
| Provides a fully configured Label Studio workspace out of the box with direct access to project files. | Introduces a slight learning curve, as the team must learn DVC commands alongside Git. |
| Built to support petabyte-scale datasets while keeping files versioned natively alongside code. | Strict Git-flow for data can feel rigid during rapid, unstructured prototyping phases. |
| Connects to S3-compatible or on-prem object stores, meaning data stays where you choose. | Requires manual mapping of our custom `.mpk` bundle format to MLflow's registry expectations. |

### Integration with ARIA Stack
* **ARIA_MoldVision**: The existing local Label Studio loop is entirely replaced. DagsHub provides the Label Studio UI natively. When running `moldvision train`, the toolchain will push the `checkpoint_best_total.pth` and final `.mpk` bundle to DagsHub’s built-in MLflow registry.
* **ARIA_MoldTrace**: The pipeline will commit new JSONL labeled records, `monitor_geometry`, and JPEG frames directly to a DVC remote repository via the Python client instead of keeping them in local `inputs/` folders.
* **ARIA_MoldPilot**: The `LocalModelRegistryService` will be updated to fetch the `manifest.json` and `.mpk` bundles securely via the MLflow API.

---

## 2. Roboflow (The End-to-End Managed CV Platform)

Roboflow provides an end-to-end interface designed specifically to automate the entire computer vision pipeline from image to inference, supported by comprehensive APIs and SDKs.

### Pros & Cons
| Pros | Cons |
|------|------|
| Best-in-class user interface explicitly tailored for computer vision (bounding boxes, instance segmentation). | Proprietary platform; restricts complete architectural control over storage (vendor lock-in). |
| Powerful model-assisted labeling features to automatically pre-label defects like `Weld_Line` or `Flash`. | Features overlap heavily with `MoldVision`'s existing dataset ingestion and augmentation logic. |

### Integration with ARIA Stack
* **ARIA_MoldVision**: Dataset ingestion (`moldvision dataset ingest`) would be deprecated in favor of Roboflow's API. Training can still utilize the internal RF-DETR PyTorch setup by exporting the dataset from Roboflow in COCO format prior to training.
* **ARIA_MoldTrace**: Instead of saving cropped inspection frames locally, the `extract_components` and `defects` stages push imagery directly into Roboflow's REST API for batch accumulation and labeling.
* **ARIA_MoldPilot**: Edge inference continues to rely on ONNX. Roboflow acts purely as the upstream dataset manager; models are still exported and wrapped in the `.mpk` format for `StartupAssistant` deployment.

---

## 3. AWS S3 + Weights & Biases (The Scalable Cloud Stack)

This approach fully commits to AWS for storage architecture, utilizing Weights & Biases (W&B) as the dedicated platform to track experiments, version datasets, and evaluate model performance.

### Pros & Cons
| Pros | Cons |
|------|------|
| W&B artifact references can point directly to data in systems like AWS S3 without duplicating the actual files. | Requires overhead to set up and manage AWS IAM policies and S3 bucket lifecycles. |
| Deep, seamless integration with PyTorch via `wandb.init()` and `wandb.log()` for tracking configurations, metrics, and gradients. | Requires the team to self-host and maintain the Label Studio instance linked to the S3 bucket. |

### Integration with ARIA Stack
* **ARIA_MoldTrace**: Fully actualizes the existing `S3StorageBackend` stub. When `create_session.py` runs, it uploads the `raw/` videos and JSONL files directly to `s3://aria-sessions/`.
* **ARIA_MoldVision**: The local Label Studio instance is configured to use S3 as its target storage. During `moldvision train`, `wandb.init()` logs the PyTorch training metrics and registers the final `.mpk` bundle into the W&B model registry.
* **ARIA_MoldPilot**: `OnnxInferenceService` periodically queries the W&B API to check for new model versions on the `stable` channel. It dynamically fetches the latest `.mpk` bundle directly from the W&B model registry or securely from the linked S3 bucket.
