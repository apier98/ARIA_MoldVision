"""Pull engine for the ARIA Data Lake.

``lake_pull`` assembles a balanced, split training dataset (``datasets/<UUID>/``)
from labeled images across sessions, then writes full provenance into
``METADATA.json → lake_pull_provenance``.

The output layout is the same ``datasets/<UUID>/`` format that the rest of the
MoldVision pipeline (``validate``, ``train``, ``export``, ``bundle``) consumes.
"""
from __future__ import annotations

import json
import random
import shutil
import uuid as _uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .datasets import create_dataset
from .lake import (
    DETECT_CLASSES,
    LABEL_STATUS_BACKGROUND,
    LABEL_STATUS_HARD_NEGATIVE,
    LABEL_STATUS_LABELED,
    SEG_CLASSES,
    LakeConfig,
    filter_index,
    load_index,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ──────────────────────────────────────────────────────────────────────────────
# COCO merge helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty_coco(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "info": {"description": "ARIA Data Lake pull", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def _merge_session_coco(
    session_coco_path: Path,
    selected_fnames: List[str],
    existing: Dict[str, Any],
    next_img_id: int,
    next_ann_id: int,
) -> Tuple[Dict[str, Any], int, int]:
    """Merge images/annotations from *session_coco_path* for *selected_fnames*.

    Mutates *existing* in place and returns (existing, next_img_id, next_ann_id).
    """
    if not session_coco_path.exists():
        return existing, next_img_id, next_ann_id

    try:
        src = json.loads(session_coco_path.read_text(encoding="utf-8"))
    except Exception:
        return existing, next_img_id, next_ann_id

    fname_set = set(selected_fnames)
    img_id_map: Dict[int, int] = {}

    for im in src.get("images", []):
        if im.get("file_name") not in fname_set:
            continue
        new_im = {k: v for k, v in im.items() if k != "id"}
        new_im["id"] = next_img_id
        img_id_map[im["id"]] = next_img_id
        existing["images"].append(new_im)
        next_img_id += 1

    for ann in src.get("annotations", []):
        if ann.get("image_id") not in img_id_map:
            continue
        new_ann = {k: v for k, v in ann.items() if k not in ("id", "image_id")}
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = img_id_map[ann["image_id"]]
        existing["annotations"].append(new_ann)
        next_ann_id += 1

    return existing, next_img_id, next_ann_id


def _add_empty_annotation_images(
    manifest_path: Path,
    lake_root: Path,
    existing: Dict[str, Any],
    next_img_id: int,
) -> Tuple[Dict[str, Any], int]:
    """Add empty-annotation images from a pool manifest (hard-negatives / backgrounds)."""
    if not manifest_path.exists():
        return existing, next_img_id

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        rel_path = entry.get("rel_path", "")
        fname = Path(rel_path).name
        # Try to get image dimensions from the file
        try:
            from .datasets import image_size
            abs_p = lake_root / rel_path
            w, h = image_size(abs_p) if abs_p.exists() else (0, 0)
        except Exception:
            w, h = 0, 0
        existing["images"].append({"id": next_img_id, "file_name": fname, "width": w, "height": h})
        next_img_id += 1

    return existing, next_img_id


# ──────────────────────────────────────────────────────────────────────────────
# Class-based image balancing
# ──────────────────────────────────────────────────────────────────────────────

def _count_classes_per_image(
    merged_coco: Dict[str, Any],
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Return {cat_id: [image_id, ...]} and {cat_id: name}."""
    cat_name = {c["id"]: c["name"] for c in merged_coco.get("categories", [])}
    img_cats: Dict[int, set] = defaultdict(set)
    for ann in merged_coco.get("annotations", []):
        img_cats[ann["image_id"]].add(ann.get("category_id"))
    cat_imgs: Dict[int, List[int]] = defaultdict(list)
    for img_id, cats in img_cats.items():
        for c in cats:
            cat_imgs[c].append(img_id)
    # background images (no annotations) — represent as category -1
    ann_img_ids = set(img_cats.keys())
    for im in merged_coco.get("images", []):
        if im["id"] not in ann_img_ids:
            cat_imgs[-1].append(im["id"])
    return dict(cat_imgs), cat_name


def _balance_coco(
    merged_coco: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Undersample over-represented classes so each class is ≤ 2× the rarest.

    Background images (empty annotations) are treated proportionally.
    """
    cat_imgs, _cat_name = _count_classes_per_image(merged_coco)

    # Exclude background (-1) from the rarest-class calculation
    non_bg_cats = {k: v for k, v in cat_imgs.items() if k != -1}
    if not non_bg_cats:
        return merged_coco

    rarest_count = min(len(v) for v in non_bg_cats.values())
    target = rarest_count * 2

    rng = random.Random(seed)
    keep_ids: set = set()

    for cat_id, img_ids in non_bg_cats.items():
        if len(img_ids) > target:
            keep_ids.update(rng.sample(img_ids, target))
        else:
            keep_ids.update(img_ids)

    # Keep background images proportionally (same fraction as the overall retention)
    original_count = len(merged_coco.get("images", []))
    bg_ids = cat_imgs.get(-1, [])
    if bg_ids and original_count:
        bg_target = max(1, int(len(bg_ids) * len(keep_ids) / max(1, original_count - len(bg_ids))))
        if len(bg_ids) > bg_target:
            keep_ids.update(rng.sample(bg_ids, bg_target))
        else:
            keep_ids.update(bg_ids)

    filtered_images = [im for im in merged_coco["images"] if im["id"] in keep_ids]
    filtered_anns = [a for a in merged_coco["annotations"] if a["image_id"] in keep_ids]

    return {**merged_coco, "images": filtered_images, "annotations": filtered_anns}


# ──────────────────────────────────────────────────────────────────────────────
# Train/valid split
# ──────────────────────────────────────────────────────────────────────────────

def _split_coco(
    merged_coco: Dict[str, Any],
    train_ratio: float,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images = list(merged_coco.get("images", []))
    anns = list(merged_coco.get("annotations", []))
    cats = merged_coco.get("categories", [])

    rng = random.Random(seed)
    rng.shuffle(images)
    cut = int(len(images) * train_ratio)
    train_imgs = images[:cut]
    valid_imgs = images[cut:]

    train_ids = {im["id"] for im in train_imgs}
    valid_ids = {im["id"] for im in valid_imgs}

    train_anns = [a for a in anns if a["image_id"] in train_ids]
    valid_anns = [a for a in anns if a["image_id"] in valid_ids]

    def _build(imgs: List, ann_list: List) -> Dict[str, Any]:
        return {"info": merged_coco.get("info", {}), "licenses": [], "images": imgs, "annotations": ann_list, "categories": cats}

    return _build(train_imgs, train_anns), _build(valid_imgs, valid_anns)


# ──────────────────────────────────────────────────────────────────────────────
# Dry-run report
# ──────────────────────────────────────────────────────────────────────────────

def _bar(n: int, total: int, width: int = 22) -> str:
    if total == 0:
        return ""
    filled = int(n / total * width)
    return "█" * filled + "░" * (width - filled)


def _print_class_distribution(merged_coco: Dict[str, Any], title: str) -> None:
    cat_name = {c["id"]: c["name"] for c in merged_coco.get("categories", [])}
    counter: Counter = Counter()
    for ann in merged_coco.get("annotations", []):
        counter[ann.get("category_id", -1)] += 1
    bg_count = len([im for im in merged_coco["images"] if
                    not any(a["image_id"] == im["id"] for a in merged_coco["annotations"])])
    if bg_count:
        counter[-1] = bg_count

    if not counter:
        return

    total = max(counter.values())
    print(f"\n{title}")
    for cat_id in sorted(counter.keys()):
        name = cat_name.get(cat_id, f"class_{cat_id}") if cat_id != -1 else "(background)"
        cnt = counter[cat_id]
        print(f"  {name:<18} {cnt:>6}  {_bar(cnt, total)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pull function
# ──────────────────────────────────────────────────────────────────────────────

def lake_pull(
    cfg: LakeConfig,
    *,
    task: str,
    sessions: Optional[List[str]] = None,
    all_sessions: bool = False,
    machine_id: Optional[str] = None,
    mold_id: Optional[str] = None,
    part_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    marker: Optional[str] = None,
    include_hard_negatives: bool = False,
    include_backgrounds: bool = False,
    max_per_session: Optional[int] = None,
    min_per_session: int = 0,
    balance_classes: bool = False,
    min_per_class: Optional[int] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    dataset_uuid: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Assemble a training dataset from labeled lake images.

    Returns the dataset UUID, or None for dry-run.
    """
    storage = cfg.storage()
    records = load_index(cfg.root)

    session_ids = sessions if (sessions and not all_sessions) else None

    labeled_records = filter_index(
        records,
        task=task,
        session_ids=session_ids,
        label_status=LABEL_STATUS_LABELED,
        machine_id=machine_id,
        mold_id=mold_id,
        part_id=part_id,
        from_date=from_date,
        to_date=to_date,
        marker=marker,
    )

    if not labeled_records:
        raise RuntimeError("No labeled images match the given filters. Did you commit any label batches?")

    # Group by session
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in labeled_records:
        by_session[r["session_id"]].append(r)

    # Drop sessions below min_per_session
    skipped_sessions: List[str] = []
    if min_per_session > 0:
        for sid in list(by_session.keys()):
            if len(by_session[sid]) < min_per_session:
                skipped_sessions.append(sid)
                del by_session[sid]

    if not by_session:
        raise RuntimeError(f"No sessions remain after applying --min-per-session {min_per_session}.")

    # Apply max_per_session
    rng = random.Random(seed)
    session_provenance: List[Dict[str, Any]] = []
    selected_records: List[Dict[str, Any]] = []

    for sid in sorted(by_session.keys()):
        recs = by_session[sid]
        total_labeled = len(recs)
        if max_per_session and len(recs) > max_per_session:
            chosen = rng.sample(recs, max_per_session)
        else:
            chosen = list(recs)
        selected_records.extend(chosen)
        session_provenance.append({
            "session_id":          sid,
            "frames_selected":     len(chosen),
            "frames_total_labeled": total_labeled,
        })

    # Determine categories
    classes = DETECT_CLASSES if task == "detect" else SEG_CLASSES
    categories = [{"id": i, "name": name, "supercategory": ""} for i, name in enumerate(classes)]

    # Collect COCO data — merge across sessions
    merged_coco = _empty_coco(categories)
    next_img_id = 1
    next_ann_id = 1

    ann_task_dir = "detect" if task == "detect" else "seg"

    per_session_fnames: Dict[str, List[str]] = defaultdict(list)
    for r in selected_records:
        per_session_fnames[r["session_id"]].append(Path(r["rel_path"]).name)

    for sid, fnames in per_session_fnames.items():
        ann_path = storage.abs_path(f"sessions/{sid}/annotations/{ann_task_dir}/_annotations.coco.json")
        merged_coco, next_img_id, next_ann_id = _merge_session_coco(
            ann_path, fnames, merged_coco, next_img_id, next_ann_id
        )

    # Append pool images
    hn_count = bg_count = 0
    if include_hard_negatives:
        before = len(merged_coco["images"])
        merged_coco, next_img_id = _add_empty_annotation_images(
            cfg.root / "pools" / "hard_negatives" / "manifest.jsonl",
            cfg.root, merged_coco, next_img_id,
        )
        hn_count = len(merged_coco["images"]) - before

    if include_backgrounds:
        before = len(merged_coco["images"])
        merged_coco, next_img_id = _add_empty_annotation_images(
            cfg.root / "pools" / "backgrounds" / "manifest.jsonl",
            cfg.root, merged_coco, next_img_id,
        )
        bg_count = len(merged_coco["images"]) - before

    # min_per_class guard
    if min_per_class is not None:
        cat_counter: Counter = Counter()
        for ann in merged_coco.get("annotations", []):
            cat_counter[ann.get("category_id", -1)] += 1
        for cat in categories:
            cnt = cat_counter.get(cat["id"], 0)
            if cnt < min_per_class:
                raise RuntimeError(
                    f"Class '{cat['name']}' has only {cnt} annotations, but --min-per-class is {min_per_class}. "
                    "Add more labeled data or lower the threshold."
                )

    # Balance classes
    if balance_classes:
        _print_class_distribution(merged_coco, "Class distribution (before --balance-classes):")
        merged_coco = _balance_coco(merged_coco, seed)

    # Print dry-run report
    if dry_run:
        print(f"\nlake pull DRY RUN — task={task}")
        print("─" * 66)
        print("Sessions (after filters):")
        for sp in session_provenance:
            cap_note = f"→ capped at {max_per_session}" if max_per_session and sp["frames_selected"] < sp["frames_total_labeled"] else f"→ all {sp['frames_selected']} included"
            print(f"  {sp['session_id']:<44}  {sp['frames_total_labeled']:>4} labeled  {cap_note}")
        for sid in skipped_sessions:
            print(f"  {sid:<44}  skipped (< --min-per-session {min_per_session})")

        _print_class_distribution(merged_coco, "Class distribution:")
        if hn_count or bg_count:
            print(f"\nPools: hard_negatives={hn_count}  backgrounds={bg_count}")

        train_c, valid_c = _split_coco(merged_coco, train_ratio, seed)
        print(f"\nTrain: {len(train_c['images'])} images / {len(train_c['annotations'])} annotations")
        print(f"Valid: {len(valid_c['images'])} images / {len(valid_c['annotations'])} annotations")
        print("\nProvenance will be saved to METADATA.json → lake_pull_provenance")
        print("─" * 66)
        print("Run without --dry-run to create dataset.")
        return None

    # Split
    train_coco, valid_coco = _split_coco(merged_coco, train_ratio, seed)

    # Create dataset
    ds_root = dataset_root or (cfg.root / "datasets")
    layout = create_dataset(
        root=ds_root,
        uuid_str=dataset_uuid,
        name=dataset_name or f"lake-{task}-{datetime.utcnow().strftime('%Y%m%d')}",
        force=bool(dataset_uuid),
        no_readme=False,
        class_names=classes,
    )
    dd = layout.dataset_dir

    # Write COCO splits
    for split_name, split_coco in (("train", train_coco), ("valid", valid_coco)):
        split_dir = dd / "coco" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(split_coco, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Copy images into datasets/<UUID>/raw/ and into coco splits
    raw_dir = dd / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _copy_image(fname: str, session_id: str, frame_type: str) -> None:
        subdir = "inspection_frames" if frame_type == "inspection" else "monitor_frames"
        src = storage.abs_path(f"sessions/{session_id}/{subdir}/{fname}")
        if src.exists():
            dst = raw_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)

    for r in selected_records:
        _copy_image(Path(r["rel_path"]).name, r["session_id"], r.get("frame_type", "inspection"))

    # Copy images referenced in pool manifest
    all_coco_images = {im["file_name"] for im in merged_coco.get("images", [])}
    for pool_rel in ("pools/hard_negatives/manifest.jsonl", "pools/backgrounds/manifest.jsonl"):
        mp = cfg.root / pool_rel
        if not mp.exists():
            continue
        for line in mp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            fname = Path(entry.get("rel_path", "")).name
            if fname in all_coco_images:
                src = storage.abs_path(entry["rel_path"])
                if src.exists():
                    dst = raw_dir / fname
                    if not dst.exists():
                        shutil.copy2(src, dst)

    # Copy images into coco split folders
    for split_name, split_coco in (("train", train_coco), ("valid", valid_coco)):
        split_dir = dd / "coco" / split_name
        for im in split_coco.get("images", []):
            fname = im.get("file_name", "")
            src = raw_dir / fname
            dst = split_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    # Write provenance into METADATA.json
    md_path = dd / "METADATA.json"
    md: Dict[str, Any] = {}
    if md_path.exists():
        try:
            md = json.loads(md_path.read_text(encoding="utf-8"))
        except Exception:
            md = {}

    md["lake_pull_provenance"] = {
        "task":        task,
        "seed":        seed,
        "train_ratio": train_ratio,
        "sessions":    session_provenance,
        "filters": {k: v for k, v in {
            "session_ids": sessions,
            "machine_id":  machine_id,
            "mold_id":     mold_id,
            "part_id":     part_id,
            "from":        from_date,
            "to":          to_date,
            "marker":      marker,
        }.items() if v is not None},
        "distribution": {
            "max_per_session": max_per_session,
            "min_per_session": min_per_session,
            "balance_classes": balance_classes,
            "min_per_class":   min_per_class,
        },
        "pools": {"hard_negatives": hn_count, "backgrounds": bg_count},
    }
    md_path.write_text(json.dumps(md, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print distribution report
    print(f"\nDataset created: {dd}")
    print(f"  UUID:   {layout.uuid}")
    print(f"  Task:   {task}")
    print(f"  Train:  {len(train_coco['images'])} images / {len(train_coco['annotations'])} annotations")
    print(f"  Valid:  {len(valid_coco['images'])} images / {len(valid_coco['annotations'])} annotations")
    _print_class_distribution(merged_coco, "  Class distribution:")
    print(f"\n  Provenance → {md_path}")

    return layout.uuid
