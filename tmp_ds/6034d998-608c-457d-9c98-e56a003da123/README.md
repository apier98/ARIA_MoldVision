# Dataset 6034d998-608c-457d-9c98-e56a003da123

Name: collaudo_andrea_namecheck

COCO-format data should be placed under `coco/train`, `coco/valid`, and `coco/test`.

Typical workflow:
1) Put images in `raw/` and labels in `yolo/` (optional)
2) Convert to COCO with the CLI (yolo->coco)
3) Train with the CLI (train)

Optional: drop mixed label exports into `labels_inbox/` and run the CLI ingest command.
