# System Architecture

## Module boundaries (per Rules.md R1)

```
┌─────────────┐      REST + WebSocket      ┌───────────────────────────┐
│   ui/        │ <───────────────────────> │         main.py            │
│ (React, CDN) │                            │  (FastAPI routes only —    │
└─────────────┘                            │   orchestration, no ML)    │
                                             └─────────────┬───────────┘
                                                            │
                        ┌───────────────────┬───────────────┼───────────────────┐
                        ▼                   ▼               ▼                   ▼
                 ┌─────────────┐    ┌──────────────┐  ┌───────────┐    ┌────────────────┐
                 │   data/      │    │  trainers/    │  │  models/   │    │  inference/     │
                 │ ingestion,   │    │ one module     │  │ artifact   │    │ prediction for  │
                 │ validation,  │    │ per model type,│  │ save/load  │    │ upload + webcam │
                 │ SQLite meta  │    │ common         │  │ only       │    │ (same endpoint) │
                 │              │    │ train()/       │  │            │    │                 │
                 │              │    │ progress()     │  │            │    │                 │
                 └─────────────┘    └──────────────┘  └───────────┘    └────────────────┘
```

`ui/` contains no ML logic — it only calls the REST/WebSocket API and renders
what comes back (R1). `models/` only saves/loads artifacts — the actual
training logic lives in `trainers/`, never in `models/` (R1).

## Data flow: upload → validation → training → inference

1. **Upload** (`POST /api/classes/{name}/images`)
   Browser sends a batch of files → `main.py` calls
   `data/validation.py:validate_image_bytes()` per file (format/size/
   resolution checks, R4) → valid images saved to
   `storage/data/<class_name>/` via `data/storage.py`, and each accepted
   image is recorded in the SQLite `images` table via `data/db.py`. Invalid
   files are reported back individually; the rest of the batch still
   proceeds (R4).

2. **Readiness check** (`GET /api/dataset/status`)
   `data/validation.py:validate_dataset_ready()` checks the per-class image
   counts against the minimum (R3: ≥2 classes, ≥10 images/class by default)
   before the UI will allow training to start.

3. **Training** (`POST /api/train`)
   `main.py` creates a `runs` row (class-name snapshot) and a `run_models`
   row per model type (all `pending`), then starts a background thread that
   runs, in order: `LogisticRegressionTrainer`, `RandomForestTrainer`,
   `CNNTrainer` — all three, every run, per R2. Each trainer:
   - extracts features (`trainers/features.py` for the two classical models;
     raw resized pixel tensors for the CNN),
   - splits held-out validation data,
   - fits the model, calling `progress_cb()` at each meaningful stage —
     including per-epoch for CNN, start/fit/done markers for the
     (atomically-fitting) scikit-learn models, satisfying R5,
   - computes accuracy + confusion matrix on the *held-out* split only (R6),
   - saves the artifact under `storage/models/<run_id>/` via `models/registry.py`.

   Progress events are broadcast over the run's WebSocket connections
   (`main.py:_broadcast`) so the UI never appears frozen, and every model's
   final status (`done`/`failed`) is persisted to SQLite (R9) — a failure in
   one trainer doesn't stop the others, and is never silently dropped from
   the record (R8).

4. **Inference** (`POST /api/predict`)
   Accepts a single image file — from either the upload form or a canvas
   snapshot of a webcam frame — through the same endpoint and code path
   (D11). `inference/predict.py` loads the latest completed run's three
   model artifacts (via `models/registry.py`'s cache, so no retraining is
   needed — R9), runs the image through all three, and returns predictions
   for all three in one response (D12/R7). A model that isn't `done` in this
   run is marked `unavailable` with a reason rather than omitted (R8).

## Persistence layout

```
storage/
├── app.db                        # SQLite: classes, images, runs, run_models
├── data/<class_name>/*.jpg       # uploaded (validated, re-encoded) images
└── models/<run_id>/
    ├── logistic_regression.joblib
    ├── random_forest.joblib
    └── cnn.keras
```

## Why WebSocket over polling (D4)

Training progress is pushed to connected clients the moment each trainer
calls its progress callback, rather than the UI polling on an interval. This
gives near-real-time per-epoch updates for the CNN without the latency or
wasted requests of polling, and leaves room for future bidirectional
messages (e.g. a cancel-training action) without changing the transport.
