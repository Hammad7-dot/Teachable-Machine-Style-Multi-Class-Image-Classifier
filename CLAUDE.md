# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local Teachable-Machine-style image classifier: create classes, upload images, train
Logistic Regression + Random Forest + CNN together, compare results, then predict from
an uploaded image or webcam frame. FastAPI backend; two independent frontends (React via
CDN, and Streamlit) talk to the same REST/WebSocket API.

This project was built against a binding spec. Before changing behavior, read:
- `Rules.md` — binding requirements (R1–R12), referenced by ID elsewhere.
- `decisions.md` — architecture decisions with rationale (D1–D17); log any new
  non-trivial decision here in the same **Decision → Rationale → Alternatives → Status** format.
- `Blocked.md` — resolved/open assumptions where the spec was silent.
- `docs/architecture.md` — module boundaries and the upload→validation→training→inference data flow.

Any requirement not in the original spec (new model type, auth, etc.) requires a
`decisions.md` entry *before* implementation (R12). Any rule violation found in review
must be fixed or explicitly downgraded via a new `decisions.md` entry, not silently ignored.

## Commands

```bash
pip install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000   # backend + React UI at http://127.0.0.1:8000
streamlit run streamlit_app.py                              # second UI at http://127.0.0.1:8501 (backend must already be running)
docker compose up --build                                   # both, containerized, shared storage volume
```

No test suite exists in this repo currently.

Bulk-load a dataset instead of clicking through uploads (backend must be running):
```bash
python3 scripts/upload_folder.py /path/to/dataset_root   # dataset_root/<class_name>/*.jpg
pip install -r requirements-optional.txt
python3 scripts/download_rock_paper_scissors.py --limit-per-class 60
```

## Architecture

Module boundaries are mandatory (R1) — no cross-cutting logic bypasses them:

- `main.py` — FastAPI routes, orchestration, WebSocket broadcast. **No ML logic.**
- `data/` — `db.py` (SQLite metadata), `storage.py` (filesystem I/O under `storage/data/`),
  `validation.py` (format/size/min-class-size checks — server-side, per file, per R4).
- `trainers/` — one module per model type (`logistic_regression.py`, `random_forest.py`,
  `cnn.py`), all implementing the common interface in `base.py` (`train()`, `progress()`,
  `evaluate()` via `TrainResult`/`emit()`). `features.py` provides HOG + color-histogram
  feature extraction shared by the two classical models.
- `models/registry.py` — artifact save/load and a cache only. **No training logic.**
- `inference/predict.py` — prediction for both upload and webcam paths, same code path
  and same endpoint contract (D11); returns all three models' predictions in one response
  (D12/R7); marks a non-`done` model `unavailable` with a reason instead of omitting it (R8).
- `ui/index.html` — React (CDN-loaded, no build step). No ML/business logic in the UI layer.
- `streamlit_app.py` — alternate UI, same four-page flow (Dashboard/Classes & Data/Train/
  Results/Predict), polls run status every 2s instead of using WebSocket (Streamlit has no
  WS client) — one status check per script run ending in `st.rerun()`, never a blocking loop.

### Data flow

1. **Upload** `POST /api/classes/{name}/images` → `data/validation.py` validates each file
   individually (bad files are rejected without failing the batch) → accepted images saved
   via `data/storage.py`, recorded in SQLite via `data/db.py`.
2. **Readiness** `GET /api/dataset/status` checks counts against R3 minimums (≥2 classes,
   ≥10 images/class by default, `data/validation.py:DEFAULT_MIN_IMAGES_PER_CLASS`).
3. **Training** `POST /api/train` creates a `runs` row + per-model `run_models` rows, then a
   background thread runs all three trainers in order (R2 — never partial). Each trainer
   calls `progress_cb()` at meaningful stages (per-epoch for CNN; start/fit/done for the
   atomically-fitting sklearn models), computes accuracy + confusion matrix on a held-out
   split only (R6 — never training accuracy), and saves its artifact via `models/registry.py`.
   A failure in one trainer doesn't stop the others and is recorded, not dropped (R8/R9).
   Progress is broadcast over `/ws/train/{run_id}`; events are buffered per-run (last 500)
   and replayed to late-joining clients to close the race where a fast model (e.g. Logistic
   Regression) finishes before the frontend's WebSocket connects.
4. **Inference** `POST /api/predict` loads the latest completed run's three artifacts (via
   `models/registry.py`'s cache — no retraining needed) and returns all three predictions.

### Persistence

```
storage/
├── app.db                        # SQLite: classes, images, runs, run_models
├── data/<class_name>/*.jpg       # validated, re-encoded images
└── models/<run_id>/{logistic_regression.joblib, random_forest.joblib, cnn.keras}
```

`storage/` is gitignored and created at runtime — never commit it. **Only ever persist bare
filenames for anything under `storage/`, never absolute paths** — reconstruct full paths at
read time from `MODELS_ROOT`/`DATA_ROOT` (computed relative to the codebase). A previously
committed `storage/` baked one machine's absolute paths into `app.db`, breaking every
prediction and retraining after cloning elsewhere (see D17); `inference/predict.py`'s
`_resolve_artifact_path` self-heals legacy absolute-path rows (including Windows paths,
which `Path(...).name` does not parse correctly on POSIX).

### Known deviations from the original spec (see decisions.md for full rationale)

- CNN trains from scratch and classical-model features are HOG + color histogram, not a
  pretrained-CNN embedding / frozen backbone (D13/D14) — the build environment has no
  network access to pretrained-weight hosts. `trainers/features.py` and `trainers/cnn.py`
  isolate this so swapping in a pretrained embedding/backbone later is localized.
- Single-user, local-only: no auth, no per-user data isolation (B4).
- Webcam prediction is capture-then-predict, not continuous live-stream (B6).

## Known repo hygiene issue

There is a nested `tm_final/` copy of the entire project living inside the repo root
(duplicating `main.py`, `data/`, `trainers/`, etc.) and a stray root-level `predict.py`
duplicating `inference/predict.py`. `decisions.md` (D17) records these as having been
removed, but they are present in the working tree — treat the top-level modules as the
real ones and confirm with the user before touching or deleting the duplicates.
