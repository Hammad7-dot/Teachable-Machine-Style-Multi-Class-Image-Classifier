# Decisions.md — Teachable Machine–Style Multi-Class Image Classifier

Spec-driven log of architecture and technical decisions. Every decision ties back
to a requirement in the Level 3 spec (`level3_ai_teachable_machine_proper.pdf`).

Format: **Decision → Rationale → Alternatives considered → Status**

---

## D1. Overall architecture: Python backend + JS/React frontend
- **Rationale:** Spec requires training real ML models (Logistic Regression, Random
  Forest, CNN/Keras) and a live webcam UI. Python is required for
  scikit-learn/TensorFlow; browser is required for webcam capture. A REST/WebSocket
  API bridges the two.
- **Alternatives considered:** Pure browser ML (TensorFlow.js only) — rejected
  because spec explicitly asks for Logistic Regression and Random Forest (not just
  CNN), which are awkward/unsupported in TF.js.
- **Status:** Accepted.

## D2. Backend framework: FastAPI
- **Rationale:** Async support needed for live training-progress streaming
  (WebSocket/SSE), good fit with scikit-learn/TensorFlow, automatic OpenAPI docs
  support the "documentation" deliverable.
- **Alternatives considered:** Flask (no native async/WebSocket), Django (too heavy
  for this scope).
- **Status:** Accepted.

## D3. Frontend framework: React
- **Rationale:** Component model maps cleanly to spec's UI needs: class manager,
  upload/webcam capture, training progress bars, side-by-side prediction panels.
- **Alternatives considered:** Vanilla JS (harder to maintain multi-model
  side-by-side views), Streamlit (poor webcam support, not "production web app"-like).
- **Status:** Accepted.

## D4. Live training progress: WebSocket, not polling
- **Rationale:** Spec requires "show training progress live." WebSocket gives
  per-epoch/per-fold updates with lower latency and less server load than polling.
- **Alternatives considered:** HTTP polling (simpler but laggy/wasteful),
  Server-Sent Events (viable alt, WebSocket chosen for bidirectional future use,
  e.g. cancel-training).
- **Status:** Accepted.

## D5. CNN implementation: TensorFlow/Keras, transfer learning by default
- **Rationale:** Spec explicitly names TensorFlow/Keras. Small user-uploaded
  datasets are typical of Teachable-Machine-style tools, so a frozen pretrained
  backbone (e.g. MobileNetV2) with a small trainable head gives usable accuracy
  without huge data.
- **Alternatives considered:** Training CNN from scratch — rejected as default
  (poor accuracy on small datasets), but keep as an "advanced" toggle.
- **Status:** Accepted.

## D6. Classical models: scikit-learn Logistic Regression + Random Forest on
    extracted feature vectors (not raw pixels)
- **Rationale:** Raw flattened pixels perform poorly and don't scale. Use a
  fixed-size embedding (e.g. pretrained CNN penultimate layer or HOG/color
  histogram fallback) as shared input features for both classical models.
- **Alternatives considered:** Raw pixel vectors (rejected — poor accuracy, huge
  dimensionality), PCA-reduced pixels (kept as fallback if no GPU/pretrained model
  available).
- **Status:** Accepted.

## D7. Modular file structure follows spec verbatim
- **Rationale:** Spec explicitly lists `data/`, `trainers/`, `models/`,
  `inference/`, `ui/`. We keep these as top-level backend packages, with `ui/`
  mapped to the React app.
- **Status:** Accepted (non-negotiable per spec — see Rules.md R1).

## D8. Dataset & model persistence: local filesystem + SQLite metadata
- **Rationale:** "Model files + saved datasets" is an explicit deliverable.
  SQLite is zero-config and sufficient for single-user/local deployment; images
  stored on disk under `data/<class_name>/`, models under `models/<run_id>/`.
- **Alternatives considered:** Full RDBMS (Postgres) — deferred as unnecessary
  complexity for the current scope; revisit if multi-user support is added.
- **Status:** Accepted.

## D9. Validation layer lives in `data/` module, enforced before training starts
- **Rationale:** Spec requires image format/size checks and minimum class size.
  Centralizing validation avoids duplicating checks across three trainers.
- **Status:** Accepted.

## D10. Evaluation metrics: accuracy + confusion matrix, computed once per model
- **Rationale:** Directly matches spec. Confusion matrix rendered client-side from
  a JSON matrix returned by the API (avoids server-side image rendering
  dependency).
- **Status:** Accepted.

## D11. Prediction sources: uploaded image and webcam frame use the same
    inference endpoint
- **Rationale:** Spec requires both prediction paths. Using one endpoint
  (accepting an image blob regardless of source) avoids code duplication and
  guarantees identical preprocessing for both paths.
- **Status:** Accepted.

## D12. Side-by-side predictions: single API call returns predictions from all
    three models
- **Rationale:** Spec requires displaying all models' predictions together.
  Batching into one response avoids UI race conditions from three separate calls.
- **Status:** Accepted.

## D13. Deviation from D6: classical-model features use HOG + color histogram, not pretrained CNN embedding
- **Rationale:** D6 proposed a pretrained CNN embedding as the primary feature
  source for Logistic Regression / Random Forest, with HOG/color-histogram as
  fallback. The build environment has no network access to pretrained-weight
  hosts (e.g. storage.googleapis.com), so the embedding path is unreachable.
  The documented fallback is used as the actual default.
- **Alternatives considered:** Block on network access being granted — rejected,
  since the fallback was already an accepted part of D6 and produces working
  models (100% held-out accuracy on the smoke-test dataset).
- **Status:** Accepted as current default. Revisit if pretrained-weight download
  access becomes available — swapping in an embedding extractor only requires
  changing `trainers/features.py`; the trainer/inference code is agnostic to
  feature-vector origin.

## D14. Deviation from D5: CNN trains from scratch, not via frozen pretrained backbone
- **Rationale:** Same root cause as D13 — no network access to download
  MobileNetV2 (or any pretrained) weights in this environment. A small CNN
  (3 conv blocks, global average pooling, dropout) is trained from scratch
  instead, sized to reduce overfitting on small datasets.
- **Alternatives considered:** Ship without a working CNN — rejected, CNN is
  explicitly required (R2). Larger from-scratch architecture — rejected as
  more prone to overfitting on tens-to-low-hundreds-of-images-per-class
  datasets (per B2's assumed scale).
- **Status:** Accepted as current default. Revisit if pretrained weights become
  available; `trainers/cnn.py` isolates the model-building code so swapping in
  a `keras.applications.MobileNetV2(weights="imagenet", include_top=False)`
  base is a localized change.

## D15. Added /api/dashboard aggregate endpoint, and sidebar+dashboard navigation in both UIs
- **Rationale:** Both frontends needed an overview landing page (dataset
  size, run history, latest run's per-model accuracy) rather than requiring
  users to piece that together from the Classes/Train/Results tabs
  individually. Rather than have each frontend call three separate
  endpoints (`/api/classes`, `/api/dataset/status`, `/api/runs`) and derive
  the summary client-side, added one backend endpoint
  (`GET /api/dashboard`) that computes it once, keeping the "no ML/business
  logic in the UI" rule (R1) intact for the aggregation logic too.
- **Alternatives considered:** Derive the dashboard purely client-side from
  existing endpoints — rejected as needlessly duplicating aggregation logic
  in two UIs (React and Streamlit) instead of once in the backend.
- **Status:** Accepted. Both `ui/index.html` (sidebar nav, replacing the old
  top-tab layout) and `streamlit_app.py` (native `st.sidebar`) consume this
  endpoint for their Dashboard page.

## D16. Bug fixes found during review
- **WebSocket progress race condition:** `POST /api/train` started the
  background training thread immediately, but the frontend only opened its
  WebSocket connection after that response returned. Any progress events
  fired in that gap (commonly the fast-finishing Logistic Regression
  model's `start`/`fit`/`done` events) were silently dropped, since
  `_broadcast` only sent to already-connected clients. Fixed by buffering
  every broadcast event per run (capped at the most recent 500) in
  `main.py` and replaying that buffer to a client immediately after it
  connects. Verified with a WebSocket client that deliberately connects
  1.5s after training starts — it now receives all 3 models' `start`
  events instead of only later ones.
- **Streamlit Train page blocked the whole app:** the polling loop was a
  `while True: ...; time.sleep(2)` inside a single script execution. In
  Streamlit that blocks the entire session — no other widget on any page,
  including the sidebar nav, can respond until the loop exits, since
  Streamlit processes one script run at a time per session. Rewritten to
  do exactly one status check per script run, ending with an explicit
  `st.rerun()` — the standard non-blocking Streamlit polling pattern, since
  each rerun is its own short, fresh execution rather than one long-lived
  blocked one.
- **Streamlit "Go train a model" button was a no-op:** it set
  `st.session_state["_nav_hint"]` and called `st.rerun()`, but the sidebar
  radio never read that key, so clicking it silently did nothing. Fixed by
  giving the radio an explicit `key="_nav_radio"` and having the button set
  that same session-state key directly before rerunning, since Streamlit
  widgets with a `key` are driven entirely by that session-state entry
  after their first render.
- **Status:** All three fixed and verified in `main.py`/`streamlit_app.py`.

## D17. Fixed a portability bug: absolute paths persisted in the database
- **Found via:** cloning the actual pushed repo and testing it fresh (rather
  than only testing in the original build environment), against real data
  the user had trained (Rock Paper Scissors, 180 images, 3 models).
- **Bug:** `run_models.artifact_path` and `images.path` both stored the
  *absolute* path returned at write time (e.g.
  `D:\Download\teachable-machine-classifier\tm_final\storage\...`). Because
  `storage/` was committed to git (see below), cloning the repo onto a
  different machine, OS, or directory broke every prediction (model
  artifacts unreachable) and any new training run (source images
  unreachable) -- even though the actual files existed right there in the
  cloned `storage/` folder, just at a different absolute path than the one
  recorded in the database.
- **Rationale for the fix:** absolute paths should never be persisted for
  anything under `storage/`, since that directory's location relative to
  the app is the only thing that should matter. Store just the filename;
  reconstruct the full path at read time from the current
  `MODELS_ROOT`/`DATA_ROOT` (both already computed relative to the
  codebase location, so they're correct on whatever machine is running).
  A backward-compatible fallback (`_resolve_artifact_path` in
  `inference/predict.py`) extracts the filename from legacy absolute paths
  --- including Windows-style ones, which `pathlib.Path(...).name` does
  NOT parse correctly on Linux/Mac since backslash isn't a path separator
  on POSIX --- so already-trained runs self-heal without retraining.
- **Alternatives considered:** Require retraining after every clone --
  rejected, unnecessarily destructive when the artifacts are already
  present and valid. Store paths relative to the repo root instead of just
  a filename -- rejected as more fragile (breaks if the repo is moved
  without `storage/` moving with it, e.g. two clones sharing one storage
  volume, which the Docker setup already does).
- **Also fixed:** added `.gitignore` (excluding `storage/`, `__pycache__/`)
  and untracked both from git, since a committed `storage/` is what let a
  machine-specific absolute path leak into version control in the first
  place. Also removed an accidental nested `tm_final/` copy of the whole
  project that had been committed (looks like a delivered zip got
  extracted inside the repo before an early commit).
- **Verified:** against the repo's real pre-existing data -- all 3 models
  on the existing (broken) run now predict correctly with zero retraining;
  a brand-new training run also completes and predicts correctly, storing
  bare filenames as designed.
- **Status:** Fixed in `main.py` and `inference/predict.py`.

---

### Change log
| Date | Decision | Change |
|------|----------|--------|
| — | — | Initial version drafted from Level 3 spec. |
| Build | D13, D14 | Logged network-access deviations discovered during implementation (no access to pretrained-weight hosts). |
| Build | D15, D16 | Added dashboard endpoint + sidebar/dashboard UI in both frontends; fixed WebSocket race condition and two Streamlit navigation/blocking bugs. |
| Build | D17 | Found and fixed a real portability bug via a clone-and-test pass against the user's actual pushed repo and real trained data: absolute paths persisted in the database broke predictions/retraining after cloning. Added `.gitignore`; untracked `storage/`; removed an accidentally-committed nested project copy.
