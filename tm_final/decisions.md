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

---

### Change log
| Date | Decision | Change |
|------|----------|--------|
| — | — | Initial version drafted from Level 3 spec. |
| Build | D13, D14 | Logged network-access deviations discovered during implementation (no access to pretrained-weight hosts). |
