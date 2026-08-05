# Rules.md — Teachable Machine–Style Multi-Class Image Classifier

Binding rules derived from the Level 3 spec. Any deviation must be logged in
`decisions.md` with rationale, or listed in `Blocked.md` if it can't be resolved
yet. Rule IDs are referenced from commits/PRs and from `decisions.md`.

---

## R1. Modular file structure is mandatory
The following top-level modules must exist and own their stated responsibility.
No cross-cutting logic may bypass them.
- `data/` — dataset ingestion, storage, validation (format/size/min-class-size)
- `trainers/` — one trainer module per model type (Logistic Regression, Random
  Forest, CNN); trainers must expose a common interface (`train()`, `progress()`,
  `evaluate()`)
- `models/` — model artifact I/O (save/load), not training logic
- `inference/` — prediction endpoints for uploaded image and webcam frame
- `ui/` — frontend app only; no ML logic in the UI layer

## R2. All three model types must be trained and evaluated on every "Train" run
A user action to train must always run Logistic Regression, Random Forest, and
CNN, and produce comparable metrics for all three. Partial runs (e.g. skipping a
model type) are not allowed in v1 unless explicitly toggled by the user.

## R3. Minimum class requirements
- At least 2 classes required to start training.
- Each class must meet a minimum image count (default: 10 images/class,
  configurable) before training is allowed. Enforce in `data/` validation, not in
  the UI alone.

## R4. Image validation is mandatory before storage
- Accepted formats: JPEG, PNG, WebP only.
- Max file size and max/min resolution must be enforced server-side (not just
  client-side) before an image is persisted to `data/`.
- Reject and report invalid files individually; do not fail the whole batch
  upload on one bad file.

## R5. Training progress must be observable live
Every trainer must emit progress events (e.g. epoch/fold + running metric) over
the WebSocket channel. A trainer that cannot report incremental progress (e.g.
scikit-learn `fit()` is atomic) must still emit start/stage/end events so the UI
never appears frozen.

## R6. Evaluation metrics are mandatory and standardized
Every trained model must produce:
- Overall accuracy
- Confusion matrix (raw counts, class-labeled)
Metrics must use a held-out validation split (not training accuracy) — never
report train-set accuracy as the headline metric.

## R7. Predictions must support both image sources
- Upload flow: user selects a file → sent to `inference/`.
- Webcam flow: browser captures a frame → sent to `inference/` via the same
  endpoint contract as upload.
Both must return predictions from **all three** trained models in one response.

## R8. No model may be silently skipped in the results UI
If a model failed to train or is missing, the UI must show its slot as
"unavailable" with a reason — never omit it silently from the side-by-side view.

## R9. Persistence rules
- Every completed training run persists: dataset snapshot reference, model
  artifact(s), metrics, and timestamp.
- Model artifacts must be reloadable without retraining (used for inference).

## R10. Documentation is a deliverable, not optional
- `README.md` must cover setup, run instructions, and how to add a class/train/
  predict.
- A system architecture doc must describe the module boundaries in R1 and data
  flow from upload → validation → training → inference.

## R11. Security & privacy baseline
- Webcam access must be explicit opt-in (browser permission prompt); no
  background capture.
- Uploaded images stay local to the deployment (no third-party upload) unless a
  future decision explicitly changes this.

## R12. No undocumented scope changes
Any requirement not in the original spec (e.g. adding a 4th model type, adding
auth) requires an entry in `decisions.md` before implementation begins.

---

### Rule violations
Any rule violation found in review must be fixed or explicitly downgraded via a
new `decisions.md` entry — it cannot simply be ignored.
