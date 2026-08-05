# Blocked.md — Teachable Machine–Style Multi-Class Image Classifier

Open questions and blockers that must be resolved before (or during) build. Each
item: **Blocker → Why it blocks → Who/what unblocks it → Current workaround**.
Move resolved items to the bottom "Resolved" section instead of deleting them.

---

_(no open blockers — all items resolved via documented defaults during build;
see Resolved section below. Reopen any item if the underlying assumption is
challenged.)_

---

## Resolved

### B1. Minimum class size threshold is unspecified
- **Resolution:** Defaulted to 10 images/class, configurable
  (`data/validation.py:DEFAULT_MIN_IMAGES_PER_CLASS`), enforced server-side in
  `data/validation.py` per R3. Still needs product sign-off if 10 is wrong.

### B2. Target dataset scale is unknown
- **Resolution:** Built assuming small, user-uploaded datasets (tens–low
  hundreds of images/class). Feature extraction and CNN sizing (small
  from-scratch conv-net, see D14) both assume this scale. Revisit if larger
  datasets are expected — would need batching/GPU support not currently built.

### B3. Deployment target/environment not specified
- **Resolution:** Built and verified for local/dev (`localhost:8000`).
  Webcam capture works over localhost per browser rules. HTTPS/hosting
  concerns are unaddressed — revisit before any non-local deploy.

### B4. Multi-user support is out of scope but not explicitly ruled out
- **Resolution:** Built single-user/local-only. SQLite + local filesystem
  (D8) confirmed sufficient at this scale. No auth, no per-user isolation.

### B5. Feature extractor for classical models not specified by spec
- **Resolution:** Implemented D6's documented fallback — HOG + color
  histogram (`trainers/features.py`) — as the actual default, since the
  pretrained-embedding path is unreachable in this environment (no network
  access to weight hosts). See decisions.md D13.

### B6. Real-time performance target for webcam predictions not defined
- **Resolution:** Implemented capture-then-predict (button-triggered single
  frame), per the documented default. Continuous-stream mode not built.

### B7. Confusion matrix visualization requirements unclear
- **Resolution:** Implemented both — a color-coded heatmap table and raw
  counts in the same table (diagonal green, off-diagonal red intensity by
  count) in the Results tab of the UI.

### B8. Model comparison beyond accuracy/confusion matrix not specified
- **Resolution:** Shipped accuracy + confusion matrix only (R6 minimum).
  Precision/recall/F1 per class not built — flagged as a future enhancement,
  not a blocker for v1.
