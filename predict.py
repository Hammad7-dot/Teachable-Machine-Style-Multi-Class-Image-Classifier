"""
inference/ -- prediction for uploaded image and webcam frame.
Per D11: both sources use the same code path. Per D12/R7: one call returns
predictions from all three models. Per R8: unavailable models say why.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data import db
from models.registry import cache, run_dir as models_run_dir
from trainers.features import extract_features
from trainers.cnn import IMG_SIZE as CNN_IMG_SIZE


def _resolve_artifact_path(run_id, stored_path):
    """artifact_path in the DB is meant to be just a filename (e.g.
    'cnn.keras'), resolved against this machine's storage/models/<run_id>/
    at read time -- not baked in as an absolute path at training time (that
    broke the moment a git-committed DB was cloned onto a different
    machine/OS/directory; see decisions.md D17). Older rows may still hold
    a full path from before that fix -- including Windows-style paths like
    'D:\\...\\cnn.keras', which pathlib.Path(...).name does NOT strip
    correctly on Linux (backslash isn't a separator on POSIX, so ".name"
    returns the whole string unchanged) -- so extract the filename by
    splitting on both '/' and '\\' explicitly rather than trusting Path.
    """
    stored_str = str(stored_path)
    candidate = Path(stored_str)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    filename = stored_str.replace("\\", "/").rsplit("/", 1)[-1]
    return str(models_run_dir(run_id) / filename)


def _predict_sklearn(artifact_path, labels, pil_img):
    bundle = cache.get_sklearn(artifact_path)
    clf = bundle["clf"]
    feats = extract_features(pil_img).reshape(1, -1)
    if "scaler" in bundle:
        feats = bundle["scaler"].transform(feats)
    probs = clf.predict_proba(feats)[0]
    order = np.argsort(-probs)
    return {"predicted_class": labels[order[0]],
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))}}


def _predict_cnn(artifact_path, labels, pil_img):
    model = cache.get_keras(artifact_path)
    im = pil_img.convert("RGB").resize(CNN_IMG_SIZE)
    arr = (np.asarray(im, dtype=np.float32) / 255.0)[None, ...]
    probs = model.predict(arr, verbose=0)[0]
    order = np.argsort(-probs)
    return {"predicted_class": labels[order[0]],
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))}}


def predict_all(pil_img, run_id=None):
    run = db.get_run(run_id) if run_id else db.latest_done_run()
    if not run:
        return {"error": "No completed training run available. Train models first."}
    if not run_id:
        run = db.get_run(run["id"])

    results = {}
    for m in run["models"]:
        model_type = m["model_type"]
        if m["status"] != "done" or not m["artifact_path"]:
            results[model_type] = {"status": "unavailable", "reason": m.get("error") or f"model status: {m['status']}"}
            continue
        labels = json.loads(m["labels"])
        try:
            resolved_path = _resolve_artifact_path(run["id"], m["artifact_path"])
            if model_type == "cnn":
                pred = _predict_cnn(resolved_path, labels, pil_img)
            else:
                pred = _predict_sklearn(resolved_path, labels, pil_img)
            pred["status"] = "ok"
            pred["accuracy_at_train"] = m["accuracy"]
            results[model_type] = pred
        except Exception as e:
            results[model_type] = {"status": "unavailable", "reason": str(e)}

    return {"run_id": run["id"], "predictions": results}
