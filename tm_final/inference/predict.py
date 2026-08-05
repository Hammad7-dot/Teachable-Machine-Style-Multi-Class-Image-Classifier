"""
inference/ -- prediction for uploaded image and webcam frame.
Per D11: both sources use the same code path. Per D12/R7: one call returns
predictions from all three models. Per R8: unavailable models say why.
"""
import json
import numpy as np
from PIL import Image

from data import db
from models.registry import cache
from trainers.features import extract_features
from trainers.cnn import IMG_SIZE as CNN_IMG_SIZE


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
            if model_type == "cnn":
                pred = _predict_cnn(m["artifact_path"], labels, pil_img)
            else:
                pred = _predict_sklearn(m["artifact_path"], labels, pil_img)
            pred["status"] = "ok"
            pred["accuracy_at_train"] = m["accuracy"]
            results[model_type] = pred
        except Exception as e:
            results[model_type] = {"status": "unavailable", "reason": str(e)}

    return {"run_id": run["id"], "predictions": results}
