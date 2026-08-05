"""
FastAPI backend entrypoint. Wires data/, trainers/, models/, inference/
together per R1 module boundaries. No ML logic lives here beyond
orchestration/threading of the trainer modules.
"""
import asyncio, json, threading, time, uuid, io
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from data import db, storage
from data.validation import (validate_image_bytes, ValidationError, validate_dataset_ready,
                              DEFAULT_MIN_IMAGES_PER_CLASS, MIN_CLASSES_TO_TRAIN)
from models.registry import run_dir, cache
from trainers.logistic_regression import LogisticRegressionTrainer
from trainers.random_forest import RandomForestTrainer
from trainers.cnn import CNNTrainer
from inference.predict import predict_all

app = FastAPI(title="Teachable Machine-Style Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TRAINER_CLASSES = {
    "logistic_regression": LogisticRegressionTrainer,
    "random_forest": RandomForestTrainer,
    "cnn": CNNTrainer,
}

_ws_clients = {}
_run_event_log = {}  # run_id -> list of already-broadcast events, replayed to late-joining clients
_main_loop = None


@app.get("/api/classes")
def get_classes():
    counts = db.class_counts()
    names = db.list_classes()
    return {"classes": [{"name": n, "image_count": counts.get(n, 0)} for n in names]}


@app.get("/api/dashboard")
def dashboard():
    """Aggregate summary for the Dashboard view in both frontends: dataset
    size, run history, and the latest completed run's per-model accuracy --
    one call instead of each UI re-deriving this from /api/classes,
    /api/dataset/status, and /api/runs separately."""
    counts = db.class_counts_full()
    names = db.list_classes()
    runs = db.list_runs()
    latest = db.latest_done_run()

    latest_summary = None
    if latest:
        full = db.get_run(latest["id"])
        models = []
        for m in full["models"]:
            models.append({
                "model_type": m["model_type"],
                "status": m["status"],
                "accuracy": m["accuracy"],
            })
        latest_summary = {"run_id": full["id"], "started_at": full["started_at"], "models": models}

    return {
        "total_classes": len(names),
        "total_images": sum(counts.values()),
        "class_counts": counts,
        "total_runs": len(runs),
        "completed_runs": len([r for r in runs if r["status"] == "done"]),
        "latest_run": latest_summary,
    }


@app.post("/api/classes")
def create_class(name: str = Form(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Class name cannot be empty")
    db.add_class(name)
    return {"ok": True}


@app.delete("/api/classes/{name}")
def remove_class(name: str):
    db.delete_class(name)
    storage.delete_class_dir(name)
    return {"ok": True}


@app.post("/api/classes/{name}/images")
async def upload_images(name: str, files: List[UploadFile] = File(...)):
    db.add_class(name)
    accepted, rejected = [], []
    for f in files:
        raw = await f.read()
        try:
            img = validate_image_bytes(raw, f.filename)
        except ValidationError as e:
            rejected.append({"filename": f.filename, "reason": e.message})
            continue
        fname, path = storage.save_image(name, img, f.filename)
        # Only the filename is trustworthy long-term (see D17) -- don't
        # persist the absolute path storage.save_image() happens to return.
        db.add_image(name, fname, fname)
        accepted.append(fname)
    return {"accepted": accepted, "rejected": rejected}


@app.get("/api/dataset/status")
def dataset_status():
    counts = db.class_counts_full()
    ready, reason = True, None
    try:
        validate_dataset_ready(counts)
    except ValidationError as e:
        ready, reason = False, e.message
    return {"class_counts": counts, "min_images_per_class": DEFAULT_MIN_IMAGES_PER_CLASS,
            "min_classes": MIN_CLASSES_TO_TRAIN, "ready_to_train": ready, "reason": reason}


def _run_training(run_id: str):
    class_names = db.list_classes()
    labels = sorted(class_names)
    label_to_idx = {c: i for i, c in enumerate(labels)}

    images = db.list_images()
    pil_images, y = [], []
    for rec in images:
        try:
            # rec["path"] is legacy: older rows may hold an absolute path
            # baked in at upload time on a different machine/OS (e.g. a
            # Windows path from a git-committed storage/app.db) -- that
            # breaks the moment the repo is cloned elsewhere. Always
            # reconstruct the path from class_name + filename against this
            # machine's actual storage/data/ instead of trusting the
            # stored path. See decisions.md D17.
            img_path = storage.class_dir(rec["class_name"]) / rec["filename"]
            im = Image.open(img_path).convert("RGB")
            pil_images.append(im)
            y.append(label_to_idx[rec["class_name"]])
        except Exception:
            continue

    rd = run_dir(run_id)

    def make_progress_cb():
        def cb(event):
            _broadcast(run_id, {"type": "progress", **event})
        return cb

    for model_type, TrainerCls in TRAINER_CLASSES.items():
        db.update_run_model(run_id, model_type, status="training")
        _broadcast(run_id, {"type": "model_status", "model": model_type, "status": "training"})
        try:
            trainer = TrainerCls(run_id, str(rd))
            result = trainer.train(pil_images, y, labels, make_progress_cb())
            db.update_run_model(run_id, model_type, status="done",
                                 # Store only the filename, never a full path -- an absolute
                                 # path baked in at training time breaks the moment the DB
                                 # (or a git-committed storage/ folder) is moved to another
                                 # machine, directory, or OS. See decisions.md D17.
                                 artifact_path=Path(result.artifact_path).name,
                                 accuracy=result.accuracy, confusion_matrix=json.dumps(result.confusion_matrix),
                                 labels=json.dumps(result.labels), finished_at=time.time())
            _broadcast(run_id, {"type": "model_status", "model": model_type, "status": "done", "accuracy": result.accuracy})
        except Exception as e:
            db.update_run_model(run_id, model_type, status="failed", error=str(e), finished_at=time.time())
            _broadcast(run_id, {"type": "model_status", "model": model_type, "status": "failed", "error": str(e)})

    db.finish_run(run_id)
    cache.invalidate()
    _broadcast(run_id, {"type": "run_complete", "run_id": run_id})


def _broadcast(run_id, message):
    # Persist every event so a client that connects after training has
    # already started (a real race: POST /api/train kicks off the
    # background thread immediately, and the UI only opens its WebSocket
    # after that response comes back -- any events fired in that gap were
    # previously dropped silently) can be caught up on connect.
    log = _run_event_log.setdefault(run_id, [])
    log.append(message)
    del log[:-500]  # bound memory: keep only the most recent 500 events/run

    clients = _ws_clients.get(run_id, [])
    if not clients or _main_loop is None:
        return
    dead = []
    for ws in clients:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(message), _main_loop)
        except Exception:
            dead.append(ws)
    for d in dead:
        clients.remove(d)


@app.post("/api/train")
def start_training():
    counts = db.class_counts_full()
    try:
        validate_dataset_ready(counts)
    except ValidationError as e:
        raise HTTPException(400, e.message)

    run_id = uuid.uuid4().hex[:12]
    db.create_run(run_id, sorted(db.list_classes()))
    _ws_clients[run_id] = []
    _run_event_log[run_id] = []

    t = threading.Thread(target=_run_training, args=(run_id,), daemon=True)
    t.start()
    return {"run_id": run_id}


@app.get("/api/runs")
def get_runs():
    return {"runs": db.list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    for m in run["models"]:
        if m.get("confusion_matrix"):
            m["confusion_matrix"] = json.loads(m["confusion_matrix"])
        if m.get("labels"):
            m["labels"] = json.loads(m["labels"])
    run["class_names"] = json.loads(run["class_names"])
    return run


@app.websocket("/ws/train/{run_id}")
async def ws_train(websocket: WebSocket, run_id: str):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    await websocket.accept()

    # Catch this client up on any events broadcast before it connected
    # (see _broadcast) -- fixes the race where fast-finishing early stages
    # (e.g. Logistic Regression) could complete before the frontend's
    # WebSocket connection was established.
    for event in _run_event_log.get(run_id, []):
        try:
            await websocket.send_json(event)
        except Exception:
            pass

    _ws_clients.setdefault(run_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients.get(run_id, []):
            _ws_clients[run_id].remove(websocket)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), run_id: str = Form(None)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")
    return predict_all(img, run_id)


UI_DIR = Path(__file__).resolve().parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(UI_DIR / "index.html"))
