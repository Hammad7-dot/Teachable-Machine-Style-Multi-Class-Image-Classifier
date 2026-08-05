"""models/ -- model artifact I/O (save/load) only, per R1. No training logic here."""
from pathlib import Path
import joblib
import tensorflow as tf

MODELS_ROOT = Path(__file__).resolve().parent.parent / "storage" / "models"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)


def run_dir(run_id):
    d = MODELS_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_sklearn_artifact(path):
    return joblib.load(path)


def load_keras_artifact(path):
    return tf.keras.models.load_model(path)


class ModelCache:
    def __init__(self):
        self._cache = {}

    def get_sklearn(self, path):
        if path not in self._cache:
            self._cache[path] = load_sklearn_artifact(path)
        return self._cache[path]

    def get_keras(self, path):
        if path not in self._cache:
            self._cache[path] = load_keras_artifact(path)
        return self._cache[path]

    def invalidate(self):
        self._cache.clear()


cache = ModelCache()
