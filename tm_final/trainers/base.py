"""Common trainer interface (R1, R5)."""
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class TrainResult:
    model_type: str
    accuracy: float
    confusion_matrix: List[List[int]]
    labels: List[str]
    artifact_path: str


class BaseTrainer:
    model_type = "base"

    def __init__(self, run_id, run_dir):
        self.run_id = run_id
        self.run_dir = run_dir

    def train(self, X, y, labels, progress_cb):
        raise NotImplementedError

    def emit(self, progress_cb, stage, message, progress=None, metric=None):
        event = {"model": self.model_type, "stage": stage, "message": message}
        if progress is not None:
            event["progress"] = progress
        if metric is not None:
            event["metric"] = metric
        progress_cb(event)
