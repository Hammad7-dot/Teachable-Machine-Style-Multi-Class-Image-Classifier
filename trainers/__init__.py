"""trainers/ -- one module per model type, common train()/progress()/evaluate() interface (R1)."""
from .base import BaseTrainer, TrainResult
__all__ = ["BaseTrainer", "TrainResult"]
