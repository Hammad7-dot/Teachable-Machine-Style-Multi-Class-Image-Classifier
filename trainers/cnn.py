"""
CNN trainer using TensorFlow/Keras per D5. Deviation: trains from scratch
(small conv-net) instead of a frozen pretrained backbone, since this
deployment has no network access to pretrained-weight hosts. See D14.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .base import BaseTrainer, TrainResult

IMG_SIZE = (96, 96)


def _images_to_array(pil_images):
    arrs = []
    for im in pil_images:
        im2 = im.convert("RGB").resize(IMG_SIZE)
        arrs.append(np.asarray(im2, dtype=np.float32) / 255.0)
    return np.stack(arrs)


class _ProgressCallback(keras.callbacks.Callback):
    def __init__(self, emit_fn, total_epochs):
        super().__init__()
        self.emit_fn = emit_fn
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        frac_done = (epoch + 1) / self.total_epochs
        progress = 0.1 + 0.8 * frac_done
        val_acc = logs.get("val_accuracy")
        if val_acc is not None:
            msg = f"Epoch {epoch + 1}/{self.total_epochs} - loss={logs.get('loss', 0):.3f} val_acc={val_acc:.3f}"
        else:
            msg = f"Epoch {epoch + 1}/{self.total_epochs} - loss={logs.get('loss', 0):.3f}"
        self.emit_fn("epoch", msg, progress=progress, metric=val_acc)


class CNNTrainer(BaseTrainer):
    model_type = "cnn"

    def train(self, pil_images, y_labels, labels, progress_cb):
        self.emit(progress_cb, "start", "Preparing image tensors...", progress=0.0)
        X = _images_to_array(pil_images)
        y = np.array(y_labels)
        n_classes = len(labels)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)

        self.emit(progress_cb, "build", "Building model...", progress=0.05)
        model = keras.Sequential([
            keras.layers.Input(shape=(*IMG_SIZE, 3)),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        epochs = 15
        cb = _ProgressCallback(
            emit_fn=lambda stage, msg, progress=None, metric=None: self.emit(progress_cb, stage, msg, progress=progress, metric=metric),
            total_epochs=epochs)

        self.emit(progress_cb, "train", f"Training for {epochs} epochs...", progress=0.1)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                  batch_size=16, verbose=0, callbacks=[cb])

        self.emit(progress_cb, "eval", "Evaluating on held-out split...", progress=0.9)
        val_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(val_probs, axis=1)
        acc = float((y_pred == y_val).mean())
        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(labels))))

        artifact_path = str(Path(self.run_dir) / "cnn.keras")
        model.save(artifact_path)

        self.emit(progress_cb, "end", f"Done. Held-out accuracy={acc:.3f}", progress=1.0, metric=acc)
        return TrainResult(model_type=self.model_type, accuracy=acc,
                            confusion_matrix=cm.tolist(), labels=labels, artifact_path=artifact_path)
