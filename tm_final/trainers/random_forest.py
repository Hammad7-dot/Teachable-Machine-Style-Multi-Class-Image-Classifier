from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from .base import BaseTrainer, TrainResult
from .features import extract_features_batch


class RandomForestTrainer(BaseTrainer):
    model_type = "random_forest"

    def train(self, pil_images, y_labels, labels, progress_cb):
        self.emit(progress_cb, "start", "Extracting features (HOG + color histogram)...", progress=0.0)
        X = extract_features_batch(pil_images)
        y = np.array(y_labels)

        self.emit(progress_cb, "split", "Splitting train/validation sets...", progress=0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)

        n_estimators = 200
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

        self.emit(progress_cb, "fit", f"Fitting Random Forest ({n_estimators} trees)...", progress=0.4)
        clf.fit(X_train, y_train)
        self.emit(progress_cb, "fit_done", "Fit complete.", progress=0.8)

        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(labels))))

        artifact_path = str(Path(self.run_dir) / "random_forest.joblib")
        joblib.dump({"clf": clf, "labels": labels}, artifact_path)

        self.emit(progress_cb, "end", f"Done. Held-out accuracy={acc:.3f}", progress=1.0, metric=acc)
        return TrainResult(model_type=self.model_type, accuracy=float(acc),
                            confusion_matrix=cm.tolist(), labels=labels, artifact_path=artifact_path)
