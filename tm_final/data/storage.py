"""Filesystem persistence for dataset images. Per D8."""
from pathlib import Path
import uuid

DATA_ROOT = Path(__file__).resolve().parent.parent / "storage" / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def class_dir(class_name):
    d = DATA_ROOT / class_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_image(class_name, pil_image, orig_filename):
    fname = f"{uuid.uuid4().hex}.jpg"
    path = class_dir(class_name) / fname
    pil_image.save(path, format="JPEG", quality=90)
    return fname, str(path)


def delete_class_dir(class_name):
    import shutil
    d = DATA_ROOT / class_name
    if d.exists():
        shutil.rmtree(d)
