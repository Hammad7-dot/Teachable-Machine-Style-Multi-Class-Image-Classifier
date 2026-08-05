"""Validation rules -- enforced server-side, per R3/R4."""
from PIL import Image
import io

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MIN_RESOLUTION = (32, 32)
MAX_RESOLUTION = (4096, 4096)
DEFAULT_MIN_IMAGES_PER_CLASS = 10
MIN_CLASSES_TO_TRAIN = 2


class ValidationError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


def validate_image_bytes(raw_bytes, filename):
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(f"{filename}: file too large ({len(raw_bytes)/1e6:.1f}MB > {MAX_FILE_SIZE_BYTES/1e6:.0f}MB limit)")
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Exception:
        raise ValidationError(f"{filename}: not a readable image file")
    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(f"{filename}: unsupported format '{fmt}' (allowed: JPEG, PNG, WebP)")
    w, h = img.size
    if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
        raise ValidationError(f"{filename}: resolution {w}x{h} below minimum {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}")
    if w > MAX_RESOLUTION[0] or h > MAX_RESOLUTION[1]:
        raise ValidationError(f"{filename}: resolution {w}x{h} exceeds maximum {MAX_RESOLUTION[0]}x{MAX_RESOLUTION[1]}")
    return img.convert("RGB")


def validate_dataset_ready(class_counts, min_per_class=DEFAULT_MIN_IMAGES_PER_CLASS):
    if len(class_counts) < MIN_CLASSES_TO_TRAIN:
        raise ValidationError(f"At least {MIN_CLASSES_TO_TRAIN} classes are required to start training (currently have {len(class_counts)}).")
    short = {c: n for c, n in class_counts.items() if n < min_per_class}
    if short:
        detail = ", ".join(f"{c} ({n}/{min_per_class})" for c, n in short.items())
        raise ValidationError(f"Some classes don't meet the minimum of {min_per_class} images: {detail}")
