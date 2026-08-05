"""
Generic bulk-uploader: point it at a local folder laid out as

    dataset_root/
        class_a/*.jpg
        class_b/*.png
        ...

(this is how most Kaggle image datasets, and tensorflow_datasets exports,
are already organized) and it uploads every image into this app's running
backend via the same /api/classes/{name}/images endpoint the UI uses.

Usage:
    python3 scripts/upload_folder.py /path/to/dataset_root [--api http://127.0.0.1:8000] [--batch-size 20]

Each immediate subfolder of dataset_root becomes a class, named after the
folder. Only .jpg/.jpeg/.png/.webp files are uploaded; anything else in a
class folder is skipped. Files the backend rejects (bad format/size/
resolution, per R4) are reported individually -- the rest of the batch
still uploads.
"""
import argparse
import sys
from pathlib import Path

import requests

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIME_BY_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Folder containing one subfolder per class")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of the running backend")
    parser.add_argument("--batch-size", type=int, default=20, help="Images per upload request")
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Optional cap on images per class (uploads the first N found)")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"No subfolders found under {root} -- expected one subfolder per class.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(class_dirs)} class folders: {[d.name for d in class_dirs]}")

    total_accepted, total_rejected = 0, 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        if args.limit_per_class:
            image_paths = image_paths[:args.limit_per_class]

        if not image_paths:
            print(f"  {class_name}: no images found, skipping")
            continue

        print(f"  {class_name}: uploading {len(image_paths)} images...")
        for batch in chunked(image_paths, args.batch_size):
            files = [
                ("files", (p.name, p.read_bytes(), MIME_BY_EXT[p.suffix.lower()]))
                for p in batch
            ]
            resp = requests.post(f"{args.api}/api/classes/{class_name}/images", files=files, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            total_accepted += len(result["accepted"])
            total_rejected += len(result["rejected"])
            for r in result["rejected"]:
                print(f"    rejected {r['filename']}: {r['reason']}")

    print(f"\nDone. {total_accepted} images accepted, {total_rejected} rejected.")

    status = requests.get(f"{args.api}/api/dataset/status", timeout=30).json()
    print("Dataset status:", status)
    if status["ready_to_train"]:
        print("Ready to train! POST /api/train or click Train in the UI.")
    else:
        print(f"Not ready yet: {status['reason']}")


if __name__ == "__main__":
    main()
