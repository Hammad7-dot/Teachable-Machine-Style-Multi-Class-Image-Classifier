"""
Downloads the Rock Paper Scissors dataset (via tensorflow_datasets) and
bulk-uploads it into this app's running backend through the same
/api/classes/{name}/images endpoint the UI uses.

Requires internet access to download.tensorflow.org (blocked in some
sandboxed environments -- if this fails with a 403/connection error there,
download the dataset manually instead and use upload_folder.py, which takes
any local folder-of-images-per-class layout).

Usage:
    pip install tensorflow-datasets
    python3 scripts/download_rock_paper_scissors.py [--limit-per-class 60] [--api http://127.0.0.1:8000]
"""
import argparse
import io
import sys
from pathlib import Path

import requests

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("This script needs tensorflow-datasets: pip install tensorflow-datasets", file=sys.stderr)
    sys.exit(1)

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of the running backend")
    parser.add_argument("--limit-per-class", type=int, default=60,
                         help="Cap images per class so training stays fast (dataset has ~840/class)")
    parser.add_argument("--batch-size", type=int, default=20, help="Images per upload request")
    args = parser.parse_args()

    class_names = {0: "rock", 1: "paper", 2: "scissors"}

    print("Downloading rock_paper_scissors via tensorflow_datasets (first run may take a while)...")
    ds, info = tfds.load("rock_paper_scissors", split="train", with_info=True)

    per_class_counts = {name: 0 for name in class_names.values()}
    buffers = {name: [] for name in class_names.values()}

    for example in ds:
        label = int(example["label"].numpy())
        class_name = class_names[label]
        if per_class_counts[class_name] >= args.limit_per_class:
            continue
        img = Image.fromarray(example["image"].numpy())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buffers[class_name].append((f"{class_name}_{per_class_counts[class_name]}.jpg", buf.getvalue()))
        per_class_counts[class_name] += 1
        if all(c >= args.limit_per_class for c in per_class_counts.values()):
            break

    print("Collected:", per_class_counts)

    for class_name, items in buffers.items():
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            files = [("files", (fname, data, "image/jpeg")) for fname, data in batch]
            resp = requests.post(f"{args.api}/api/classes/{class_name}/images", files=files, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            print(f"  {class_name}: uploaded batch {i // args.batch_size + 1} "
                  f"({len(result['accepted'])} accepted, {len(result['rejected'])} rejected)")
            for r in result["rejected"]:
                print(f"    rejected {r['filename']}: {r['reason']}")

    status = requests.get(f"{args.api}/api/dataset/status", timeout=30).json()
    print("\nDataset status:", status)
    if status["ready_to_train"]:
        print("Ready to train! POST /api/train or click Train in the UI.")


if __name__ == "__main__":
    main()
