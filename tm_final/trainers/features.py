"""
Shared feature extraction for classical models per D6/B5. Uses HOG + color
histogram (no external pretrained weight download required -- this
deployment has no internet access to weight hosts, so the "pretrained CNN
embedding" branch of D6 is unreachable). See decisions.md D13.
"""
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

FEATURE_IMG_SIZE = (96, 96)


def extract_features(pil_img):
    img = pil_img.convert("RGB").resize(FEATURE_IMG_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = rgb2gray(arr)
    hog_feats = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    color_hist = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 1))
        color_hist.append(hist.astype(np.float32) / (arr.shape[0] * arr.shape[1]))
    color_hist = np.concatenate(color_hist)
    return np.concatenate([hog_feats, color_hist]).astype(np.float32)


def extract_features_batch(pil_images):
    return np.stack([extract_features(im) for im in pil_images])
