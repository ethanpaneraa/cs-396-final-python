#!/usr/bin/env python3
"""
generate_attacks_full.py

Extended attack generator for edge-detection adversarial demos.
Produces clean vs. perturbed grayscale and edge maps for multiple parameters,
and computes simple confidence metrics for each variant.
"""

import os
import json
import cv2
import numpy as np
from typing import Tuple

# -----------------------------
# 1) HARD-CODED CONFIGURATION
# -----------------------------
# Map logical names to source image paths
# UPDATE THESE PATHS TO YOUR ACTUAL IMAGE LOCATIONS
SOURCES = {
    "stop":   "source_images/stop.png",
    "ped":    "source_images/ped.jpg",
    "street": "source_images/street.jpg"
}

# Alternative: use any images you have available
# SOURCES = {
#     "image1": "path/to/your/first/image.jpg",
#     "image2": "path/to/your/second/image.jpg",
#     "image3": "path/to/your/third/image.jpg"
# }

# Edge detectors to apply
DETECTORS = ["sobel", "canny"]

# Output resolutions
RESOLUTIONS = [128, 256, 512]

# Patch sizes (side-length in pixels at target resolution)
PATCH_SIZES = [20, 40, 60]
# Percent of patch pixels to flip
FLIP_PCTS = [0.01, 0.03, 0.05]
# Perturbation magnitudes
DELTAS = [25, 75, 255]

# Canny thresholds
CANNY_SETTINGS = [
    (50, 150),
    (100, 200),
    (150, 300)
]

# Contour-area threshold for detection confidence
MIN_AREA = 300

# Output directory for generated images and metadata
OUTPUT_DIR = "attacks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metadata dictionary
metadata = {}

# -----------------------------
# 2) UTILITY FUNCTIONS
# -----------------------------

def load_and_resize_grayscale(path: str, size: int) -> np.ndarray:
    """Load image and convert to grayscale, with better error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image (possibly corrupted or unsupported format): {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def compute_sobel_edges(gray: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x**2 + sobel_y**2)
    norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def compute_canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, low, high)


def random_pixel_perturbation(gray: np.ndarray, num_pixels: int, delta: int) -> np.ndarray:
    pert = gray.astype(np.int16).copy()
    h, w = pert.shape
    ys = np.random.randint(0, h, size=num_pixels)
    xs = np.random.randint(0, w, size=num_pixels)
    for y, x in zip(ys, xs):
        change = delta if np.random.rand() < 0.5 else -delta
        pert[y, x] = np.clip(pert[y, x] + change, 0, 255)
    return pert.astype(np.uint8)


def compute_confidences(edge_map: np.ndarray) -> Tuple[float,float]:
    # binary threshold
    _, b = cv2.threshold(edge_map, 1, 255, cv2.THRESH_BINARY)
    res = edge_map.shape[0]
    edge_frac = np.count_nonzero(b) / (res * res)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    contour_conf = max_area / (res * res)
    return edge_frac, contour_conf


def save_png(img: np.ndarray, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)

# -----------------------------
# 3) MAIN GENERATION LOOP
# -----------------------------

def main():
    np.random.seed(42)

    # Check if source images exist before starting
    print("Checking source images...")
    for src_name, src_path in SOURCES.items():
        if not os.path.exists(src_path):
            print(f"ERROR: Source image not found: {src_path}")
            print("Please update the SOURCES dictionary with correct image paths.")
            return
        else:
            print(f"✓ Found: {src_name} -> {src_path}")

    print("\nStarting attack generation...")

    for src_name, src_path in SOURCES.items():
        print(f"Processing source: {src_name}")
        for detector in DETECTORS:
            for res in RESOLUTIONS:
                try:
                    gray_clean = load_and_resize_grayscale(src_path, res)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Error loading {src_path}: {e}")
                    continue

                for patch_size in PATCH_SIZES:
                    for pct in FLIP_PCTS:
                        flip_count = int(patch_size * patch_size * pct)
                        for delta in DELTAS:
                            base = f"{src_name}-{detector}-{res}x{res}-{patch_size}px-{int(pct*100)}pct-Δ{delta}"

                            # Clean edges
                            if detector == "sobel":
                                edge_clean = compute_sobel_edges(gray_clean)
                            else:
                                # use middle Canny thresholds
                                low, high = CANNY_SETTINGS[1]
                                edge_clean = compute_canny_edges(gray_clean, low, high)

                            # Perturb
                            gray_pert = random_pixel_perturbation(gray_clean, flip_count, delta)

                            # Perturbed edges
                            if detector == "sobel":
                                edge_pert = compute_sobel_edges(gray_pert)
                            else:
                                low, high = CANNY_SETTINGS[1]
                                edge_pert = compute_canny_edges(gray_pert, low, high)

                            # Compute confidences
                            ec_frac, ec_cont = compute_confidences(edge_clean)
                            ep_frac, ep_cont = compute_confidences(edge_pert)

                            # Save images
                            save_png(gray_clean, f"{base}-gray-clean.png")
                            save_png(gray_pert,  f"{base}-gray-pert.png")
                            save_png(edge_clean, f"{base}-edge-clean.png")
                            save_png(edge_pert,  f"{base}-edge-pert.png")

                            # Store metadata
                            metadata[base] = {
                                "src": src_name,
                                "detector": detector,
                                "resolution": res,
                                "patch_size": patch_size,
                                "flip_pct": pct,
                                "delta": delta,
                                "edge_frac_clean": ec_frac,
                                "contour_conf_clean": ec_cont,
                                "edge_frac_pert": ep_frac,
                                "contour_conf_pert": ep_cont
                            }
                            print(f"Generated: {base}")

    # Write metadata.json
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("All done—attacks/ filled and metadata.json written.")

if __name__ == "__main__":
    main()
