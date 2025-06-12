import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple

# --------------------------------------------------
# Constants & Global Records
# --------------------------------------------------

IMAGE_SOURCES: Dict[str, str] = {
    "stop_sign":    "source_images/stop.png",
    "pedestrian":   "source_images/ped.jpg",
    "street_scene": "source_images/street.jpg"
}

OUTPUT_DIRECTORY: str = "progressive_attacks"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

METADATA_RECORDS: Dict[str, Any] = {}
ATTACK_RESULTS: Dict[str, Any] = {}

# --------------------------------------------------
# I/O & Preprocessing
# --------------------------------------------------

def load_and_resize_and_convert_to_grayscale(
    file_path: str,
    target_size: int
) -> np.ndarray:
    """
    /**
     * Load an image from disk, convert to grayscale, and resize to square.
     *
     * @param {string} filePath – Path to source image.
     * @param {number} targetSize – Desired width & height in pixels.
     * @returns {numpy.ndarray} 8-bit grayscale image of shape (targetSize, targetSize).
     * @throws {FileNotFoundError} If the file does not exist.
     * @throws {ValueError} If the image cannot be decoded.
     */
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found: {file_path}")

    bgr = cv2.imread(file_path)
    if bgr is None:
        raise ValueError(f"Failed to load or decode image: {file_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

# --------------------------------------------------
# Edge Detectors
# --------------------------------------------------

def compute_sobel_edge_map(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Compute Sobel gradient magnitude map.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @returns {numpy.ndarray} Normalized 8-bit gradient magnitude.
     */
    """
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def compute_canny_edge_map(
    gray_image: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> np.ndarray:
    """
    /**
     * Compute Canny edge map.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @param {number} lowThreshold – Lower hysteresis threshold.
     * @param {number} highThreshold – Upper hysteresis threshold.
     * @returns {numpy.ndarray} Edge map (binary 8-bit).
     */
    """
    return cv2.Canny(gray_image, low_threshold, high_threshold)

def compute_laplacian_edge_map(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Compute Laplacian-of-Gaussian edges.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @returns {numpy.ndarray} Normalized 8-bit Laplacian edges.
     */
    """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    normalized = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def compute_roberts_edge_map(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Compute Roberts Cross edge map.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @returns {numpy.ndarray} Normalized 8-bit Roberts edges.
     */
    """
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    f = gray_image.astype(np.float32)
    gx = cv2.filter2D(f, -1, kx)
    gy = cv2.filter2D(f, -1, ky)
    magnitude = np.sqrt(gx**2 + gy**2)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def compute_edge_map(
    gray_image: np.ndarray,
    method: str
) -> np.ndarray:
    """
    /**
     * Dispatch to the selected edge detector.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @param {string} method – One of 'sobel','canny','laplacian','roberts'.
     * @returns {numpy.ndarray} Edge map.
     * @throws {ValueError} If method is unsupported.
     */
    """
    if method == "sobel":
        return compute_sobel_edge_map(gray_image)
    if method == "canny":
        return compute_canny_edge_map(gray_image)
    if method == "laplacian":
        return compute_laplacian_edge_map(gray_image)
    if method == "roberts":
        return compute_roberts_edge_map(gray_image)
    raise ValueError(f"Unsupported edge detector: {method}")

# --------------------------------------------------
# Progressive Attack Primitives
# --------------------------------------------------

def apply_random_pixel_perturbation(
    gray_image: np.ndarray,
    num_pixels: int,
    delta: int
) -> np.ndarray:
    """
    /**
     * Randomly perturb given number of pixels by ±delta.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @param {number} numPixels – Count of random pixels to alter.
     * @param {number} delta – Intensity to add or subtract.
     * @returns {numpy.ndarray} Perturbed image.
     */
    """
    perturbed = gray_image.astype(np.int16).copy()
    h, w = perturbed.shape
    ys = np.random.randint(0, h, size=num_pixels)
    xs = np.random.randint(0, w, size=num_pixels)
    for y, x in zip(ys, xs):
        change = delta if np.random.rand() < 0.5 else -delta
        perturbed[y, x] = np.clip(perturbed[y, x] + change, 0, 255)
    return perturbed.astype(np.uint8)

def apply_targeted_edge_blur_attack(
    gray_image: np.ndarray,
    strength_ratio: float = 0.3
) -> np.ndarray:
    """
    /**
     * Blur only the strongest edges to degrade detection.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @param {number} strengthRatio – Fraction of top-percentile edges to blur.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    edges = compute_sobel_edge_map(gray_image)
    thresh = np.percentile(edges, (1 - strength_ratio) * 100)
    mask = edges > thresh

    img_f = gray_image.astype(np.float32)
    blurred1 = cv2.GaussianBlur(img_f, (31, 31), 8.0)
    img_f[mask] = blurred1[mask]

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask.astype(np.uint8), se, iterations=2)
    blurred2 = cv2.GaussianBlur(img_f, (21, 21), 5.0)
    img_f[dilated > 0] = blurred2[dilated > 0]

    return img_f.astype(np.uint8)

def apply_gradient_direction_attack(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Reverse gradient direction on strongest edges.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    thresh = np.percentile(mag, 90)
    strong = mag > thresh

    attacked = gray_image.astype(np.float32)
    for y, x in np.argwhere(strong):
        if 1 <= y < gray_image.shape[0]-1 and 1 <= x < gray_image.shape[1]-1:
            dx, dy = gx[y, x], gy[y, x]
            m = min(100, mag[y, x]*0.8)
            sign = np.sign(dx) if abs(dx)>abs(dy) else np.sign(dy)
            attacked[y, x] = np.clip(attacked[y, x] - sign*m, 0, 255)
            for oy in (-1,0,1):
                for ox in (-1,0,1):
                    ny, nx = y+oy, x+ox
                    if 0<=ny<gray_image.shape[0] and 0<=nx<gray_image.shape[1]:
                        attacked[ny, nx] = np.clip(attacked[ny, nx] - sign*m*0.3, 0, 255)
    return attacked.astype(np.uint8)

def apply_contour_disruption_attack(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Disrupt up to three largest contours by blurring & adding holes.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    edges = compute_sobel_edge_map(gray_image)
    _, bw = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray_image

    top3 = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    attacked = gray_image.astype(np.float32)

    for cnt in top3:
        if cv2.contourArea(cnt) < 100:
            continue
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=20)
        blur1 = cv2.GaussianBlur(attacked, (41, 41), 10.0)
        attacked[mask>0] = blur1[mask>0]

        pts = cnt.reshape(-1,2)
        holes = min(5, len(pts)//10)
        for _ in range(holes):
            idx = np.random.randint(len(pts))
            cx, cy = pts[idx]
            s = 15
            y1, y2 = max(0, cy-s), min(gray_image.shape[0], cy+s)
            x1, x2 = max(0, cx-s), min(gray_image.shape[1], cx+s)
            attacked[y1:y2, x1:x2] = np.mean(gray_image)
    return attacked.astype(np.uint8)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_edge_confidence_metrics(edge_map: np.ndarray) -> Tuple[float, float]:
    """
    /**
     * Compute basic confidence: fraction of edge pixels & max-contour density.
     *
     * @param {numpy.ndarray} edgeMap – Binary edge map (0 or 255).
     * @returns {[number, number]} [edgeFraction, maxContourFraction].
     */
    """
    size = edge_map.shape[0]
    _, binary = cv2.threshold(edge_map, 1, 255, cv2.THRESH_BINARY)
    edge_fraction = np.count_nonzero(binary) / (size * size)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    return edge_fraction, max_area / (size * size)

def calculate_attack_effectiveness_metrics(
    clean_edges: np.ndarray,
    attacked_edges: np.ndarray
) -> Dict[str, Any]:
    """
    /**
     * Compare clean vs. attacked edges to compute reduction metrics and success.
     *
     * @param {numpy.ndarray} cleanEdges – Before-attack edge map.
     * @param {numpy.ndarray} attackedEdges – After-attack edge map.
     * @returns {Object} Metrics: edge_density_reduction, contour_area_reduction, attack_success.
     */
    """
    clean_density = np.mean(clean_edges > 20)
    attacked_density = np.mean(attacked_edges > 20)
    density_reduction = ((clean_density - attacked_density) / clean_density) if clean_density>0 else 0.0

    _, bin_clean = cv2.threshold(clean_edges, 20, 255, cv2.THRESH_BINARY)
    _, bin_attacked = cv2.threshold(attacked_edges, 20, 255, cv2.THRESH_BINARY)
    cont_clean, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_attacked, _ = cv2.findContours(bin_attacked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_clean = max((cv2.contourArea(c) for c in cont_clean), default=0.0)
    max_attacked = max((cv2.contourArea(c) for c in cont_attacked), default=0.0)
    contour_reduction = ((max_clean - max_attacked) / max_clean) if max_clean>0 else 0.0

    success = bool(density_reduction>0.15 or contour_reduction>0.2 or abs(density_reduction)>0.3)
    return {
        "edge_density_reduction": float(density_reduction),
        "contour_area_reduction": float(contour_reduction),
        "attack_success": success
    }

def save_image_png(image: np.ndarray, filename: str) -> None:
    """
    /**
     * Save an image array as a PNG file in the output directory.
     *
     * @param {numpy.ndarray} image – Image to write.
     * @param {string} filename – Base filename (no path).
     * @returns {void}
     */
    """
    path = os.path.join(OUTPUT_DIRECTORY, filename)
    cv2.imwrite(path, image)

# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main() -> None:
    """
    /**
     * Run progressive attack sequence on each source image,
     * collect metadata, results, and write JSON outputs.
     */
    """
    np.random.seed(42)
    print("Verifying source images...")
    for name, path in IMAGE_SOURCES.items():
        if not os.path.exists(path):
            print(f"ERROR: Missing image '{name}' at {path}")
            return
        print(f"✓ Found '{name}'")

    print("\nStarting progressive attacks...")
    attack_sequence = [
        {"level": "gentle_pixels",     "detectors": ["sobel","canny"],
         "type": "pixel",  "params": {"patch_size":20, "flip_pct":0.01, "delta":25}},
        {"level": "moderate_pixels",   "detectors": ["sobel","canny"],
         "type": "pixel",  "params": {"patch_size":40, "flip_pct":0.03, "delta":75}},
        {"level": "aggressive_pixels", "detectors": ["sobel","canny"],
         "type": "pixel",  "params": {"patch_size":60, "flip_pct":0.05, "delta":255}},
        {"level": "smart_edge",        "detectors": ["sobel","canny"],
         "type": "edge_blur"},
        {"level": "gradient",          "detectors": ["sobel","canny"],
         "type": "gradient_reverse"},
        {"level": "contour",           "detectors": ["sobel","canny","laplacian","roberts"],
         "type": "contour_disrupt"},
    ]

    for src_name, src_path in IMAGE_SOURCES.items():
        print(f"\nProcessing '{src_name}'")
        try:
            gray_clean = load_and_resize_and_convert_to_grayscale(src_path, 256)
        except Exception as e:
            print(f"Failed to load '{src_name}': {e}")
            continue

        for attack in attack_sequence:
            for detector in attack["detectors"]:
                # compute clean edges
                clean_edges = compute_edge_map(gray_clean, detector)

                # apply progressive attack
                if attack["type"] == "pixel":
                    p = attack["params"]
                    flips = int(p["patch_size"]**2 * p["flip_pct"])
                    gray_pert = apply_random_pixel_perturbation(gray_clean, flips, p["delta"])
                elif attack["type"] == "edge_blur":
                    gray_pert = apply_targeted_edge_blur_attack(gray_clean, strength_ratio=0.3)
                elif attack["type"] == "gradient_reverse":
                    gray_pert = apply_gradient_direction_attack(gray_clean)
                else:
                    gray_pert = apply_contour_disruption_attack(gray_clean)

                pert_edges = compute_edge_map(gray_pert, detector)

                # compute confidences & effectiveness
                ec_frac, ec_cont = calculate_edge_confidence_metrics(clean_edges)
                ep_frac, ep_cont = calculate_edge_confidence_metrics(pert_edges)
                metrics = calculate_attack_effectiveness_metrics(clean_edges, pert_edges)

                base = f"{src_name}-{detector}-{attack['level']}"
                # save outputs
                save_image_png(gray_clean,      f"{base}-clean.png")
                save_image_png(gray_pert,       f"{base}-pert.png")
                save_image_png(clean_edges,     f"{base}-edges-clean.png")
                save_image_png(pert_edges,      f"{base}-edges-pert.png")

                # record metadata & results
                METADATA_RECORDS[base] = {
                    "source": src_name,
                    "detector": detector,
                    "attack_level": attack["level"],
                    "edge_fraction_clean":  ec_frac,
                    "contour_confidence_clean": ec_cont,
                    "edge_fraction_perturbed": ep_frac,
                    "contour_confidence_perturbed": ep_cont,
                    **metrics
                }
                ATTACK_RESULTS[base] = {
                    "source": src_name,
                    "detector": detector,
                    "attack_type": attack["type"],
                    **metrics
                }

                print(f"  → Generated '{base}'")

    # write JSON outputs
    with open(os.path.join(OUTPUT_DIRECTORY, "metadata.json"), 'w') as f:
        json.dump(METADATA_RECORDS, f, indent=2)
    with open(os.path.join(OUTPUT_DIRECTORY, "attack_results.json"), 'w') as f:
        json.dump(ATTACK_RESULTS, f, indent=2)

    total = len(ATTACK_RESULTS)
    successes = sum(1 for r in ATTACK_RESULTS.values() if r["attack_success"])
    print(f"\nAll done — results saved to '{OUTPUT_DIRECTORY}/'")
    print(f"Successful attacks: {successes}/{total}")

if __name__ == "__main__":
    main()
