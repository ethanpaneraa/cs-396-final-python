import os
import json
import cv2
import numpy as np
from typing import Dict, Any

# --------------------------------------------------
# Configuration Constants
# --------------------------------------------------

IMAGE_SOURCES: Dict[str, str] = {
    "stop_sign":   "source_images/stop.png",
    "pedestrian":  "source_images/ped.jpg",
    "street_scene":"source_images/street.jpg"
}

RESULTS_DIRECTORY: str = "targeted_attacks"
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

# --------------------------------------------------
# Image I/O & Preprocessing
# --------------------------------------------------

def load_and_resize_and_convert_to_grayscale(
    image_path: str,
    target_size: int = 256
) -> np.ndarray:
    """
    /**
     * Load an image, convert to grayscale, and resize to a square.
     *
     * @param {string} imagePath – Path to the source image file.
     * @param {number} targetSize – Width and height (pixels) for output.
     * @returns {numpy.ndarray} 8-bit grayscale image of shape (targetSize, targetSize).
     * @throws {FileNotFoundError} If the image cannot be loaded.
     */
    """
    color_bgr = cv2.imread(image_path)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")
    gray_image = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(
        gray_image,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )
    return resized_gray

# --------------------------------------------------
# Edge Detection
# --------------------------------------------------

def compute_sobel_edge_map(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Compute edge magnitude using a 3×3 Sobel operator.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @returns {numpy.ndarray} 8-bit normalized gradient magnitude map.
     */
    """
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    normalized = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
    )
    return normalized.astype(np.uint8)

# --------------------------------------------------
# Adversarial Edge Attacks
# --------------------------------------------------

def apply_targeted_edge_blur_attack(
    gray_image: np.ndarray,
    attack_strength_ratio: float = 0.1
) -> np.ndarray:
    """
    /**
     * Blur only the strongest edges to degrade detection.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} attackStrengthRatio – Fraction of top-percentile edges to blur.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    edge_map = compute_sobel_edge_map(gray_image)
    threshold_value = np.percentile(edge_map, (1 - attack_strength_ratio) * 100)
    strong_edge_mask = edge_map > threshold_value

    float_image = gray_image.astype(np.float32)
    blurred_once = cv2.GaussianBlur(float_image, (31, 31), 8.0)
    float_image[strong_edge_mask] = blurred_once[strong_edge_mask]

    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(
        strong_edge_mask.astype(np.uint8), struct_elem, iterations=2
    )

    blurred_twice = cv2.GaussianBlur(float_image, (21, 21), 5.0)
    float_image[dilated_mask > 0] = blurred_twice[dilated_mask > 0]

    return float_image.astype(np.uint8)

def apply_gradient_direction_attack(
    gray_image: np.ndarray,
    attack_strength_ratio: float = 0.1
) -> np.ndarray:
    """
    /**
     * Reverse gradient direction on the strongest edges.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} attackStrengthRatio – Fraction of strongest gradients to flip.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    threshold_value = np.percentile(grad_magnitude, (1 - attack_strength_ratio) * 100)
    strong_edge_indices = np.argwhere(grad_magnitude > threshold_value)

    attacked = gray_image.astype(np.float32)
    for y, x in strong_edge_indices:
        if 1 <= y < gray_image.shape[0] - 1 and 1 <= x < gray_image.shape[1] - 1:
            gx, gy = sobel_x[y, x], sobel_y[y, x]
            magnitude = min(100, grad_magnitude[y, x] * 0.8)
            # Flip strongest direction
            if abs(gx) > abs(gy):
                attacked[y, x] = np.clip(attacked[y, x] - np.sign(gx) * magnitude, 0, 255)
            else:
                attacked[y, x] = np.clip(attacked[y, x] - np.sign(gy) * magnitude, 0, 255)
            # Spread effect to neighbors
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gray_image.shape[0] and 0 <= nx < gray_image.shape[1]:
                        attacked[ny, nx] = np.clip(
                            attacked[ny, nx] - np.sign(gx + gy) * magnitude * 0.3,
                            0, 255
                        )
    return attacked.astype(np.uint8)

def apply_contour_disruption_attack(gray_image: np.ndarray) -> np.ndarray:
    """
    /**
     * Disrupt up to three largest contours by blurring and adding holes.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    edge_map = compute_sobel_edge_map(gray_image)
    _, binary_map = cv2.threshold(edge_map, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return gray_image

    top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    attacked = gray_image.copy().astype(np.float32)

    for contour in top_contours:
        if cv2.contourArea(contour) < 100:
            continue
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=20)
        blurred = cv2.GaussianBlur(attacked, (41, 41), 10.0)
        attacked[mask > 0] = blurred[mask > 0]

        points = contour.reshape(-1, 2)
        holes = min(5, len(points) // 10)
        for _ in range(holes):
            idx = np.random.randint(len(points))
            cx, cy = points[idx]
            s = 15
            y1, y2 = max(0, cy - s), min(gray_image.shape[0], cy + s)
            x1, x2 = max(0, cx - s), min(gray_image.shape[1], cx + s)
            attacked[y1:y2, x1:x2] = np.mean(gray_image)

    return attacked.astype(np.uint8)

# --------------------------------------------------
# Effectiveness Metrics
# --------------------------------------------------

def calculate_attack_effectiveness_metrics(
    reference_edges: np.ndarray,
    attacked_edges: np.ndarray
) -> Dict[str, Any]:
    """
    /**
     * Compare pre- and post-attack edge maps to produce metrics.
     *
     * @param {numpy.ndarray} referenceEdges – Edge map before attack.
     * @param {numpy.ndarray} attackedEdges – Edge map after attack.
     * @returns {Object} Metrics: edge density reduction, contour area reduction, success flag.
     */
    """
    pre_density = np.mean(reference_edges > 20)
    post_density = np.mean(attacked_edges > 20)
    density_reduction = ((pre_density - post_density) / pre_density) if pre_density > 0 else 0.0

    _, bin_pre = cv2.threshold(reference_edges, 20, 255, cv2.THRESH_BINARY)
    _, bin_post = cv2.threshold(attacked_edges, 20, 255, cv2.THRESH_BINARY)
    cont_pre, _ = cv2.findContours(bin_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_post, _ = cv2.findContours(bin_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_pre = max((cv2.contourArea(c) for c in cont_pre), default=0.0)
    max_post = max((cv2.contourArea(c) for c in cont_post), default=0.0)
    contour_reduction = ((max_pre - max_post) / max_pre) if max_pre > 0 else 0.0

    success = bool(
        density_reduction > 0.15 or contour_reduction > 0.2 or abs(density_reduction) > 0.3
    )

    return {
        "edge_density_reduction": float(density_reduction),
        "contour_area_reduction": float(contour_reduction),
        "attack_success": success
    }

# --------------------------------------------------
# Main Execution
# --------------------------------------------------

def main() -> None:
    """
    /**
     * Batch-process images with multiple adversarial edge-attack methods,
     * save outputs and effectiveness metrics to disk.
     */
    """
    results: Dict[str, Any] = {}
    attack_functions: Dict[str, Any] = {
        "edge_blur":         apply_targeted_edge_blur_attack,
        "gradient_reverse":  apply_gradient_direction_attack,
        "contour_disrupt":   apply_contour_disruption_attack
    }

    for source_name, source_path in IMAGE_SOURCES.items():
        if not os.path.exists(source_path):
            print(f"⚠️ Skipping '{source_name}': not found at {source_path}")
            continue

        print(f"Processing '{source_name}'")
        try:
            gray_clean = load_and_resize_and_convert_to_grayscale(source_path)
            edges_clean = compute_sobel_edge_map(gray_clean)

            for method_name, method_func in attack_functions.items():
                strength = 0.3 if method_name == "edge_blur" else 0.1
                attacked = method_func(gray_clean, attack_strength_ratio=strength) \
                           if method_name == "edge_blur" \
                           else method_func(gray_clean)
                edges_attacked = compute_sobel_edge_map(attacked)

                metrics = calculate_attack_effectiveness_metrics(edges_clean, edges_attacked)
                result_key = f"{source_name}_{method_name}"
                results[result_key] = {
                    "source":         source_name,
                    "attack_method":  method_name,
                    **metrics
                }

                # Save images
                cv2.imwrite(f"{RESULTS_DIRECTORY}/{result_key}_clean.png",      gray_clean)
                cv2.imwrite(f"{RESULTS_DIRECTORY}/{result_key}_attacked.png",   attacked)
                cv2.imwrite(f"{RESULTS_DIRECTORY}/{result_key}_edges_clean.png",   edges_clean)
                cv2.imwrite(f"{RESULTS_DIRECTORY}/{result_key}_edges_attacked.png", edges_attacked)

                status = "SUCCESS" if metrics["attack_success"] else "FAILED"
                print(f"  {method_name}: {status} (edge reduction: {metrics['edge_density_reduction']:.1%})")

        except Exception as exc:
            print(f"Error processing '{source_name}': {exc}")

    # Write JSON results
    with open(f"{RESULTS_DIRECTORY}/attack_results.json", "w") as json_file:
        json.dump(results, json_file, indent=2)

    total = len(results)
    successes = sum(1 for r in results.values() if r["attack_success"])
    print(f"\nResults saved to '{RESULTS_DIRECTORY}/'")
    print(f"Successful attacks: {successes}/{total}")

if __name__ == "__main__":
    main()
