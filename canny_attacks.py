import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple

# --------------------------------------------------
# Configuration Constants
# --------------------------------------------------

IMAGE_SOURCES: Dict[str, str] = {
    "stop_sign":   "source_images/stop.png",
    "pedestrian":  "source_images/ped.jpg",
    "street_scene":"source_images/street.jpg"
}

RESULTS_DIRECTORY: str = "canny_targeted_attacks"
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
# Canny Edge Detection with Different Configurations
# --------------------------------------------------

def compute_canny_edge_map(
    gray_image: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_kernel_size: int = 5
) -> np.ndarray:
    """
    /**
     * Compute Canny edges with configurable parameters.
     *
     * @param {numpy.ndarray} grayImage – 8-bit grayscale input.
     * @param {number} lowThreshold – Lower hysteresis threshold.
     * @param {number} highThreshold – Upper hysteresis threshold.
     * @param {number} blurKernelSize – Gaussian blur kernel size.
     * @returns {numpy.ndarray} Binary edge map.
     */
    """
    if blur_kernel_size > 1:
        blurred = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)
    else:
        blurred = gray_image

    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges

# --------------------------------------------------
# Canny-Specific Attack Methods
# --------------------------------------------------

def apply_hysteresis_threshold_attack(
    gray_image: np.ndarray,
    noise_level: float = 0.15
) -> np.ndarray:
    """
    /**
     * Attack targeting Canny's hysteresis thresholding.
     * Adds noise specifically in the range between low and high thresholds.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} noiseLevel – Noise intensity factor.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    # Compute gradients to identify threshold regions
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Identify pixels in the hysteresis range (between typical low and high thresholds)
    low_thresh_val = np.percentile(gradient_magnitude, 70)
    high_thresh_val = np.percentile(gradient_magnitude, 90)

    hysteresis_mask = (gradient_magnitude > low_thresh_val) & (gradient_magnitude < high_thresh_val)

    attacked = gray_image.astype(np.float32).copy()

    # Add structured noise to hysteresis regions
    noise = np.random.normal(0, noise_level * 128, gray_image.shape)
    attacked[hysteresis_mask] += noise[hysteresis_mask]

    # Also add some salt-and-pepper noise to break connectivity
    salt_pepper_mask = np.random.random(gray_image.shape) < 0.02
    attacked[salt_pepper_mask & hysteresis_mask] = np.random.choice([0, 255],
                                                                    size=np.sum(salt_pepper_mask & hysteresis_mask))

    return np.clip(attacked, 0, 255).astype(np.uint8)

def apply_gradient_smoothing_attack(
    gray_image: np.ndarray,
    smooth_factor: float = 0.3
) -> np.ndarray:
    """
    /**
     * Attack targeting Canny's gradient computation.
     * Selectively smooths gradients to reduce edge strength.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} smoothFactor – Smoothing intensity.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    # Compute initial edges to identify targets
    initial_edges = compute_canny_edge_map(gray_image)

    # Dilate edges to get smoothing regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_regions = cv2.dilate(initial_edges, kernel, iterations=2)

    attacked = gray_image.astype(np.float32).copy()

    # Apply directional smoothing along edge normals
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Normalize gradients
    magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
    norm_x = grad_x / magnitude
    norm_y = grad_y / magnitude

    # Create anisotropic blur that preserves some structure while reducing detectability
    smoothed = cv2.bilateralFilter(gray_image, 9, 75, 75)

    # Blend based on edge regions and smooth factor
    blend_mask = (edge_regions > 0).astype(np.float32) * smooth_factor
    attacked = attacked * (1 - blend_mask) + smoothed * blend_mask

    return attacked.astype(np.uint8)

def apply_non_maximum_suppression_attack(
    gray_image: np.ndarray,
    disruption_strength: float = 0.4
) -> np.ndarray:
    """
    /**
     * Attack targeting Canny's non-maximum suppression step.
     * Creates competing gradients to confuse edge thinning.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} disruptionStrength – Attack intensity.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    attacked = gray_image.astype(np.float32).copy()

    # Compute gradient direction
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_direction = np.arctan2(grad_y, grad_x)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Identify strong edge pixels
    strong_edges = gradient_magnitude > np.percentile(gradient_magnitude, 80)

    # Create perpendicular gradients to confuse NMS
    h, w = gray_image.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if strong_edges[y, x]:
                # Get perpendicular direction
                perp_angle = gradient_direction[y, x] + np.pi/2

                # Add competing gradients in perpendicular direction
                offset_x = int(np.round(np.cos(perp_angle)))
                offset_y = int(np.round(np.sin(perp_angle)))

                # Create intensity variations to generate false edges
                for i in range(-2, 3):
                    ny, nx = y + i*offset_y, x + i*offset_x
                    if 0 <= ny < h and 0 <= nx < w:
                        perturbation = disruption_strength * 50 * np.sin(i * np.pi / 2)
                        attacked[ny, nx] = np.clip(attacked[ny, nx] + perturbation, 0, 255)

    # Apply slight Gaussian noise to make attack less obvious
    noise = np.random.normal(0, 10, attacked.shape)
    attacked = np.clip(attacked + noise, 0, 255)

    return attacked.astype(np.uint8)

def apply_connectivity_breaking_attack(
    gray_image: np.ndarray,
    break_probability: float = 0.3
) -> np.ndarray:
    """
    /**
     * Attack targeting Canny's edge linking through hysteresis.
     * Breaks edge connectivity by introducing gaps.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} breakProbability – Probability of breaking edge connections.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    # First detect edges to know where to break connectivity
    edges = compute_canny_edge_map(gray_image)

    # Find edge pixels
    edge_pixels = np.argwhere(edges > 0)

    attacked = gray_image.copy()

    # For each edge pixel, potentially create a gap
    for y, x in edge_pixels:
        if np.random.random() < break_probability:
            # Create small gaps by modifying pixel intensities
            gap_size = np.random.randint(2, 5)

            # Determine gap direction based on edge orientation
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)[y, x]
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)[y, x]

            # Gap perpendicular to gradient
            if abs(grad_x) > abs(grad_y):
                # Vertical gap
                for dy in range(-gap_size//2, gap_size//2 + 1):
                    if 0 <= y + dy < gray_image.shape[0]:
                        # Smooth transition to background
                        attacked[y + dy, x] = np.mean([
                            gray_image[max(0, y + dy - 2), x],
                            gray_image[min(gray_image.shape[0] - 1, y + dy + 2), x]
                        ])
            else:
                # Horizontal gap
                for dx in range(-gap_size//2, gap_size//2 + 1):
                    if 0 <= x + dx < gray_image.shape[1]:
                        attacked[y, x + dx] = np.mean([
                            gray_image[y, max(0, x + dx - 2)],
                            gray_image[y, min(gray_image.shape[1] - 1, x + dx + 2)]
                        ])

    return attacked

def apply_multi_scale_attack(
    gray_image: np.ndarray,
    scale_factor: float = 0.3
) -> np.ndarray:
    """
    /**
     * Attack using multi-scale perturbations that affect Canny at different thresholds.
     *
     * @param {numpy.ndarray} grayImage – Grayscale input.
     * @param {number} scaleFactor – Multi-scale blending factor.
     * @returns {numpy.ndarray} Attacked image.
     */
    """
    attacked = gray_image.astype(np.float32).copy()

    # Create perturbations at multiple scales
    scales = [3, 7, 15]
    perturbations = []

    for scale in scales:
        # Create scale-specific noise pattern
        noise = np.random.randn(gray_image.shape[0] // scale, gray_image.shape[1] // scale)

        # Upsample to original size
        noise_upsampled = cv2.resize(noise, (gray_image.shape[1], gray_image.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)

        # Normalize and scale
        noise_upsampled = noise_upsampled * (20 / scale)
        perturbations.append(noise_upsampled)

    # Combine multi-scale perturbations
    combined_perturbation = sum(perturbations) * scale_factor

    # Apply selectively to regions likely to be edges
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Target mid-range gradients
    mid_gradient_mask = (gradient_magnitude > np.percentile(gradient_magnitude, 40)) & \
                       (gradient_magnitude < np.percentile(gradient_magnitude, 90))

    attacked[mid_gradient_mask] += combined_perturbation[mid_gradient_mask]

    return np.clip(attacked, 0, 255).astype(np.uint8)

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

    # For Canny, we get binary edges, so adjust threshold
    _, bin_pre = cv2.threshold(reference_edges, 1, 255, cv2.THRESH_BINARY)
    _, bin_post = cv2.threshold(attacked_edges, 1, 255, cv2.THRESH_BINARY)

    cont_pre, _ = cv2.findContours(bin_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_post, _ = cv2.findContours(bin_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_pre = max((cv2.contourArea(c) for c in cont_pre), default=0.0)
    max_post = max((cv2.contourArea(c) for c in cont_post), default=0.0)
    contour_reduction = ((max_pre - max_post) / max_pre) if max_pre > 0 else 0.0

    # For Canny, also measure connectivity
    num_contours_pre = len(cont_pre)
    num_contours_post = len(cont_post)
    fragmentation_increase = (num_contours_post - num_contours_pre) / max(num_contours_pre, 1)

    success = bool(
        density_reduction > 0.15 or
        contour_reduction > 0.2 or
        fragmentation_increase > 0.5
    )

    return {
        "edge_density_reduction": float(density_reduction),
        "contour_area_reduction": float(contour_reduction),
        "fragmentation_increase": float(fragmentation_increase),
        "attack_success": success
    }

# --------------------------------------------------
# Main Execution
# --------------------------------------------------

def main() -> None:
    """
    /**
     * Batch-process images with Canny-specific adversarial attacks.
     */
    """
    results: Dict[str, Any] = {}
    attack_functions: Dict[str, Any] = {
        "hysteresis_threshold": apply_hysteresis_threshold_attack,
        "gradient_smoothing": apply_gradient_smoothing_attack,
        "non_max_suppression": apply_non_maximum_suppression_attack,
        "connectivity_breaking": apply_connectivity_breaking_attack,
        "multi_scale": apply_multi_scale_attack,
    }

    # Test with different Canny configurations
    canny_configs = [
        {"low": 50, "high": 150, "blur": 5, "name": "standard"},
        {"low": 100, "high": 200, "blur": 5, "name": "high_threshold"},
        {"low": 30, "high": 100, "blur": 3, "name": "sensitive"},
    ]

    for source_name, source_path in IMAGE_SOURCES.items():
        if not os.path.exists(source_path):
            print(f"⚠️ Skipping '{source_name}': not found at {source_path}")
            continue

        print(f"\nProcessing '{source_name}'")
        try:
            gray_clean = load_and_resize_and_convert_to_grayscale(source_path)

            for canny_config in canny_configs:
                config_name = canny_config["name"]
                print(f"  Canny config: {config_name}")

                # Compute clean edges with this configuration
                edges_clean = compute_canny_edge_map(
                    gray_clean,
                    low_threshold=canny_config["low"],
                    high_threshold=canny_config["high"],
                    blur_kernel_size=canny_config["blur"]
                )

                for method_name, method_func in attack_functions.items():
                    # Apply attack
                    if method_name == "hysteresis_threshold":
                        attacked = method_func(gray_clean, noise_level=0.15)
                    elif method_name == "gradient_smoothing":
                        attacked = method_func(gray_clean, smooth_factor=0.3)
                    elif method_name == "non_max_suppression":
                        attacked = method_func(gray_clean, disruption_strength=0.4)
                    elif method_name == "connectivity_breaking":
                        attacked = method_func(gray_clean, break_probability=0.3)
                    else:  # multi_scale
                        attacked = method_func(gray_clean, scale_factor=0.3)

                    # Compute edges on attacked image
                    edges_attacked = compute_canny_edge_map(
                        attacked,
                        low_threshold=canny_config["low"],
                        high_threshold=canny_config["high"],
                        blur_kernel_size=canny_config["blur"]
                    )

                    # Calculate metrics
                    metrics = calculate_attack_effectiveness_metrics(edges_clean, edges_attacked)

                    # Create unique key
                    result_key = f"{source_name}_{config_name}_{method_name}"
                    results[result_key] = {
                        "source": source_name,
                        "canny_config": config_name,
                        "attack_method": method_name,
                        **metrics
                    }

                    # Save images
                    base_name = f"{source_name}_{config_name}_{method_name}"
                    cv2.imwrite(f"{RESULTS_DIRECTORY}/{base_name}_clean.png", gray_clean)
                    cv2.imwrite(f"{RESULTS_DIRECTORY}/{base_name}_attacked.png", attacked)
                    cv2.imwrite(f"{RESULTS_DIRECTORY}/{base_name}_edges_clean.png", edges_clean)
                    cv2.imwrite(f"{RESULTS_DIRECTORY}/{base_name}_edges_attacked.png", edges_attacked)

                    status = "SUCCESS" if metrics["attack_success"] else "FAILED"
                    print(f"    {method_name}: {status} " +
                          f"(edge reduction: {metrics['edge_density_reduction']:.1%}, " +
                          f"fragmentation: {metrics['fragmentation_increase']:.1%})")

        except Exception as exc:
            print(f"Error processing '{source_name}': {exc}")

    # Write JSON results
    with open(f"{RESULTS_DIRECTORY}/attack_results.json", "w") as json_file:
        json.dump(results, json_file, indent=2)

    # Write metadata about configurations
    metadata = {
        "attack_methods": {
            "hysteresis_threshold": "Targets the dual-threshold hysteresis by adding noise in the threshold gap",
            "gradient_smoothing": "Selectively smooths gradients to reduce edge strength below detection threshold",
            "non_max_suppression": "Creates competing gradients to confuse the edge thinning process",
            "connectivity_breaking": "Introduces gaps in edges to break hysteresis-based edge linking",
            "multi_scale": "Uses multi-scale perturbations to affect edge detection at different sensitivities"
        },
        "canny_configurations": canny_configs,
        "metrics_explanation": {
            "edge_density_reduction": "Reduction in total detected edge pixels",
            "contour_area_reduction": "Reduction in the largest connected edge region",
            "fragmentation_increase": "Increase in number of disconnected edge segments",
            "attack_success": "True if any metric exceeds threshold"
        }
    }

    with open(f"{RESULTS_DIRECTORY}/metadata.json", "w") as json_file:
        json.dump(metadata, json_file, indent=2)

    total = len(results)
    successes = sum(1 for r in results.values() if r["attack_success"])
    print(f"\nResults saved to '{RESULTS_DIRECTORY}/'")
    print(f"Successful attacks: {successes}/{total}")
    print("\nAttack Summary:")

    # Summary by attack method
    for method in attack_functions.keys():
        method_results = [r for k, r in results.items() if method in k]
        method_successes = sum(1 for r in method_results if r["attack_success"])
        print(f"  {method}: {method_successes}/{len(method_results)} successful")

if __name__ == "__main__":
    main()
