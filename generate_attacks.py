import os
import json
import cv2
import numpy as np
from typing import Tuple, Dict, Any

SOURCES = {
    "stop": "source_images/stop.png",
    "ped": "source_images/ped.jpg",
    "street": "source_images/street.jpg"
}

OUTPUT_DIR = "progressive_attacks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = {}
results = {}

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

def compute_canny_edges(gray: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    return cv2.Canny(gray, low, high)

def compute_laplacian_edges(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compute_roberts_edges(gray: np.ndarray) -> np.ndarray:
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gray_float = gray.astype(np.float32)
    grad_x = cv2.filter2D(gray_float, -1, roberts_x)
    grad_y = cv2.filter2D(gray_float, -1, roberts_y)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compute_edges(gray: np.ndarray, detector: str) -> np.ndarray:
    if detector == "sobel":
        return compute_sobel_edges(gray)
    elif detector == "canny":
        return compute_canny_edges(gray)
    elif detector == "laplacian":
        return compute_laplacian_edges(gray)
    elif detector == "roberts":
        return compute_roberts_edges(gray)

def random_pixel_perturbation(gray: np.ndarray, num_pixels: int, delta: int) -> np.ndarray:
    pert = gray.astype(np.int16).copy()
    h, w = pert.shape
    ys = np.random.randint(0, h, size=num_pixels)
    xs = np.random.randint(0, w, size=num_pixels)
    for y, x in zip(ys, xs):
        change = delta if np.random.rand() < 0.5 else -delta
        pert[y, x] = np.clip(pert[y, x] + change, 0, 255)
    return pert.astype(np.uint8)

def targeted_edge_attack(gray: np.ndarray, attack_strength: float = 0.3) -> np.ndarray:
    edges = compute_sobel_edges(gray)
    threshold = np.percentile(edges, (1 - attack_strength) * 100)
    strong_edge_mask = edges > threshold

    attacked = gray.copy().astype(np.float32)
    blurred = cv2.GaussianBlur(attacked, (31, 31), 8.0)
    attacked[strong_edge_mask] = blurred[strong_edge_mask]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(strong_edge_mask.astype(np.uint8), kernel, iterations=2)
    super_blurred = cv2.GaussianBlur(attacked, (21, 21), 5.0)
    attacked[dilated_mask > 0] = super_blurred[dilated_mask > 0]

    return attacked.astype(np.uint8)

def gradient_direction_attack(gray: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    threshold = np.percentile(grad_mag, 90)
    strong_edges = grad_mag > threshold

    attacked = gray.copy().astype(np.float32)

    for y, x in np.argwhere(strong_edges):
        if 1 <= y < gray.shape[0]-1 and 1 <= x < gray.shape[1]-1:
            gx, gy = sobel_x[y, x], sobel_y[y, x]
            magnitude = min(100, grad_mag[y, x] * 0.8)

            if abs(gx) > abs(gy):
                attacked[y, x] = np.clip(attacked[y, x] - np.sign(gx) * magnitude, 0, 255)
            else:
                attacked[y, x] = np.clip(attacked[y, x] - np.sign(gy) * magnitude, 0, 255)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gray.shape[0] and 0 <= nx < gray.shape[1]:
                        attacked[ny, nx] = np.clip(attacked[ny, nx] - np.sign(gx + gy) * magnitude * 0.3, 0, 255)

    return attacked.astype(np.uint8)

def contour_disruption_attack(gray: np.ndarray) -> np.ndarray:
    edges = compute_sobel_edges(gray)
    _, binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    attacked = gray.copy()

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=20)

        blurred = cv2.GaussianBlur(attacked, (41, 41), 10.0)
        attacked[mask > 0] = blurred[mask > 0]

        contour_points = contour.reshape(-1, 2)
        num_holes = min(5, len(contour_points) // 10)

        for _ in range(num_holes):
            if len(contour_points) > 0:
                idx = np.random.randint(0, len(contour_points))
                cx, cy = contour_points[idx]

                hole_size = 15
                y1 = max(0, cy - hole_size)
                y2 = min(gray.shape[0], cy + hole_size)
                x1 = max(0, cx - hole_size)
                x2 = min(gray.shape[1], cx + hole_size)

                background_color = np.mean(gray)
                attacked[y1:y2, x1:x2] = background_color

    return attacked

def compute_confidences(edge_map: np.ndarray) -> Tuple[float,float]:
    _, b = cv2.threshold(edge_map, 1, 255, cv2.THRESH_BINARY)
    res = edge_map.shape[0]
    edge_frac = np.count_nonzero(b) / (res * res)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    contour_conf = max_area / (res * res)
    return edge_frac, contour_conf

def compute_attack_effectiveness(clean_edges: np.ndarray, attacked_edges: np.ndarray) -> dict:
    clean_density = np.mean(clean_edges > 20)
    attacked_density = np.mean(attacked_edges > 20)
    density_reduction = (clean_density - attacked_density) / clean_density if clean_density > 0 else 0

    _, clean_binary = cv2.threshold(clean_edges, 20, 255, cv2.THRESH_BINARY)
    _, attacked_binary = cv2.threshold(attacked_edges, 20, 255, cv2.THRESH_BINARY)

    clean_contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    attacked_contours, _ = cv2.findContours(attacked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_max_area = max([cv2.contourArea(c) for c in clean_contours]) if clean_contours else 0
    attacked_max_area = max([cv2.contourArea(c) for c in attacked_contours]) if attacked_contours else 0

    contour_reduction = (clean_max_area - attacked_max_area) / clean_max_area if clean_max_area > 0 else 0

    return {
        "edge_density_reduction": float(density_reduction),
        "contour_area_reduction": float(contour_reduction),
        "attack_success": bool(density_reduction > 0.15 or contour_reduction > 0.2 or abs(density_reduction) > 0.3)
    }

def save_png(img: np.ndarray, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)

def main():
    global metadata, results

    np.random.seed(42)

    print("Checking source images...")
    for src_name, src_path in SOURCES.items():
        if not os.path.exists(src_path):
            print(f"ERROR: Source image not found: {src_path}")
            print("Please update the SOURCES dictionary with correct image paths.")
            return
        else:
            print(f"✓ Found: {src_name} -> {src_path}")

    print("\nStarting progressive attack generation...")

    attack_sequence = [
        {"name": "gentle_pixels", "detectors": ["sobel", "canny"], "type": "pixel",
         "params": {"patch_size": 20, "flip_pct": 0.01, "delta": 25}},

        {"name": "moderate_pixels", "detectors": ["sobel", "canny"], "type": "pixel",
         "params": {"patch_size": 40, "flip_pct": 0.03, "delta": 75}},

        {"name": "aggressive_pixels", "detectors": ["sobel", "canny"], "type": "pixel",
         "params": {"patch_size": 60, "flip_pct": 0.05, "delta": 255}},

        {"name": "smart_edge", "detectors": ["sobel", "canny"], "type": "edge_blur"},

        {"name": "gradient", "detectors": ["sobel", "canny"], "type": "gradient_reverse"},

        {"name": "contour", "detectors": ["sobel", "canny", "laplacian", "roberts"], "type": "contour_disrupt"},
    ]

    for src_name, src_path in SOURCES.items():
        print(f"Processing source: {src_name}")

        try:
            gray_clean = load_and_resize_grayscale(src_path, 256)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading {src_path}: {e}")
            continue

        for attack_info in attack_sequence:
            for detector in attack_info["detectors"]:

                # Compute clean edges
                if detector == "sobel":
                    edge_clean = compute_sobel_edges(gray_clean)
                elif detector == "canny":
                    edge_clean = compute_canny_edges(gray_clean)
                elif detector == "laplacian":
                    edge_clean = compute_laplacian_edges(gray_clean)
                elif detector == "roberts":
                    edge_clean = compute_roberts_edges(gray_clean)

                # Apply attack
                if attack_info["type"] == "pixel":
                    params = attack_info["params"]
                    flip_count = int(params["patch_size"] * params["patch_size"] * params["flip_pct"])
                    gray_pert = random_pixel_perturbation(gray_clean, flip_count, params["delta"])

                    # Generate base name (original script style)
                    base = f"{src_name}-{detector}-256x256-{params['patch_size']}px-{int(params['flip_pct']*100)}pct-Δ{params['delta']}"

                elif attack_info["type"] == "edge_blur":
                    gray_pert = targeted_edge_attack(gray_clean, attack_strength=0.3)
                    base = f"{src_name}_{detector}_edge_blur"

                elif attack_info["type"] == "gradient_reverse":
                    gray_pert = gradient_direction_attack(gray_clean)
                    base = f"{src_name}_{detector}_gradient_reverse"

                elif attack_info["type"] == "contour_disrupt":
                    gray_pert = contour_disruption_attack(gray_clean)
                    base = f"{src_name}_{detector}_contour_disrupt"

                # Compute perturbed edges
                edge_pert = compute_edges(gray_pert, detector)

                # Compute confidences (original script style)
                ec_frac, ec_cont = compute_confidences(edge_clean)
                ep_frac, ep_cont = compute_confidences(edge_pert)

                # Compute effectiveness (targeted attack style)
                effectiveness = compute_attack_effectiveness(edge_clean, edge_pert)

                # Save images
                save_png(gray_clean, f"{base}-gray-clean.png")
                save_png(gray_pert, f"{base}-gray-pert.png")
                save_png(edge_clean, f"{base}-edge-clean.png")
                save_png(edge_pert, f"{base}-edge-pert.png")

                # Store metadata (original script style)
                metadata[base] = {
                    "src": src_name,
                    "detector": detector,
                    "resolution": 256,
                    "attack_type": attack_info["type"],
                    "level": attack_info["name"],
                    "edge_frac_clean": ec_frac,
                    "contour_conf_clean": ec_cont,
                    "edge_frac_pert": ep_frac,
                    "contour_conf_pert": ep_cont,
                    **effectiveness
                }

                # Store results (targeted attack style)
                results[base] = {
                    "source": src_name,
                    "attack_method": attack_info["type"],
                    "detector": detector,
                    **effectiveness
                }

                # Print progress (ORIGINAL SCRIPT STYLE)
                print(f"Generated: {base}")

    # Print final results (TARGETED ATTACK STYLE)
    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Print summary (TARGETED ATTACK STYLE)
    successful_attacks = sum(1 for r in results.values() if r["attack_success"])
    print(f"Successful attacks: {successful_attacks}/{len(results)}")

    # Write metadata.json (original script style)
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Write attack_results.json (targeted attack style)
    with open(os.path.join(OUTPUT_DIR, "attack_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print("All done—progressive_attacks/ filled and metadata.json written.")

if __name__ == "__main__":
    main()
