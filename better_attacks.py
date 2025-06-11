import os
import json
import cv2
import numpy as np
from typing import Tuple

SOURCES = {
    "stop": "source_images/stop.png",
    "ped": "source_images/ped.jpg",
    "street": "source_images/street.jpg"
}

OUTPUT_DIR = "targeted_attacks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_resize_grayscale(path: str, size: int = 256) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

def compute_sobel_edges(gray: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x**2 + sobel_y**2)
    norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def targeted_edge_attack(gray: np.ndarray, attack_strength: float = 0.1) -> np.ndarray:
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

def gradient_direction_attack(gray: np.ndarray, attack_strength: float = 0.1) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    threshold = np.percentile(grad_mag, (1 - attack_strength) * 100)
    strong_edges = grad_mag > threshold

    attacked = gray.copy().astype(np.float32)

    for y, x in np.argwhere(strong_edges):
        if 1 <= y < gray.shape[0]-1 and 1 <= x < gray.shape[1]-1:
            gx = sobel_x[y, x]
            gy = sobel_y[y, x]

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

def main():
    results = {}

    attack_methods = {
        "edge_blur": targeted_edge_attack,
        "gradient_reverse": gradient_direction_attack,
        "contour_disrupt": contour_disruption_attack
    }

    for src_name, src_path in SOURCES.items():
        if not os.path.exists(src_path):
            print(f"Skipping {src_name} - file not found: {src_path}")
            continue

        print(f"Processing: {src_name}")

        try:
            gray_clean = load_and_resize_grayscale(src_path)
            edges_clean = compute_sobel_edges(gray_clean)

            for attack_name, attack_func in attack_methods.items():
                if attack_name == "edge_blur":
                    gray_attacked = attack_func(gray_clean, attack_strength=0.3)
                else:
                    gray_attacked = attack_func(gray_clean)

                edges_attacked = compute_sobel_edges(gray_attacked)
                effectiveness = compute_attack_effectiveness(edges_clean, edges_attacked)
                base_name = f"{src_name}_{attack_name}"
                cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_gray_clean.png", gray_clean)
                cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_gray_attacked.png", gray_attacked)
                cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_edges_clean.png", edges_clean)
                cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_edges_attacked.png", edges_attacked)
                results[base_name] = {
                    "source": src_name,
                    "attack_method": attack_name,
                    **effectiveness
                }

                success_str = "SUCCESS" if effectiveness["attack_success"] else "FAILED"
                print(f"  {attack_name}: {success_str} (edge reduction: {effectiveness['edge_density_reduction']:.1%})")

        except Exception as e:
            print(f"Error processing {src_name}: {e}")
            continue

    with open(f"{OUTPUT_DIR}/attack_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")

    successful_attacks = sum(1 for r in results.values() if r["attack_success"])
    print(f"Successful attacks: {successful_attacks}/{len(results)}")

if __name__ == "__main__":
    main()
