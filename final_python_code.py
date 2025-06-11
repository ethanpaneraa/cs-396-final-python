import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    detector: str = "sobel"
    resolution: int = 256

    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    canny_blur_kernel: int = 5

    sobel_kernel_size: int = 3
    sobel_direction: str = "both"
    attack_type: str = "contour_disrupt"
    attack_intensity: str = "moderate"

    lighting: str = "normal"
    add_noise: bool = False
    noise_level: float = 0.1

class EffectiveConfigurableSimulator:
    def __init__(self):
        self.config = SimulationConfig()

        self.DETECTORS = {
            "sobel": "Sobel Filter - Classic gradient-based detection",
            "canny": "Canny Edge Detection - Multi-stage with hysteresis",
            "laplacian": "Laplacian of Gaussian - Second derivative method",
            "roberts": "Roberts Cross - Simple, fast detection"
        }

        self.ATTACK_TYPES = {
            "pixel_perturbation": "Random pixel noise",
            "edge_blur": "Targeted edge smoothing",
            "gradient_reverse": "Gradient direction reversal",
            "contour_disrupt": "Contour boundary disruption",
            "geometric": "Rotation/scaling transforms",
            "occlusion": "Strategic patches/masks"
        }

        self.INTENSITIES = {
            "subtle": {"strength": 0.2, "desc": "Barely noticeable"},
            "moderate": {"strength": 0.4, "desc": "Visible but not obvious"},
            "aggressive": {"strength": 0.6, "desc": "Clearly visible changes"},
            "extreme": {"strength": 0.9, "desc": "Heavily distorted"}
        }

        self.LIGHTING = {
            "normal": {"brightness": 1.0, "contrast": 1.0},
            "low_light": {"brightness": 0.6, "contrast": 0.8},
            "high_contrast": {"brightness": 1.2, "contrast": 1.5},
            "overexposed": {"brightness": 1.8, "contrast": 0.7}
        }

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.config.resolution, self.config.resolution))

        lighting = self.LIGHTING[self.config.lighting]
        gray = gray.astype(np.float32)
        gray = gray * lighting["brightness"]
        gray = np.clip(128 + (gray - 128) * lighting["contrast"], 0, 255)

        if self.config.add_noise:
            noise = np.random.normal(0, self.config.noise_level * 255, gray.shape)
            gray = np.clip(gray + noise, 0, 255)

        return gray.astype(np.uint8)

    def compute_edges(self, gray: np.ndarray) -> np.ndarray:
        if self.config.detector == "sobel":
            return self._sobel_edges(gray)
        elif self.config.detector == "canny":
            return self._canny_edges(gray)
        elif self.config.detector == "laplacian":
            return self._laplacian_edges(gray)
        elif self.config.detector == "roberts":
            return self._roberts_edges(gray)
        else:
            raise ValueError(f"Unknown detector: {self.config.detector}")

    def _sobel_edges(self, gray: np.ndarray) -> np.ndarray:
        ksize = self.config.sobel_kernel_size

        if self.config.sobel_direction == "horizontal":
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            grad = np.abs(sobel)
        elif self.config.sobel_direction == "vertical":
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = np.abs(sobel)
        else:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = np.sqrt(sobel_x**2 + sobel_y**2)

        return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _canny_edges(self, gray: np.ndarray) -> np.ndarray:
        if self.config.canny_blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.config.canny_blur_kernel, self.config.canny_blur_kernel), 0)

        return cv2.Canny(gray, self.config.canny_low_threshold, self.config.canny_high_threshold)

    def _laplacian_edges(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _roberts_edges(self, gray: np.ndarray) -> np.ndarray:
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        gray_float = gray.astype(np.float32)
        grad_x = cv2.filter2D(gray_float, -1, roberts_x)
        grad_y = cv2.filter2D(gray_float, -1, roberts_y)
        grad = np.sqrt(grad_x**2 + grad_y**2)

        return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def apply_attack(self, gray: np.ndarray) -> np.ndarray:
        if self.config.attack_type == "pixel_perturbation":
            return self._pixel_perturbation_attack(gray)
        elif self.config.attack_type == "edge_blur":
            return self._aggressive_edge_blur_attack(gray)
        elif self.config.attack_type == "gradient_reverse":
            return self._aggressive_gradient_reverse_attack(gray)
        elif self.config.attack_type == "contour_disrupt":
            return self._aggressive_contour_disruption_attack(gray)
        elif self.config.attack_type == "geometric":
            return self._geometric_attack(gray)
        elif self.config.attack_type == "occlusion":
            return self._occlusion_attack(gray)
        else:
            return gray

    def _aggressive_edge_blur_attack(self, gray: np.ndarray) -> np.ndarray:
        edges = self._sobel_edges(gray)

        attack_strength = 0.3
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

    def _aggressive_gradient_reverse_attack(self, gray: np.ndarray) -> np.ndarray:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        attack_strength = 0.1
        threshold = np.percentile(grad_mag, (1 - attack_strength) * 100)
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

    def _aggressive_contour_disruption_attack(self, gray: np.ndarray) -> np.ndarray:
        edges = self._sobel_edges(gray)
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

    def _pixel_perturbation_attack(self, gray: np.ndarray) -> np.ndarray:
        intensity = self.INTENSITIES[self.config.attack_intensity]["strength"]
        attacked = gray.astype(np.float32)
        num_pixels = int(gray.size * intensity * 0.1)

        h, w = gray.shape
        for _ in range(num_pixels):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            change = np.random.normal(0, intensity * 100)
            attacked[y, x] = np.clip(attacked[y, x] + change, 0, 255)

        return attacked.astype(np.uint8)

    def _geometric_attack(self, gray: np.ndarray) -> np.ndarray:
        intensity = self.INTENSITIES[self.config.attack_intensity]["strength"]
        h, w = gray.shape
        center = (w // 2, h // 2)

        angle = np.random.uniform(-intensity * 15, intensity * 15)
        scale = 1 + np.random.uniform(-intensity * 0.2, intensity * 0.2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        attacked = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return attacked

    def _occlusion_attack(self, gray: np.ndarray) -> np.ndarray:
        intensity = self.INTENSITIES[self.config.attack_intensity]["strength"]
        attacked = gray.copy()
        h, w = gray.shape

        num_patches = int(intensity * 5) + 1
        patch_size = int(intensity * min(h, w) * 0.2)

        for _ in range(num_patches):
            y = np.random.randint(0, max(1, h - patch_size))
            x = np.random.randint(0, max(1, w - patch_size))

            if np.random.rand() > 0.5:
                attacked[y:y+patch_size, x:x+patch_size] = np.random.randint(0, 256)
            else:
                region = attacked[y:y+patch_size, x:x+patch_size]
                attacked[y:y+patch_size, x:x+patch_size] = cv2.GaussianBlur(region, (15, 15), 5)

        return attacked

    def compute_attack_effectiveness(self, clean_edges: np.ndarray, attacked_edges: np.ndarray) -> dict:
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

    def run_simulation(self, image_path: str, output_dir: str = "effective_simulation_results") -> Dict[str, Any]:
        """Run simulation with aggressive attacks"""
        os.makedirs(output_dir, exist_ok=True)

        gray_clean = self.load_and_preprocess(image_path)
        edges_clean = self.compute_edges(gray_clean)
        gray_attacked = self.apply_attack(gray_clean)
        edges_attacked = self.compute_edges(gray_attacked)
        effectiveness = self.compute_attack_effectiveness(edges_clean, edges_attacked)

        base_name = f"sim_{self.config.detector}_{self.config.attack_type}_{self.config.attack_intensity}"

        cv2.imwrite(f"{output_dir}/{base_name}_original.png", gray_clean)
        cv2.imwrite(f"{output_dir}/{base_name}_attacked.png", gray_attacked)
        cv2.imwrite(f"{output_dir}/{base_name}_edges_clean.png", edges_clean)
        cv2.imwrite(f"{output_dir}/{base_name}_edges_attacked.png", edges_attacked)

        results = {
            "configuration": {
                "detector": self.config.detector,
                "attack_type": self.config.attack_type,
                "attack_intensity": self.config.attack_intensity,
                "lighting": self.config.lighting,
                "resolution": self.config.resolution
            },
            "metrics": effectiveness,
            "files": {
                "original": f"{base_name}_original.png",
                "attacked": f"{base_name}_attacked.png",
                "edges_clean": f"{base_name}_edges_clean.png",
                "edges_attacked": f"{base_name}_edges_attacked.png"
            }
        }

        with open(f"{output_dir}/{base_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

def main():
    simulator = EffectiveConfigurableSimulator()

    test_images = [
        "source_images/stop.png",
        "source_images/ped.jpg",
        "source_images/street.jpg"
    ]

    test_configs = [
        {"detector": "sobel", "attack_type": "contour_disrupt", "attack_intensity": "moderate"},
        {"detector": "canny", "attack_type": "edge_blur", "attack_intensity": "aggressive", "lighting": "low_light"},
        {"detector": "laplacian", "attack_type": "geometric", "attack_intensity": "subtle"},
        {"detector": "roberts", "attack_type": "occlusion", "attack_intensity": "extreme", "add_noise": True}
    ]

    all_results = {}
    total_successful = 0
    total_attacks = 0

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} - file not found")
            continue

        image_name = os.path.basename(image_path).split('.')[0]
        print(f"Processing: {image_name}")

        for i, config in enumerate(test_configs):
            simulator.configure(**config)
            results = simulator.run_simulation(image_path)

            attack_key = f"{image_name}_{config['detector']}_{config['attack_type']}"
            all_results[attack_key] = results

            effectiveness = results['metrics']
            success_str = "SUCCESS" if effectiveness["attack_success"] else "FAILED"
            print(f"  {config['attack_type']}: {success_str} (edge reduction: {effectiveness['edge_density_reduction']:.1%})")

            if effectiveness["attack_success"]:
                total_successful += 1
            total_attacks += 1

    print(f"\nResults saved to effective_simulation_results/")
    print(f"Successful attacks: {total_successful}/{total_attacks}")

if __name__ == "__main__":
    main()
