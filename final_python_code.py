#!/usr/bin/env python3
"""
Configurable Edge Detection Attack Simulator
Perfect for interactive web demonstrations
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Configuration for the edge detection simulation"""

    # Edge Detection Settings
    detector: str = "sobel"
    resolution: int = 256

    # Canny-specific parameters
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    canny_blur_kernel: int = 5

    # Sobel-specific parameters
    sobel_kernel_size: int = 3
    sobel_direction: str = "both"  # "both", "horizontal", "vertical"

    # Attack settings
    attack_type: str = "contour_disrupt"
    attack_intensity: str = "moderate"

    # Environmental conditions
    lighting: str = "normal"
    add_noise: bool = False
    noise_level: float = 0.1

class ConfigurableEdgeSimulator:
    def __init__(self):
        self.config = SimulationConfig()

        # Define available options
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
            "subtle": {"strength": 0.1, "desc": "Barely noticeable"},
            "moderate": {"strength": 0.3, "desc": "Visible but not obvious"},
            "aggressive": {"strength": 0.5, "desc": "Clearly visible changes"},
            "extreme": {"strength": 0.8, "desc": "Heavily distorted"}
        }

        self.LIGHTING = {
            "normal": {"brightness": 1.0, "contrast": 1.0},
            "low_light": {"brightness": 0.6, "contrast": 0.8},
            "high_contrast": {"brightness": 1.2, "contrast": 1.5},
            "overexposed": {"brightness": 1.8, "contrast": 0.7}
        }

    def configure(self, **kwargs):
        """Update simulation configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load image and apply environmental conditions"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load: {image_path}")

        # Convert to grayscale and resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.config.resolution, self.config.resolution))

        # Apply lighting conditions
        lighting = self.LIGHTING[self.config.lighting]
        gray = gray.astype(np.float32)
        gray = gray * lighting["brightness"]
        gray = np.clip(128 + (gray - 128) * lighting["contrast"], 0, 255)

        # Add noise if enabled
        if self.config.add_noise:
            noise = np.random.normal(0, self.config.noise_level * 255, gray.shape)
            gray = np.clip(gray + noise, 0, 255)

        return gray.astype(np.uint8)

    def compute_edges(self, gray: np.ndarray) -> np.ndarray:
        """Compute edges using configured detector"""

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
        """Sobel edge detection with configurable parameters"""
        ksize = self.config.sobel_kernel_size

        if self.config.sobel_direction == "horizontal":
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            grad = np.abs(sobel)
        elif self.config.sobel_direction == "vertical":
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = np.abs(sobel)
        else:  # both
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            grad = np.sqrt(sobel_x**2 + sobel_y**2)

        return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _canny_edges(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge detection with configurable parameters"""
        # Apply Gaussian blur first
        if self.config.canny_blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.config.canny_blur_kernel, self.config.canny_blur_kernel), 0)

        return cv2.Canny(gray, self.config.canny_low_threshold, self.config.canny_high_threshold)

    def _laplacian_edges(self, gray: np.ndarray) -> np.ndarray:
        """Laplacian of Gaussian edge detection"""
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _roberts_edges(self, gray: np.ndarray) -> np.ndarray:
        """Roberts cross-gradient edge detection"""
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        gray_float = gray.astype(np.float32)
        grad_x = cv2.filter2D(gray_float, -1, roberts_x)
        grad_y = cv2.filter2D(gray_float, -1, roberts_y)
        grad = np.sqrt(grad_x**2 + grad_y**2)

        return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def apply_attack(self, gray: np.ndarray) -> np.ndarray:
        """Apply configured attack to image"""
        intensity = self.INTENSITIES[self.config.attack_intensity]["strength"]

        if self.config.attack_type == "pixel_perturbation":
            return self._pixel_perturbation_attack(gray, intensity)
        elif self.config.attack_type == "edge_blur":
            return self._edge_blur_attack(gray, intensity)
        elif self.config.attack_type == "gradient_reverse":
            return self._gradient_reverse_attack(gray, intensity)
        elif self.config.attack_type == "contour_disrupt":
            return self._contour_disruption_attack(gray, intensity)
        elif self.config.attack_type == "geometric":
            return self._geometric_attack(gray, intensity)
        elif self.config.attack_type == "occlusion":
            return self._occlusion_attack(gray, intensity)
        else:
            return gray

    def _pixel_perturbation_attack(self, gray: np.ndarray, intensity: float) -> np.ndarray:
        """Random pixel perturbation attack"""
        attacked = gray.astype(np.float32)
        num_pixels = int(gray.size * intensity * 0.1)  # Affect 10% at max intensity

        h, w = gray.shape
        for _ in range(num_pixels):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            change = np.random.normal(0, intensity * 100)
            attacked[y, x] = np.clip(attacked[y, x] + change, 0, 255)

        return attacked.astype(np.uint8)

    def _edge_blur_attack(self, gray: np.ndarray, intensity: float) -> np.ndarray:
        """Targeted edge blurring attack"""
        edges = self.compute_edges(gray)
        threshold = np.percentile(edges, (1 - intensity) * 100)
        edge_mask = edges > threshold

        attacked = gray.copy().astype(np.float32)
        blur_size = int(intensity * 20) * 2 + 1  # Odd number for kernel
        blurred = cv2.GaussianBlur(attacked, (blur_size, blur_size), intensity * 5)
        attacked[edge_mask] = blurred[edge_mask]

        return attacked.astype(np.uint8)

    def _gradient_reverse_attack(self, gray: np.ndarray, intensity: float) -> np.ndarray:
        """Gradient direction reversal attack"""
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        threshold = np.percentile(grad_mag, (1 - intensity) * 100)
        strong_edges = grad_mag > threshold

        attacked = gray.copy().astype(np.float32)

        for y, x in np.argwhere(strong_edges):
            if 1 <= y < gray.shape[0]-1 and 1 <= x < gray.shape[1]-1:
                gx, gy = sobel_x[y, x], sobel_y[y, x]
                magnitude = min(100, grad_mag[y, x] * intensity)

                if abs(gx) > abs(gy):
                    attacked[y, x] = np.clip(attacked[y, x] - np.sign(gx) * magnitude, 0, 255)
                else:
                    attacked[y, x] = np.clip(attacked[y, x] - np.sign(gy) * magnitude, 0, 255)

        return attacked.astype(np.uint8)

    def _contour_disruption_attack(self, gray: np.ndarray, intensity: float) -> np.ndarray:
        """Contour boundary disruption attack"""
        edges = self.compute_edges(gray)
        _, binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        attacked = gray.copy()
        num_contours = min(3, len(contours))
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            mask = np.zeros_like(gray)
            thickness = int(intensity * 30) + 5
            cv2.drawContours(mask, [contour], -1, 255, thickness=thickness)

            blur_size = int(intensity * 40) + 1
            if blur_size % 2 == 0:
                blur_size += 1
            blurred = cv2.GaussianBlur(attacked, (blur_size, blur_size), intensity * 8)
            attacked[mask > 0] = blurred[mask > 0]

        return attacked

    def _geometric_attack(self, gray: np.ndarray, intensity: float) -> np.ndarray:
        """Geometric transformation attack"""
        h, w = gray.shape
        center = (w // 2, h // 2)

        # Random rotation and scaling
        angle = np.random.uniform(-intensity * 15, intensity * 15)  # Max 15 degrees
        scale = 1 + np.random.uniform(-intensity * 0.2, intensity * 0.2)  # Max 20% scale change

        M = cv2.getRotationMatrix2D(center, angle, scale)
        attacked = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return attacked

    def _occlusion_attack(self, gray: np.ndarray, intensity: float) -> np.ndarray:
        """Strategic occlusion attack"""
        attacked = gray.copy()
        h, w = gray.shape

        # Number and size of occlusion patches based on intensity
        num_patches = int(intensity * 5) + 1
        patch_size = int(intensity * min(h, w) * 0.2)

        for _ in range(num_patches):
            y = np.random.randint(0, max(1, h - patch_size))
            x = np.random.randint(0, max(1, w - patch_size))

            # Fill with random color or blur
            if np.random.rand() > 0.5:
                attacked[y:y+patch_size, x:x+patch_size] = np.random.randint(0, 256)
            else:
                region = attacked[y:y+patch_size, x:x+patch_size]
                attacked[y:y+patch_size, x:x+patch_size] = cv2.GaussianBlur(region, (15, 15), 5)

        return attacked

    def compute_metrics(self, clean_edges: np.ndarray, attacked_edges: np.ndarray) -> Dict[str, float]:
        """Compute attack effectiveness metrics"""
        # Edge density comparison
        clean_density = np.mean(clean_edges > 20)
        attacked_density = np.mean(attacked_edges > 20)
        density_change = (attacked_density - clean_density) / clean_density if clean_density > 0 else 0

        # Structural similarity
        from skimage.metrics import structural_similarity as ssim
        similarity = ssim(clean_edges, attacked_edges)

        # Contour analysis
        _, clean_binary = cv2.threshold(clean_edges, 20, 255, cv2.THRESH_BINARY)
        _, attacked_binary = cv2.threshold(attacked_edges, 20, 255, cv2.THRESH_BINARY)

        clean_contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        attacked_contours, _ = cv2.findContours(attacked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clean_max_area = max([cv2.contourArea(c) for c in clean_contours]) if clean_contours else 0
        attacked_max_area = max([cv2.contourArea(c) for c in attacked_contours]) if attacked_contours else 0

        contour_change = (attacked_max_area - clean_max_area) / clean_max_area if clean_max_area > 0 else 0

        return {
            "edge_density_change": float(density_change),
            "contour_area_change": float(contour_change),
            "structural_similarity": float(similarity),
            "attack_effectiveness": float(abs(density_change) + abs(contour_change) + (1 - similarity))
        }

    def run_simulation(self, image_path: str, output_dir: str = "simulation_results") -> Dict[str, Any]:
        """Run complete simulation with current configuration"""
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess image
        gray_clean = self.load_and_preprocess(image_path)

        # Compute clean edges
        edges_clean = self.compute_edges(gray_clean)

        # Apply attack
        gray_attacked = self.apply_attack(gray_clean)

        # Compute attacked edges
        edges_attacked = self.compute_edges(gray_attacked)

        # Compute metrics
        metrics = self.compute_metrics(edges_clean, edges_attacked)

        # Save results
        base_name = f"sim_{self.config.detector}_{self.config.attack_type}_{self.config.attack_intensity}"

        cv2.imwrite(f"{output_dir}/{base_name}_original.png", gray_clean)
        cv2.imwrite(f"{output_dir}/{base_name}_attacked.png", gray_attacked)
        cv2.imwrite(f"{output_dir}/{base_name}_edges_clean.png", edges_clean)
        cv2.imwrite(f"{output_dir}/{base_name}_edges_attacked.png", edges_attacked)

        # Return comprehensive results
        results = {
            "configuration": {
                "detector": self.config.detector,
                "attack_type": self.config.attack_type,
                "attack_intensity": self.config.attack_intensity,
                "lighting": self.config.lighting,
                "resolution": self.config.resolution
            },
            "metrics": metrics,
            "files": {
                "original": f"{base_name}_original.png",
                "attacked": f"{base_name}_attacked.png",
                "edges_clean": f"{base_name}_edges_clean.png",
                "edges_attacked": f"{base_name}_edges_attacked.png"
            }
        }

        # Save metadata
        with open(f"{output_dir}/{base_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

def main():
    """Example usage demonstrating different configurations"""
    simulator = ConfigurableEdgeSimulator()

    # Test different configurations
    test_configs = [
        {"detector": "sobel", "attack_type": "contour_disrupt", "attack_intensity": "moderate"},
        {"detector": "canny", "attack_type": "edge_blur", "attack_intensity": "aggressive", "lighting": "low_light"},
        {"detector": "laplacian", "attack_type": "geometric", "attack_intensity": "subtle"},
        {"detector": "roberts", "attack_type": "occlusion", "attack_intensity": "extreme", "add_noise": True}
    ]

    image_path = "source_images/stop.png"  # Update this path

    for i, config in enumerate(test_configs):
        print(f"\nRunning simulation {i+1}/{len(test_configs)}")
        print(f"Config: {config}")

        simulator.configure(**config)
        results = simulator.run_simulation(image_path)

        print(f"Attack effectiveness: {results['metrics']['attack_effectiveness']:.3f}")
        print(f"Edge density change: {results['metrics']['edge_density_change']:.1%}")

if __name__ == "__main__":
    main()
