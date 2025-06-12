import os
import json
import cv2
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """
    /**
     * Configuration for edge-detection and adversarial-attack simulations.
     *
     * @property {string} edgeDetectorAlgorithm – Algorithm for edge detection ('sobel', 'canny', etc.).
     * @property {number} imageResolution – Target width/height (pixels) for resizing.
     * @property {number} cannyLowThreshold – Lower threshold for Canny detector.
     * @property {number} cannyHighThreshold – Upper threshold for Canny detector.
     * @property {number} cannyBlurKernelSize – Kernel size for pre-Canny Gaussian blur.
     * @property {number} sobelFilterKernelSize – Kernel size for Sobel filter.
     * @property {string} sobelGradientDirection – 'horizontal', 'vertical', or 'both'.
     * @property {string} adversarialAttackType – Type of adversarial attack.
     * @property {string} adversarialAttackStrength – Strength level of the attack.
     * @property {string} lightingMode – Lighting profile ('normal', 'low_light', etc.).
     * @property {boolean} includeNoise – Whether to add Gaussian noise.
     * @property {number} noiseIntensityRatio – Intensity ratio for added noise.
     */
    """
    edge_detector_algorithm: str           = "sobel"
    image_resolution: int                  = 256
    canny_low_threshold: int               = 100
    canny_high_threshold: int              = 200
    canny_blur_kernel_size: int            = 5
    sobel_filter_kernel_size: int          = 3
    sobel_gradient_direction: str          = "both"
    adversarial_attack_type: str           = "contour_disrupt"
    adversarial_attack_strength: str       = "moderate"
    lighting_mode: str                     = "normal"
    include_noise: bool                    = False
    noise_intensity_ratio: float           = 0.1


class EdgeAttackSimulationEngine:
    """
    /**
     * Engine to run edge-detection + adversarial-attack simulations.
     * @class
     */
    """

    def __init__(self) -> None:
        """
        /**
         * Initialize with default SimulationConfig and lookup tables.
         */
        """
        self.config = SimulationConfig()

        self.SUPPORTED_EDGE_DETECTORS: Dict[str, str] = {
            "sobel":       "Sobel Filter — gradient magnitude",
            "canny":       "Canny — multi-stage with hysteresis",
            "laplacian":   "Laplacian of Gaussian — second derivative",
            "roberts":     "Roberts Cross — simple cross kernels",
        }

        self.SUPPORTED_ATTACK_TYPES: Dict[str, str] = {
            "pixel_perturbation":    "Random pixel noise",
            "edge_blur":             "Targeted edge smoothing",
            "gradient_reverse":      "Reverse gradient direction",
            "contour_disrupt":       "Disrupt contour boundaries",
            "geometric":             "Rotation & scaling transforms",
            "occlusion":             "Strategic masking patches",
        }

        self.ATTACK_STRENGTH_LEVELS: Dict[str, Dict[str, Any]] = {
            "subtle":     {"strength": 0.2, "description": "Barely noticeable"},
            "moderate":   {"strength": 0.4, "description": "Visible but not obvious"},
            "aggressive": {"strength": 0.6, "description": "Clearly visible"},
            "extreme":    {"strength": 0.9, "description": "Heavily distorted"},
        }

        self.LIGHTING_MODES: Dict[str, Dict[str, float]] = {
            "normal":        {"brightness": 1.0, "contrast": 1.0},
            "low_light":     {"brightness": 0.6, "contrast": 0.8},
            "high_contrast": {"brightness": 1.2, "contrast": 1.5},
            "overexposed":   {"brightness": 1.8, "contrast": 0.7},
        }

    def configure(self, **overrides: Any) -> None:
        """
        /**
         * Override default simulation parameters.
         *
         * @param {Object} overrides – Key/value pairs of config parameters.
         * @returns {void}
         */
        """
        for key, val in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)
            else:
                print(f"⚠️ Warning: Unknown config parameter '{key}'")

    def load_and_preprocess_image(self, source_image_path: str) -> np.ndarray:
        """
        /**
         * Load an image, convert to grayscale, resize, adjust lighting & optionally add noise.
         *
         * @param {string} sourceImagePath – Path to the input image.
         * @returns {numpy.ndarray} Preprocessed 8-bit grayscale image.
         * @throws {FileNotFoundError} If the file cannot be loaded.
         */
        """
        bgr = cv2.imread(source_image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load '{source_image_path}'")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (self.config.image_resolution, self.config.image_resolution),
            interpolation=cv2.INTER_AREA
        )

        lighting = self.LIGHTING_MODES[self.config.lighting_mode]
        gray = gray.astype(np.float32)
        gray *= lighting["brightness"]
        gray = np.clip(128 + (gray - 128) * lighting["contrast"], 0, 255)

        if self.config.include_noise:
            noise = np.random.normal(
                0,
                self.config.noise_intensity_ratio * 255,
                gray.shape
            )
            gray = np.clip(gray + noise, 0, 255)

        return gray.astype(np.uint8)

    def detect_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        /**
         * Apply configured edge detector to a grayscale image.
         *
         * @param {numpy.ndarray} preprocessed – Grayscale input.
         * @returns {numpy.ndarray} Edge map as 8-bit image.
         * @throws {ValueError} For unsupported detector.
         */
        """
        alg = self.config.edge_detector_algorithm
        if alg == "sobel":
            return self._compute_sobel_edges(preprocessed)
        if alg == "canny":
            return self._compute_canny_edges(preprocessed)
        if alg == "laplacian":
            return self._compute_laplacian_edges(preprocessed)
        if alg == "roberts":
            return self._compute_roberts_edges(preprocessed)
        raise ValueError(f"Unsupported detector '{alg}'")

    def _compute_sobel_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Compute Sobel gradient magnitude.
         * @param {numpy.ndarray} gray – Grayscale image.
         * @returns {numpy.ndarray}
         */
        """
        k = self.config.sobel_filter_kernel_size
        dir = self.config.sobel_gradient_direction
        if dir == "horizontal":
            mag = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k))
        elif dir == "vertical":
            mag = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k))
        else:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            mag = np.sqrt(gx**2 + gy**2)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _compute_canny_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Compute Canny edges with optional pre-blur.
         */
        """
        if self.config.canny_blur_kernel_size > 1:
            gray = cv2.GaussianBlur(
                gray,
                (self.config.canny_blur_kernel_size,) * 2,
                0
            )
        return cv2.Canny(
            gray,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold
        )

    def _compute_laplacian_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Compute Laplacian of Gaussian edges.
         */
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        mag = np.abs(lap)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _compute_roberts_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Compute Roberts Cross edges.
         */
        """
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        f = gray.astype(np.float32)
        gx = cv2.filter2D(f, -1, kx)
        gy = cv2.filter2D(f, -1, ky)
        mag = np.sqrt(gx**2 + gy**2)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def apply_adversarial_attack(self, clean: np.ndarray) -> np.ndarray:
        """
        /**
         * Apply the configured adversarial attack.
         *
         * @param {numpy.ndarray} clean – Preprocessed input.
         * @returns {numpy.ndarray} Attacked image.
         */
        """
        t = self.config.adversarial_attack_type
        if t == "pixel_perturbation":
            return self._attack_pixel_perturbation(clean)
        if t == "edge_blur":
            return self._attack_edge_blur(clean)
        if t == "gradient_reverse":
            return self._attack_gradient_reverse(clean)
        if t == "contour_disrupt":
            return self._attack_contour_disrupt(clean)
        if t == "geometric":
            return self._attack_geometric(clean)
        if t == "occlusion":
            return self._attack_occlusion(clean)
        return clean

    def _attack_pixel_perturbation(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Randomly perturb pixels based on strength.
         */
        """
        strength = self.ATTACK_STRENGTH_LEVELS[self.config.adversarial_attack_strength]["strength"]
        perturbed = gray.astype(np.float32)
        count = int(gray.size * strength * 0.1)
        h, w = gray.shape
        for _ in range(count):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            delta = np.random.normal(0, strength * 100)
            perturbed[y, x] = np.clip(perturbed[y, x] + delta, 0, 255)
        return perturbed.astype(np.uint8)

    def _attack_edge_blur(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Blur strong edges selectively.
         """

        edges = self._compute_sobel_edges(gray)
        strength = self.ATTACK_STRENGTH_LEVELS[self.config.adversarial_attack_strength]["strength"]
        thresh = np.percentile(edges, (1 - strength) * 100)
        mask = edges > thresh

        buf = gray.copy().astype(np.float32)
        blurred = cv2.GaussianBlur(buf, (31, 31), 8.0)
        buf[mask] = blurred[mask]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        super_blur = cv2.GaussianBlur(buf, (21, 21), 5.0)
        buf[dil > 0] = super_blur[dil > 0]

        return buf.astype(np.uint8)

    def _attack_gradient_reverse(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Reverse gradient direction on strong edges.
         */
        """
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)

        strength = self.ATTACK_STRENGTH_LEVELS[self.config.adversarial_attack_strength]["strength"]
        thresh = np.percentile(mag, (1 - strength) * 100)
        edges = mag > thresh

        attacked = gray.copy().astype(np.float32)
        for y, x in np.argwhere(edges):
            if 1 <= y < gray.shape[0] - 1 and 1 <= x < gray.shape[1] - 1:
                gx, gy = sx[y, x], sy[y, x]
                m = min(100, mag[y, x] * 0.8)
                sign = np.sign(gx) if abs(gx) > abs(gy) else np.sign(gy)
                attacked[y, x] = np.clip(attacked[y, x] - sign * m, 0, 255)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gray.shape[0] and 0 <= nx < gray.shape[1]:
                            attacked[ny, nx] = np.clip(
                                attacked[ny, nx] - sign * m * 0.3, 0, 255
                            )
        return attacked.astype(np.uint8)

    def _attack_contour_disrupt(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Disrupt largest contours with blurring and random holes.
         */
        """
        edges = self._compute_sobel_edges(gray)
        _, bw = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        top3 = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        attacked = gray.copy()
        for c in top3:
            if cv2.contourArea(c) < 100:
                continue
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [c], -1, 255, thickness=20)
            blur1 = cv2.GaussianBlur(attacked, (41, 41), 10.0)
            attacked[mask > 0] = blur1[mask > 0]

            pts = c.reshape(-1, 2)
            holes = min(5, len(pts) // 10)
            for _ in range(holes):
                idx = np.random.randint(len(pts))
                cx, cy = pts[idx]
                s = 15
                y1, y2 = max(0, cy - s), min(gray.shape[0], cy + s)
                x1, x2 = max(0, cx - s), min(gray.shape[1], cx + s)
                attacked[y1:y2, x1:x2] = np.mean(gray)
        return attacked

    def _attack_geometric(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Apply random rotation & scaling.
         */
        """
        strength = self.ATTACK_STRENGTH_LEVELS[self.config.adversarial_attack_strength]["strength"]
        h, w = gray.shape
        angle = np.random.uniform(-strength * 15, strength * 15)
        scale = 1 + np.random.uniform(-strength * 0.2, strength * 0.2)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _attack_occlusion(self, gray: np.ndarray) -> np.ndarray:
        """
        /**
         * Mask random patches or blur them.
         */
        """
        strength = self.ATTACK_STRENGTH_LEVELS[self.config.adversarial_attack_strength]["strength"]
        attacked = gray.copy()
        h, w = gray.shape
        patches = int(strength * 5) + 1
        size = int(strength * min(h, w) * 0.2)
        for _ in range(patches):
            y, x = np.random.randint(0, h - size), np.random.randint(0, w - size)
            if np.random.rand() > 0.5:
                attacked[y:y+size, x:x+size] = np.random.randint(0, 256)
            else:
                region = attacked[y:y+size, x:x+size]
                attacked[y:y+size, x:x+size] = cv2.GaussianBlur(region, (15, 15), 5)
        return attacked

    def calculate_attack_effectiveness_metrics(
        self,
        reference_edges: np.ndarray,
        attacked_edges: np.ndarray
    ) -> Dict[str, Any]:
        """
        /**
         * Compare clean vs. attacked edge maps to compute metrics.
         *
         * @param {numpy.ndarray} referenceEdges – Edge map before attack.
         * @param {numpy.ndarray} attackedEdges – Edge map after attack.
         * @returns {Object} Metrics including density & contour reduction, plus success flag.
         */
        """
        density_pre = np.mean(reference_edges > 20)
        density_post = np.mean(attacked_edges > 20)
        reduction_density = ((density_pre - density_post) / density_pre) if density_pre > 0 else 0.0

        _, bin_pre = cv2.threshold(reference_edges, 20, 255, cv2.THRESH_BINARY)
        _, bin_post = cv2.threshold(attacked_edges, 20, 255, cv2.THRESH_BINARY)
        cont_pre, _ = cv2.findContours(bin_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_post, _ = cv2.findContours(bin_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_pre = max((cv2.contourArea(c) for c in cont_pre), default=0.0)
        max_post = max((cv2.contourArea(c) for c in cont_post), default=0.0)
        reduction_contour = ((max_pre - max_post) / max_pre) if max_pre > 0 else 0.0

        success = bool(
            reduction_density > 0.15 or reduction_contour > 0.2 or abs(reduction_density) > 0.3
        )

        return {
            "edge_density_reduction": float(reduction_density),
            "contour_area_reduction": float(reduction_contour),
            "attack_success": success,
        }

    def execute_simulation_on_image(
        self,
        source_image_path: str,
        results_directory: str = "effective_simulation_results"
    ) -> Dict[str, Any]:
        """
        /**
         * Run full pipeline on an image: preprocess, detect edges, attack, re-detect, save outputs & metrics.
         *
         * @param {string} sourceImagePath – Input image path.
         * @param {string} resultsDirectory – Directory for outputs.
         * @returns {Object} Summary of config, metrics & file names.
         */
        """
        os.makedirs(results_directory, exist_ok=True)

        gray_clean    = self.load_and_preprocess_image(source_image_path)
        edges_clean   = self.detect_edges(gray_clean)
        gray_attacked = self.apply_adversarial_attack(gray_clean)
        edges_attacked= self.detect_edges(gray_attacked)
        metrics       = self.calculate_attack_effectiveness_metrics(edges_clean, edges_attacked)

        base = os.path.splitext(os.path.basename(source_image_path))[0]
        sim_id = (
            f"{base}_sim_{self.config.edge_detector_algorithm}_"
            f"{self.config.adversarial_attack_type}_"
            f"{self.config.adversarial_attack_strength}"
        )

        # Save images
        cv2.imwrite(f"{results_directory}/{sim_id}_original.png",      gray_clean)
        cv2.imwrite(f"{results_directory}/{sim_id}_attacked.png",      gray_attacked)
        cv2.imwrite(f"{results_directory}/{sim_id}_edges_clean.png",   edges_clean)
        cv2.imwrite(f"{results_directory}/{sim_id}_edges_attacked.png",edges_attacked)

        payload = {
            "configuration": {
                "edge_detector_algorithm":        self.config.edge_detector_algorithm,
                "adversarial_attack_type":        self.config.adversarial_attack_type,
                "adversarial_attack_strength":    self.config.adversarial_attack_strength,
                "lighting_mode":                  self.config.lighting_mode,
                "image_resolution":               self.config.image_resolution,
            },
            "metrics": metrics,
            "files": {
                "original":      f"{sim_id}_original.png",
                "attacked":      f"{sim_id}_attacked.png",
                "edges_clean":   f"{sim_id}_edges_clean.png",
                "edges_attacked":f"{sim_id}_edges_attacked.png",
            },
        }

        with open(f"{results_directory}/{sim_id}_results.json", "w") as jf:
            json.dump(payload, jf, indent=2)

        return payload


def main() -> None:
    """
    /**
     * Batch-process a set of images with various configurations.
     */
    """
    engine = EdgeAttackSimulationEngine()

    image_sources = [
        "source_images/stop.png",
        "source_images/ped.jpg",
        "source_images/street.jpg",
    ]

    batch_configs = [
        {"edge_detector_algorithm":          "sobel",
         "adversarial_attack_type":          "contour_disrupt",
         "adversarial_attack_strength":      "moderate"},
        {"edge_detector_algorithm":          "canny",
         "adversarial_attack_type":          "edge_blur",
         "adversarial_attack_strength":      "aggressive",
         "lighting_mode":                    "low_light"},
        {"edge_detector_algorithm":          "laplacian",
         "adversarial_attack_type":          "geometric",
         "adversarial_attack_strength":      "subtle"},
        {"edge_detector_algorithm":          "roberts",
         "adversarial_attack_type":          "occlusion",
         "adversarial_attack_strength":      "extreme",
         "include_noise":                    True},
    ]

    total_success = 0
    total_runs    = 0

    for img_path in image_sources:
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing file: {img_path}")
            continue
        print(f"Processing '{img_path}'")
        for cfg in batch_configs:
            engine.configure(**cfg)
            result = engine.execute_simulation_on_image(img_path)
            success = result["metrics"]["attack_success"]
            status  = "SUCCESS" if success else "FAILED"
            print(f"  {cfg['adversarial_attack_type']}: {status} (density reduction: {result['metrics']['edge_density_reduction']:.1%})")
            total_success += int(success)
            total_runs    += 1

    print(f"\nResults saved in 'effective_simulation_results/'. {total_success}/{total_runs} attacks succeeded.")


if __name__ == "__main__":
    main()
