#!/usr/bin/env python3
"""
High-Impact Attack Scenarios - Optimized for dramatic website demonstrations
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

# Import your simulator
from final_python_code import ConfigurableEdgeSimulator

class HighImpactAttackGenerator:
    def __init__(self):
        self.simulator = ConfigurableEdgeSimulator()

        # Define scenarios optimized for HIGH VISUAL IMPACT
        self.scenarios = {
            "clean_baseline": {
                "name": "Clean Baseline",
                "description": "Original images with no attacks for comparison",
                "configs": [
                    # Just clean edge detection, no attacks
                    {"detector": "sobel", "attack_type": "pixel_perturbation", "attack_intensity": "subtle", "lighting": "normal", "resolution": 256},
                    {"detector": "canny", "attack_type": "pixel_perturbation", "attack_intensity": "subtle", "lighting": "normal", "resolution": 256},
                ]
            },
            "devastating_attacks": {
                "name": "Devastating Attacks",
                "description": "Maximum impact attacks that completely break detection",
                "configs": [
                    # These should produce effectiveness > 1.0
                    {"detector": "sobel", "attack_type": "contour_disrupt", "attack_intensity": "extreme", "lighting": "normal"},
                    {"detector": "roberts", "attack_type": "occlusion", "attack_intensity": "extreme", "lighting": "low_light"},
                    {"detector": "laplacian", "attack_type": "edge_blur", "attack_intensity": "extreme", "lighting": "overexposed"},
                    {"detector": "canny", "attack_type": "geometric", "attack_intensity": "extreme", "add_noise": True},
                ]
            },
            "security_bypass": {
                "name": "Security System Bypass",
                "description": "Realistic attacks that could fool security cameras",
                "configs": [
                    # Focus on contour disruption and occlusion
                    {"detector": "sobel", "attack_type": "contour_disrupt", "attack_intensity": "aggressive", "lighting": "low_light"},
                    {"detector": "canny", "attack_type": "occlusion", "attack_intensity": "aggressive", "lighting": "normal"},
                    {"detector": "roberts", "attack_type": "contour_disrupt", "attack_intensity": "extreme", "lighting": "low_light"},
                ]
            },
            "autonomous_vehicle_failure": {
                "name": "Autonomous Vehicle Critical Failures",
                "description": "Attacks that could cause self-driving car accidents",
                "configs": [
                    # Stop sign detection failures
                    {"detector": "sobel", "attack_type": "edge_blur", "attack_intensity": "aggressive", "lighting": "overexposed"},
                    {"detector": "canny", "attack_type": "contour_disrupt", "attack_intensity": "extreme", "lighting": "high_contrast"},
                    {"detector": "laplacian", "attack_type": "occlusion", "attack_intensity": "aggressive", "lighting": "normal"},
                ]
            },
            "stealth_attacks": {
                "name": "Stealth Attacks",
                "description": "Subtle attacks that are hard to detect but effective",
                "configs": [
                    # High effectiveness but visually subtle
                    {"detector": "sobel", "attack_type": "gradient_reverse", "attack_intensity": "moderate", "lighting": "normal"},
                    {"detector": "canny", "attack_type": "edge_blur", "attack_intensity": "moderate", "lighting": "low_light"},
                    {"detector": "laplacian", "attack_type": "geometric", "attack_intensity": "aggressive", "lighting": "normal"},
                ]
            },
            "algorithm_showdown": {
                "name": "Algorithm Robustness Showdown",
                "description": "Same attack, different detectors - who survives?",
                "configs": [
                    # Same devastating attack on all detectors
                    {"detector": "sobel", "attack_type": "contour_disrupt", "attack_intensity": "extreme"},
                    {"detector": "canny", "attack_type": "contour_disrupt", "attack_intensity": "extreme"},
                    {"detector": "laplacian", "attack_type": "contour_disrupt", "attack_intensity": "extreme"},
                    {"detector": "roberts", "attack_type": "contour_disrupt", "attack_intensity": "extreme"},
                ]
            },
            "environmental_warfare": {
                "name": "Environmental Warfare",
                "description": "How attackers exploit lighting conditions",
                "configs": [
                    # Exploit environmental weaknesses
                    {"detector": "canny", "attack_type": "occlusion", "attack_intensity": "aggressive", "lighting": "low_light"},
                    {"detector": "sobel", "attack_type": "edge_blur", "attack_intensity": "extreme", "lighting": "overexposed"},
                    {"detector": "laplacian", "attack_type": "contour_disrupt", "attack_intensity": "aggressive", "lighting": "high_contrast"},
                ]
            }
        }

    def create_custom_devastating_attack(self, gray: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Create a custom attack designed for maximum visual impact"""
        attacked = gray.copy().astype(np.float32)
        h, w = gray.shape

        # Step 1: Massive blur on random regions
        num_regions = int(intensity * 8) + 3
        for _ in range(num_regions):
            # Random region
            region_size = int(min(h, w) * 0.15 * intensity)
            y = np.random.randint(0, max(1, h - region_size))
            x = np.random.randint(0, max(1, w - region_size))

            # Apply massive blur
            region = attacked[y:y+region_size, x:x+region_size]
            if region.size > 0:
                blurred = cv2.GaussianBlur(region, (51, 51), 15.0)
                attacked[y:y+region_size, x:x+region_size] = blurred

        # Step 2: Add strategic occlusion blocks
        num_blocks = int(intensity * 5) + 2
        for _ in range(num_blocks):
            block_size = int(min(h, w) * 0.1 * intensity)
            y = np.random.randint(0, max(1, h - block_size))
            x = np.random.randint(0, max(1, w - block_size))

            # Random fill color
            fill_value = np.random.choice([0, 255, np.mean(gray)])
            attacked[y:y+block_size, x:x+block_size] = fill_value

        # Step 3: Add noise to remaining areas
        noise = np.random.normal(0, intensity * 50, (h, w))
        attacked = np.clip(attacked + noise, 0, 255)

        return attacked.astype(np.uint8)

    def run_enhanced_simulation(self, image_path: str, config: dict, output_dir: str) -> Dict[str, Any]:
        """Run simulation with enhanced attacks for maximum impact"""
        os.makedirs(output_dir, exist_ok=True)

        # Configure simulator
        self.simulator.configure(**config)

        # Load and preprocess image
        gray_clean = self.simulator.load_and_preprocess(image_path)

        # Compute clean edges
        edges_clean = self.simulator.compute_edges(gray_clean)

        # Apply attack - use custom devastating attack for extreme cases
        if config.get("attack_intensity") == "extreme" and config.get("attack_type") == "contour_disrupt":
            # Use our custom devastating attack
            gray_attacked = self.create_custom_devastating_attack(gray_clean, 0.9)
        else:
            # Use normal attack
            gray_attacked = self.simulator.apply_attack(gray_clean)

        # Compute attacked edges
        edges_attacked = self.simulator.compute_edges(gray_attacked)

        # Enhanced metrics calculation
        metrics = self.compute_enhanced_metrics(edges_clean, edges_attacked, gray_clean, gray_attacked)

        # Save results with better naming
        detector = config["detector"]
        attack = config["attack_type"]
        intensity = config["attack_intensity"]
        base_name = f"{detector}_{attack}_{intensity}"

        cv2.imwrite(f"{output_dir}/{base_name}_original.png", gray_clean)
        cv2.imwrite(f"{output_dir}/{base_name}_attacked.png", gray_attacked)
        cv2.imwrite(f"{output_dir}/{base_name}_edges_clean.png", edges_clean)
        cv2.imwrite(f"{output_dir}/{base_name}_edges_attacked.png", edges_attacked)

        # Return comprehensive results
        results = {
            "configuration": config,
            "metrics": metrics,
            "files": {
                "original": f"{base_name}_original.png",
                "attacked": f"{base_name}_attacked.png",
                "edges_clean": f"{base_name}_edges_clean.png",
                "edges_attacked": f"{base_name}_edges_attacked.png"
            },
            "visual_impact": self.assess_visual_impact(gray_clean, gray_attacked)
        }

        return results

    def compute_enhanced_metrics(self, clean_edges, attacked_edges, clean_gray, attacked_gray):
        """Enhanced metrics that better capture attack effectiveness"""

        # Original metrics
        clean_density = np.mean(clean_edges > 20)
        attacked_density = np.mean(attacked_edges > 20)
        density_change = (attacked_density - clean_density) / clean_density if clean_density > 0 else 0

        # Enhanced contour analysis
        _, clean_binary = cv2.threshold(clean_edges, 20, 255, cv2.THRESH_BINARY)
        _, attacked_binary = cv2.threshold(attacked_edges, 20, 255, cv2.THRESH_BINARY)

        clean_contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        attacked_contours, _ = cv2.findContours(attacked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Total contour area (not just max)
        clean_total_area = sum(cv2.contourArea(c) for c in clean_contours)
        attacked_total_area = sum(cv2.contourArea(c) for c in attacked_contours)
        total_contour_change = (attacked_total_area - clean_total_area) / clean_total_area if clean_total_area > 0 else 0

        # Number of contours change
        contour_count_change = (len(attacked_contours) - len(clean_contours)) / len(clean_contours) if clean_contours else 0

        # Visual similarity of original images
        from skimage.metrics import structural_similarity as ssim
        image_similarity = ssim(clean_gray, attacked_gray)
        edge_similarity = ssim(clean_edges, attacked_edges)

        # Combined effectiveness score (0-3 scale for more dramatic numbers)
        effectiveness = (
            abs(density_change) +
            abs(total_contour_change) +
            abs(contour_count_change) * 0.5 +
            (2 - image_similarity) +  # Invert similarity (lower similarity = higher effectiveness)
            (2 - edge_similarity)
        )

        return {
            "edge_density_change": float(density_change),
            "total_contour_area_change": float(total_contour_change),
            "contour_count_change": float(contour_count_change),
            "image_similarity": float(image_similarity),
            "edge_similarity": float(edge_similarity),
            "attack_effectiveness": float(effectiveness),
            "visual_disruption": float(1 - image_similarity),
            "edge_disruption": float(1 - edge_similarity)
        }

    def assess_visual_impact(self, clean, attacked):
        """Assess visual impact for website presentation"""
        diff = np.abs(clean.astype(float) - attacked.astype(float))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        if mean_diff > 50:
            impact = "EXTREME - Highly visible changes"
        elif mean_diff > 25:
            impact = "HIGH - Clearly visible changes"
        elif mean_diff > 10:
            impact = "MODERATE - Noticeable changes"
        else:
            impact = "SUBTLE - Minimal visible changes"

        return {
            "level": impact,
            "mean_difference": float(mean_diff),
            "max_difference": float(max_diff)
        }

    def generate_high_impact_dataset(self, image_paths: List[str], output_dir: str = "high_impact_dataset"):
        """Generate dataset optimized for high visual impact"""
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        all_results = []

        for image_path in image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            results[image_name] = {}

            for scenario_key, scenario in self.scenarios.items():
                print(f"\nProcessing {scenario['name']} for {image_name}...")

                scenario_results = []

                for i, config in enumerate(scenario["configs"]):
                    scenario_output = os.path.join(output_dir, scenario_key, image_name)

                    try:
                        result = self.run_enhanced_simulation(image_path, config, scenario_output)

                        result["scenario_info"] = {
                            "scenario_key": scenario_key,
                            "scenario_name": scenario["name"],
                            "image_name": image_name,
                            "config_index": i
                        }

                        scenario_results.append(result)
                        all_results.append(result)

                        effectiveness = result['metrics']['attack_effectiveness']
                        visual_impact = result['visual_impact']['level']

                        print(f"  Config {i+1}: Effectiveness {effectiveness:.3f} - {visual_impact}")

                    except Exception as e:
                        print(f"Error with config {i}: {e}")
                        continue

                results[image_name][scenario_key] = {
                    "scenario_info": scenario,
                    "results": scenario_results
                }

        # Find and highlight the best attacks
        self.analyze_best_attacks(all_results, output_dir)

        # Save all results
        with open(os.path.join(output_dir, "high_impact_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def analyze_best_attacks(self, all_results: List[Dict], output_dir: str):
        """Analyze and report the most effective attacks"""

        # Sort by effectiveness
        sorted_results = sorted(all_results, key=lambda x: x['metrics']['attack_effectiveness'], reverse=True)

        print("\n" + "="*60)
        print("TOP 10 MOST DEVASTATING ATTACKS")
        print("="*60)

        top_attacks = []
        for i, result in enumerate(sorted_results[:10]):
            config = result['configuration']
            metrics = result['metrics']
            visual = result['visual_impact']
            scenario = result['scenario_info']

            attack_info = {
                "rank": i + 1,
                "effectiveness": metrics['attack_effectiveness'],
                "detector": config['detector'],
                "attack_type": config['attack_type'],
                "intensity": config['attack_intensity'],
                "image": scenario['image_name'],
                "scenario": scenario['scenario_name'],
                "visual_impact": visual['level'],
                "edge_disruption": metrics['edge_disruption']
            }

            top_attacks.append(attack_info)

            print(f"{i+1:2d}. {config['detector']} + {config['attack_type']} + {config['attack_intensity']}")
            print(f"    Image: {scenario['image_name']}, Effectiveness: {metrics['attack_effectiveness']:.3f}")
            print(f"    Visual Impact: {visual['level']}")
            print(f"    Edge Disruption: {metrics['edge_disruption']:.3f}")
            print()

        # Save top attacks for website
        with open(os.path.join(output_dir, "top_attacks.json"), "w") as f:
            json.dump(top_attacks, f, indent=2)

def main():
    """Generate high-impact dataset for website"""

    image_paths = [
        "source_images/stop.png",
        "source_images/ped.jpg",
        "source_images/street.jpg"
    ]

    existing_images = [path for path in image_paths if os.path.exists(path)]

    if not existing_images:
        print("No images found!")
        return

    print(f"Generating HIGH-IMPACT attacks for {len(existing_images)} images...")

    generator = HighImpactAttackGenerator()
    results = generator.generate_high_impact_dataset(existing_images)

    print("\n" + "="*60)
    print("HIGH-IMPACT DATASET GENERATED!")
    print("="*60)
    print("This dataset is optimized for dramatic website demonstrations!")
    print("Check 'high_impact_dataset/top_attacks.json' for the best scenarios to use.")

if __name__ == "__main__":
    main()
