#!/usr/bin/env python3
"""
Visualize attack results with side-by-side comparisons
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = "targeted_attacks"
results_file = os.path.join(RESULTS_DIR, "attack_results.json")

def load_results():
    with open(results_file, 'r') as f:
        return json.load(f)

def create_comparison_grid(image_set, title):
    """Create a 2x2 grid showing: original gray, attacked gray, original edges, attacked edges"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    gray_clean = cv2.imread(f"{RESULTS_DIR}/{image_set}_gray_clean.png", cv2.IMREAD_GRAYSCALE)
    gray_attacked = cv2.imread(f"{RESULTS_DIR}/{image_set}_gray_attacked.png", cv2.IMREAD_GRAYSCALE)
    edges_clean = cv2.imread(f"{RESULTS_DIR}/{image_set}_edges_clean.png", cv2.IMREAD_GRAYSCALE)
    edges_attacked = cv2.imread(f"{RESULTS_DIR}/{image_set}_edges_attacked.png", cv2.IMREAD_GRAYSCALE)
    axes[0,0].imshow(gray_clean, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')

    axes[0,1].imshow(gray_attacked, cmap='gray')
    axes[0,1].set_title('Attacked Image')
    axes[0,1].axis('off')

    axes[1,0].imshow(edges_clean, cmap='gray')
    axes[1,0].set_title('Original Edges')
    axes[1,0].axis('off')

    axes[1,1].imshow(edges_attacked, cmap='gray')
    axes[1,1].set_title('Attacked Edges')
    axes[1,1].axis('off')

    plt.tight_layout()
    return fig

def create_summary_visualization():
    """Create a comprehensive summary of all attacks"""
    results = load_results()

    # Group by source image
    sources = {}
    for key, data in results.items():
        src = data['source']
        if src not in sources:
            sources[src] = []
        sources[src].append((key, data))

    # Create summary plots for each source
    for src_name, attacks in sources.items():
        fig, axes = plt.subplots(len(attacks), 4, figsize=(16, 4*len(attacks)))
        if len(attacks) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Attack Summary: {src_name.upper()}', fontsize=18, fontweight='bold')

        for i, (image_set, data) in enumerate(attacks):
            attack_method = data['attack_method']
            edge_reduction = data['edge_density_reduction']
            contour_reduction = data['contour_area_reduction']
            success = data['attack_success']

            # Load images
            try:
                gray_clean = cv2.imread(f"{RESULTS_DIR}/{image_set}_gray_clean.png", cv2.IMREAD_GRAYSCALE)
                gray_attacked = cv2.imread(f"{RESULTS_DIR}/{image_set}_gray_attacked.png", cv2.IMREAD_GRAYSCALE)
                edges_clean = cv2.imread(f"{RESULTS_DIR}/{image_set}_edges_clean.png", cv2.IMREAD_GRAYSCALE)
                edges_attacked = cv2.imread(f"{RESULTS_DIR}/{image_set}_edges_attacked.png", cv2.IMREAD_GRAYSCALE)

                axes[i,0].imshow(gray_clean, cmap='gray')
                axes[i,0].set_title('Original')
                axes[i,0].axis('off')

                axes[i,1].imshow(gray_attacked, cmap='gray')
                axes[i,1].set_title('Attacked')
                axes[i,1].axis('off')

                axes[i,2].imshow(edges_clean, cmap='gray')
                axes[i,2].set_title('Original Edges')
                axes[i,2].axis('off')

                axes[i,3].imshow(edges_attacked, cmap='gray')
                axes[i,3].set_title('Attacked Edges')
                axes[i,3].axis('off')

                # Add attack info
                status_color = 'green' if success else 'red'
                status_text = 'SUCCESS' if success else 'FAILED'

                attack_info = (f"{attack_method.replace('_', ' ').title()}\n"
                             f"Status: {status_text}\n"
                             f"Edge Δ: {edge_reduction:.1%}\n"
                             f"Contour Δ: {contour_reduction:.1%}")

                axes[i,1].text(0.02, 0.98, attack_info, transform=axes[i,1].transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))

            except Exception as e:
                print(f"Error loading images for {image_set}: {e}")

        plt.tight_layout()

        # Save summary plot
        output_path = f"{RESULTS_DIR}/summary_{src_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary: {output_path}")
        plt.show()

def create_effectiveness_chart():
    """Create a chart showing attack effectiveness"""
    results = load_results()

    attacks = []
    edge_reductions = []
    contour_reductions = []
    successes = []

    for key, data in results.items():
        attacks.append(f"{data['source']}\n{data['attack_method']}")
        edge_reductions.append(data['edge_density_reduction'] * 100)
        contour_reductions.append(data['contour_area_reduction'] * 100)
        successes.append(data['attack_success'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Edge reduction chart
    colors = ['green' if s else 'red' for s in successes]
    bars1 = ax1.bar(range(len(attacks)), edge_reductions, color=colors, alpha=0.7)
    ax1.set_title('Edge Density Reduction by Attack', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Edge Reduction (%)')
    ax1.set_xticks(range(len(attacks)))
    ax1.set_xticklabels(attacks, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, edge_reductions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

    # Contour reduction chart
    bars2 = ax2.bar(range(len(attacks)), contour_reductions, color=colors, alpha=0.7)
    ax2.set_title('Contour Area Reduction by Attack', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Contour Reduction (%)')
    ax2.set_xticks(range(len(attacks)))
    ax2.set_xticklabels(attacks, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, contour_reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    output_path = f"{RESULTS_DIR}/effectiveness_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved effectiveness chart: {output_path}")
    plt.show()

def print_summary_stats():
    """Print summary statistics"""
    results = load_results()

    total_attacks = len(results)
    successful = sum(1 for r in results.values() if r['attack_success'])

    print(f"\n{'='*50}")
    print(f"ATTACK SUMMARY STATISTICS")
    print(f"{'='*50}")
    print(f"Total attacks: {total_attacks}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/total_attacks:.1%}")
    print()

    # Best performing attacks
    print("MOST EFFECTIVE ATTACKS:")
    print("-" * 30)

    by_edge_reduction = sorted(results.items(),
                              key=lambda x: x[1]['edge_density_reduction'],
                              reverse=True)[:3]

    for i, (name, data) in enumerate(by_edge_reduction, 1):
        print(f"{i}. {name}")
        print(f"   Edge reduction: {data['edge_density_reduction']:.1%}")
        print(f"   Contour reduction: {data['contour_area_reduction']:.1%}")
        print()

    print("\nNOTE: Negative edge reduction means the attack created MORE edges")
    print("This is still successful because it corrupts the original edge structure!")

def main():
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Run better_attacks.py first to generate results.")
        return

    print("Creating attack visualizations...")

    # Print summary statistics
    print_summary_stats()

    # Create comprehensive summary for each source image
    create_summary_visualization()

    # Create effectiveness chart
    create_effectiveness_chart()

    print(f"\nAll visualizations saved to {RESULTS_DIR}/")
    print("\nCheck the generated images to see your attack results!")

if __name__ == "__main__":
    main()
