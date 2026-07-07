"""
Analysis and visualization script for token probability and gradient distribution.
Groups tokens by probability intervals and computes gradient statistics for each interval.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_prob_gradient_distribution(
    prob_gradient_pairs: List[Tuple[float, float]],
    num_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze gradient distribution across probability intervals.
    
    Args:
        prob_gradient_pairs: List of (probability, gradient) tuples
        num_bins: Number of probability bins (default: 10 for [0, 0.1], [0.1, 0.2], ..., [0.9, 1.0])
        
    Returns:
        Dictionary with statistics for each probability interval
    """
    # Group by probability intervals
    bins = defaultdict(list)
    
    for prob, grad in prob_gradient_pairs:
        # Determine which bin this probability belongs to
        bin_idx = min(int(prob * num_bins), num_bins - 1)
        bins[bin_idx].append(grad)
    
    # Compute statistics for each bin
    results = {}
    for bin_idx in range(num_bins):
        bin_gradients = bins[bin_idx]
        if len(bin_gradients) > 0:
            grad_array = np.array(bin_gradients)
            results[f"[{bin_idx/num_bins:.1f}, {(bin_idx+1)/num_bins:.1f})"] = {
                "count": len(bin_gradients),
                "mean": float(np.mean(grad_array)),
                "std": float(np.std(grad_array)),
                "min": float(np.min(grad_array)),
                "max": float(np.max(grad_array)),
            }
        else:
            results[f"[{bin_idx/num_bins:.1f}, {(bin_idx+1)/num_bins:.1f})"] = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
    
    return results


def visualize_prob_gradient_distribution(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Token Probability vs Gradient Distribution",
):
    """
    Visualize the probability-gradient distribution analysis.
    
    Args:
        results: Dictionary with statistics for each probability interval
        save_path: Path to save the figure
        title: Title for the plot
    """
    # Extract data for plotting
    bin_labels = list(results.keys())
    means = [results[bin_label]["mean"] for bin_label in bin_labels]
    stds = [results[bin_label]["std"] for bin_label in bin_labels]
    counts = [results[bin_label]["count"] for bin_label in bin_labels]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Mean gradient with error bars (std)
    ax1 = axes[0]
    x_pos = np.arange(len(bin_labels))
    ax1.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
        label="Mean ± Std",
    )
    ax1.set_xlabel("Probability Interval", fontsize=12)
    ax1.set_ylabel("Gradient (Mean ± Std)", fontsize=12)
    ax1.set_title(f"{title} - Mean and Standard Deviation", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Count of tokens in each bin
    ax2 = axes[1]
    ax2.bar(x_pos, counts, alpha=0.7, color="steelblue", edgecolor="black")
    ax2.set_xlabel("Probability Interval", fontsize=12)
    ax2.set_ylabel("Number of Tokens", fontsize=12)
    ax2.set_title("Token Count per Probability Interval", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        if count > 0:
            ax2.text(i, count, str(count), ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")
    plt.close()


def save_analysis_results_txt(
    results: Dict[str, Dict[str, float]],
    save_path: str,
):
    """
    Save analysis results to TXT file.
    
    Args:
        results: Dictionary with statistics for each probability interval
        save_path: Path to save the TXT file
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Probability Interval\tCount\tMean\tStd\tMin\tMax\n")
        for bin_label, stats in results.items():
            f.write(f"{bin_label}\t{stats['count']}\t{stats['mean']:.6f}\t{stats['std']:.6f}\t{stats['min']:.6f}\t{stats['max']:.6f}\n")
    print(f"Analysis results saved to: {save_path}")


def load_prob_gradient_data(log_file_path: str) -> List[Tuple[float, float]]:
    """
    Load probability-gradient pairs from log file.
    
    Expected format: Each line contains probability and gradient values
    Format: "prob=0.123,grad=0.456" or JSON format
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        List of (probability, gradient) tuples
    """
    prob_gradient_pairs = []
    
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found: {log_file_path}")
        return prob_gradient_pairs
    
    with open(log_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse different formats
            try:
                # Format: {"prob": 0.123, "grad": 0.456}
                if line.startswith("{"):
                    data = json.loads(line)
                    prob = float(data.get("prob", 0.0))
                    grad = float(data.get("grad", 0.0))
                    prob_gradient_pairs.append((prob, grad))
                # Format: prob=0.123,grad=0.456
                elif "prob=" in line and "grad=" in line:
                    parts = line.split(",")
                    prob = float(parts[0].split("=")[1])
                    grad = float(parts[1].split("=")[1])
                    prob_gradient_pairs.append((prob, grad))
            except (ValueError, KeyError, IndexError) as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue
    
    return prob_gradient_pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze token probability and gradient distribution")
    parser.add_argument("--log_file", type=str, required=True, help="Path to log file with prob-gradient pairs")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Output directory for results")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of probability bins")
    parser.add_argument("--title", type=str, default="Token Probability vs Gradient Distribution", help="Plot title")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.log_file}")
    prob_gradient_pairs = load_prob_gradient_data(args.log_file)
    print(f"Loaded {len(prob_gradient_pairs)} probability-gradient pairs")
    
    if len(prob_gradient_pairs) == 0:
        print("No data found. Exiting.")
        exit(1)
    
    # Analyze
    print("Analyzing distribution...")
    results = analyze_prob_gradient_distribution(prob_gradient_pairs, num_bins=args.num_bins)
    
    # Print results
    print("\n" + "=" * 60)
    print("Analysis Results:")
    print("=" * 60)
    for bin_label, stats in results.items():
        print(f"\n{bin_label}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean:  {stats['mean']:.6f}")
        print(f"  Std:   {stats['std']:.6f}")
        print(f"  Min:   {stats['min']:.6f}")
        print(f"  Max:   {stats['max']:.6f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save TXT file
    txt_path = os.path.join(args.output_dir, "prob_gradient_data.txt")
    save_analysis_results_txt(results, txt_path)
    
    # Save PNG figure
    fig_path = os.path.join(args.output_dir, "prob_gradient_distribution.png")
    visualize_prob_gradient_distribution(results, fig_path, title=args.title)
    
    print(f"\nAnalysis complete!")

