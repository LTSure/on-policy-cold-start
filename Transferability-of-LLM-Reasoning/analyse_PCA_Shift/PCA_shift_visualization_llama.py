import os  # for file path operations
import json  # for reading JSON files
import pandas as pd  # for DataFrame creation and manipulation
import matplotlib as mpl  # for Matplotlib configuration
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for numerical operations
import argparse  # for command line arguments

# ---------- Global plot style settings ----------
mpl.rcParams.update({
    # Use serif font family for all text
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    # Base font size
    "font.size": 18,
    # Title and label sizes
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    # Line and marker styles
    "lines.linewidth": 2,
    "lines.markersize": 8,
    # Axis spine and grid settings
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    # Legend styling
    "legend.frameon": True,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,
    # Cycle through specific colors for plot lines and markers
    "axes.prop_cycle": mpl.cycler("color", ["#0989D3", "#E6593A", "#009E73"]),
    "legend.edgecolor": "#0A0A0A"
})

# ---------- Input file configurations ----------
# Parse command line arguments
parser = argparse.ArgumentParser(description="Visualize PCA shift analysis across tasks and models")
parser.add_argument('--tasks', nargs='+', default=["MATH500", "GSM8K", "Livecodebench"],
                    help="List of tasks to visualize (default: MATH500 GSM8K Livecodebench)")
parser.add_argument('--base_dir', type=str, 
                    default="/cpfs04/user/liutianshuo/math/simpleRL-reason/Transferability-of-LLM-Reasoning/analyse_PCA_Shift",
                    help="Base directory for PCA shift results")
parser.add_argument('--output_path', type=str, default="PCA_Shift_Llama_3.2_3B_4_tasks.pdf",
                    help="Output path for the visualization PDF")
args = parser.parse_args()

# Base directory for PCA shift results
base_dir = args.base_dir

# Task directories
tasks = args.tasks

# Model configurations
models = [
    {
        "name": "Llama-3.2-3B-On-Policy",
        "file_suffix": "Llama-3.2-3B-ppo-sft_alfworld_pca_shift.json",
        "state_map": {"Original": "Base", "Updated": "On Policy"},
    },
    {
        "name": "Llama-3.2-3B-SFT", 
        "file_suffix": "Llama-3.2-3B-sft_alfworld_pca_shift.json",
        "state_map": {"Original": "Base", "Updated": "SFT"},
    }
]

# ---------- Constants ----------
# Marker shape for each state
markers = {"Original": "o", "Updated": "^"}
# Z-order to control overlay of points
zorder_order = {"Original": 1, "Updated": 2}
# List of benchmarks to plot (column order) - now handled by tasks list
# target_benchmarks = [
#     "MATH500_PCA_Shift",
#     "AIME25_PCA_Shift", 
#     "GSM8K_PCA_Shift",
#     "Livecodebench_PCA_Shift"
# ]
# Background colors for each row (first two rows)
row_colors = {
    0: "#F3F8FF",  # light blue for first row
    1: "#FFF6EB",  # light orange for second row
}

# ---------- Utility functions ----------
def build_df(path: str) -> pd.DataFrame:
    """
    Load JSON file and construct a DataFrame with columns: benchmark, layer, state, shift, principle.
    """
    with open(path, "r") as f:
        content = json.load(f)
    rows = []
    # Iterate through each benchmark block
    for block in content:
        bench = block["benchmark"]
        for d in block["data"]:
            # Add benchmark name to each row
            rows.append({"benchmark": bench, **d})
    df = pd.DataFrame(rows)
    # Standardize state naming
    df["state"] = df["state"].replace({"step1": "Updated"})
    return df


def calculate_dynamic_limits(df, padding_factor=0.15):
    """
    Calculate dynamic axis limits based on data ranges with padding.
    """
    x_min, x_max = df["shift"].min(), df["shift"].max()
    y_min, y_max = df["principle"].min(), df["principle"].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    xlim = (x_min - x_range * padding_factor, x_max + x_range * padding_factor)
    ylim = (y_min - y_range * padding_factor, y_max + y_range * padding_factor)
    
    return xlim, ylim

def calculate_column_limits(all_data_for_column, padding_factor=0.15):
    """
    Calculate consistent axis limits for all subplots in a column.
    """
    all_shifts = []
    all_principles = []
    
    for data in all_data_for_column:
        if len(data) > 0:
            all_shifts.extend(data["shift"].tolist())
            all_principles.extend(data["principle"].tolist())
    
    if not all_shifts:
        return None, None
    
    x_min, x_max = min(all_shifts), max(all_shifts)
    y_min, y_max = min(all_principles), max(all_principles)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    xlim = (x_min - x_range * padding_factor, x_max + x_range * padding_factor)
    ylim = (y_min - y_range * padding_factor, y_max + y_range * padding_factor)
    
    return xlim, ylim


def plot_subplot(ax, df, state_map, xlim, ylim):
    """
    Plot lines and points on a given Axes:
    - Gray lines connect original vs updated points per layer
    - Colored markers show original and updated positions
    """
    # Plot gray connecting lines for each layer
    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer].sort_values("state")
        ax.plot(
            sub["shift"], sub["principle"],
            color="gray", linewidth=1, alpha=0.5, zorder=1
        )
    # Scatter points for each state
    for state in df["state"].unique():
        if state not in markers:
            continue
        sub = df[df["state"] == state]
        ax.scatter(
            sub["shift"], sub["principle"],
            marker=markers[state],
            label=state_map[state],
            alpha=0.65,
            zorder=zorder_order[state]
        )
    # Apply axis limits and styling
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(direction="out")

# ---------- Create combined figure ----------
# Create a 2-row, dynamic-column grid of subplots based on number of tasks
num_tasks = len(tasks)
fig, axs = plt.subplots(2, num_tasks, figsize=(4*num_tasks, 8))
if num_tasks == 1:
    axs = axs.reshape(2, 1)  # Ensure 2D array for single task
plt.subplots_adjust(wspace=0.35, hspace=0.4)
# Main title above the grid
fig.suptitle("PCA Shift across Models and Tasks", fontsize=24, y=1.05)

# First pass: collect all data for each column to calculate consistent limits
column_data = {}
for col, task in enumerate(tasks):
    column_data[col] = []
    for row, model in enumerate(models):
        file_path = f"{base_dir}/{task}/{model['file_suffix']}"
        if os.path.exists(file_path):
            benchmark = f"{task}_PCA_Shift"
            data = build_df(file_path)
            data = data[data["benchmark"] == benchmark]
            column_data[col].append(data)
        else:
            column_data[col].append(pd.DataFrame())  # Empty DataFrame for missing data

# Calculate consistent axis limits for each column
column_limits = {}
for col in range(len(tasks)):
    xlim, ylim = calculate_column_limits(column_data[col])
    column_limits[col] = (xlim, ylim)

# Populate each subplot with consistent limits per column
for col, task in enumerate(tasks):
    for row, model in enumerate(models):
        # Construct file path for this task and model
        file_path = f"{base_dir}/{task}/{model['file_suffix']}"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        # Load data and filter for current benchmark
        benchmark = f"{task}_PCA_Shift"
        data = build_df(file_path)
        data = data[data["benchmark"] == benchmark]
        
        ax = axs[row, col]
        # Set row-specific background color
        ax.set_facecolor(row_colors[row])
        
        # Use consistent limits for this column
        xlim, ylim = column_limits[col]
        plot_subplot(ax, data, model["state_map"], xlim, ylim)
        
        # Add column title on first row
        if row == 0:
            ax.set_title(task, pad=6, fontsize=16)
        # X axis label on second row
        if row == 1:
            ax.set_xlabel("(PC1 Δ)")
        # Y axis label on first column
        if col == 0:
            ax.set_ylabel("(PC2)")
            # Add legend only on first column
            leg = ax.legend(loc="upper center", frameon=True, fancybox=True, fontsize=14)
            leg.get_frame().set_edgecolor("#0A0A0A")
            leg.get_frame().set_linewidth(1)

# ---------- Add row titles above each row ----------
row_titles = [
    "Llama-3.2-3B-On-Policy; Base: Llama-3.2-3B",
    "Llama-3.2-3B-SFT; Base: Llama-3.2-3B"
]
for row, row_title in enumerate(row_titles):
    row_axes = axs[row, :]
    # Compute horizontal span of the row
    pos_left = row_axes[0].get_position().x0
    pos_right = row_axes[-1].get_position().x1
    pos_y = row_axes[0].get_position().y1 + 0.04  # a bit above the subplots
    pos_x = (pos_left + pos_right) / 2
    # Place centered text as row title
    fig.text(
        pos_x, pos_y, row_title,
        ha='center', va='bottom', fontsize=18, fontweight='bold'
    )

# ---------- Add category annotations above columns ----------
# Label columns based on task type
for col, task in enumerate(tasks):
    if task == "Livecodebench":
        category = "Code Generation"
    elif task in ["MATH500", "GSM8K"]:
        category = "Math"
    else:
        category = "Other"
    axs[0, col].annotate(
        category, xy=(0.5, 1.25), xycoords="axes fraction",
        fontsize=20, ha="center"
    )

# ---------- Save figure to PDF ----------
fig.savefig(
    args.output_path,
    format="pdf", dpi=300, bbox_inches="tight"
)
plt.close()  # Close the figure to free memory 