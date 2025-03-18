#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import re
import json
from pathlib import Path
import collections.abc

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# ================ DEFAULT APPEARANCE CONFIGURATION VARIABLES ================
# These will be used if not specified in the config file
DEFAULT_STYLE = {
    "font_sizes": {
        "base": 18,
        "axes_label": 20,
        "title": 22,
        "tick_label": 18,
        "legend": 18
    },
    "figure": {
        "width": 12,
        "height": 8,
        "dpi": 300,
        "use_y_label": True,
        "use_legend": True,
        "ticks_on_right": False,
        "log_scale": False
    },
    "markers": {
        "average_size": 2000,
        "individual_size": 90,
        "average_line_width": 3,
        "individual_line_width": 1.5,
        "reference_line_width": 2.0
    },
    "layout": {
        "jitter_factor": 50,
        "padding": 2.0,
        "marker_alpha": 0.7,
        "reference_line_alpha": 0.65,
        "grid_line_width": 0.5,
        "grid_alpha": 0.7,
        "legend_y_offset": 0,
        "legend_position": "upper right"
    }
}
# ================ END DEFAULT CONFIGURATION ================

# Check that the script was called with at least three arguments:
#   1. Config key (e.g., "gol")
#   2. Path to experiments CSV
#   3. Path to references CSV
if len(sys.argv) < 4:
    print("Usage: python3 print-graph.py <config_keys> <path_to_experiments> <path_to_references> [tag]")
    sys.exit(1)

# Get command-line arguments.
config_key = sys.argv[1]
experiments_path = sys.argv[2]
references_path = sys.argv[3]
tag = sys.argv[4] if len(sys.argv) > 4 else ""

# Load the JSON configuration file.
script_dir = Path(__file__).parent
config_filename = script_dir / "print-graph.config.json"
if not os.path.exists(config_filename):
    print(f"Error: Config file '{config_filename}' not found.")
    sys.exit(1)

with open(config_filename, "r") as f:
    config_all = json.load(f)

config_keys = config_key.split("@")

# Get the configuration for the specified keys. The latter keys should "patch" the preceding ones.

config = {}
for key in config_keys:
    if (key not in config_all):
        print(f"Error: Config key '{key}' not found in the config file.")
        sys.exit(1)

    config = update(config, config_all[key])

# Get style configuration (or use defaults)
style_config = config.get("style", {})

# Extract style parameters with fallbacks to defaults
font_sizes = style_config.get("font_sizes", DEFAULT_STYLE["font_sizes"])
figure_config = style_config.get("figure", DEFAULT_STYLE["figure"])
markers_config = style_config.get("markers", DEFAULT_STYLE["markers"])
layout_config = style_config.get("layout", DEFAULT_STYLE["layout"])

# Set Use TeX to False to use the default font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.weight": "bold",
    "font.size": font_sizes.get("base", DEFAULT_STYLE["font_sizes"]["base"]),
    "axes.labelsize": font_sizes.get("axes_label", DEFAULT_STYLE["font_sizes"]["axes_label"]),
    "axes.titlesize": font_sizes.get("title", DEFAULT_STYLE["font_sizes"]["title"]),
    "xtick.labelsize": font_sizes.get("tick_label", DEFAULT_STYLE["font_sizes"]["tick_label"]),
    "ytick.labelsize": font_sizes.get("tick_label", DEFAULT_STYLE["font_sizes"]["tick_label"]),
    "legend.fontsize": font_sizes.get("legend", DEFAULT_STYLE["font_sizes"]["legend"]),
})

# Check that experiments and references files exist.
if not os.path.exists(experiments_path):
    print(f"Error: Experiments file not found at {experiments_path}")
    sys.exit(1)
if not os.path.exists(references_path):
    print(f"Error: References file not found at {references_path}")
    sys.exit(1)

# Load the CSV files.
df_experiments = pd.read_csv(experiments_path, delimiter=";")
df_references = pd.read_csv(references_path, delimiter=";")

# Sort the dataframes by time in descending order.
df_references = df_references.sort_values(by='time', ascending=False)

# Filter out invalid records (must be verified and compiled).
df_experiments = df_experiments[(df_experiments['verified'] == True) & (df_experiments['compiled'] == True)]
df_references = df_references[(df_references['verified'] == True) & (df_references['compiled'] == True)]

# Extract solution class from experiment_id (e.g., "game_of_life01" from "game_of_life01-01/gol").
def extract_solution_class(experiment_id):
    match = re.match(r'((histogram|game_of_life)\d+(?:_[a-zA-Z]+)?)', experiment_id)
    if match:
        return match.group(1)
    return experiment_id

df_experiments['solution_class'] = df_experiments['experiment_id'].apply(extract_solution_class)
df_references['solution_class'] = df_references['experiment_id'].apply(extract_solution_class)

# Extract solution number from experiment_id (e.g., "01" from "game_of_life01-01/gol").
def extract_solution_number(experiment_id):
    match = re.search(r'-(\d+)/', experiment_id)
    if match:
        return int(match.group(1))
    return 0

df_experiments['solution_number'] = df_experiments['experiment_id'].apply(extract_solution_number)
df_references['solution_number'] = df_references['experiment_id'].apply(extract_solution_number)

# Filter references based on the config "included_references_ids" list.
included_refs = config.get("included_references_ids", [])
if included_refs:
    df_references = df_references[df_references['experiment_id'].isin(included_refs)]

# Transform individual times using the transformer function from the config.
# The transform string (e.g., "time / 16384 / 200") is evaluated for each time.
transform_expr = config.get("time_transform", "time")
def transform_time(t):
    # 'time' in the expression is replaced by t.
    return eval(transform_expr, {"time": t})
df_experiments["time"] = df_experiments["time"].apply(transform_time)
df_references["time"] = df_references["time"].apply(transform_time)

if df_experiments.empty:
    print("No valid experimental data found after filtering. Exiting.")
    sys.exit(1)

# Prepare the plot.
plt.figure(figsize=(
    figure_config.get("width", DEFAULT_STYLE["figure"]["width"]),
    figure_config.get("height", DEFAULT_STYLE["figure"]["height"])
), dpi=figure_config.get("dpi", DEFAULT_STYLE["figure"]["dpi"]))

# Get unique solution classes (sorted) from experiments.
all_solution_classes = sorted(set(df_experiments['solution_class'].unique()))

# Map solution classes to labels using the config.
class_labels_map = config.get("class_ids_to_labels", {})
# Use the mapped label if available; otherwise fall back to the solution class itself.
all_solution_labels = [class_labels_map.get(cls, cls) for cls in all_solution_classes]

# Define colors and markers.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
reference_colors = ['orange', 'green', 'magenta', '#008DFF', 'red']
reference_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))]

# Create x-position mapping for solution classes.
x_positions = {cls: i for i, cls in enumerate(all_solution_classes)}

# Plot experimental data points.
for i, (sol_class, group) in enumerate(df_experiments.groupby('solution_class')):
    color = colors[i % len(colors)]
    x_pos = x_positions[sol_class]
    n=0
    # Add jitter to x position to avoid overlap.
    jitter_factor = layout_config.get("jitter_factor", DEFAULT_STYLE["layout"]["jitter_factor"])
    jitter = np.linspace(-len(group)/jitter_factor, len(group)/jitter_factor, len(group))
    np.random.shuffle(jitter)  # Shuffle the jitter to randomize the spread.

    average_time = group['time'].median()

    plt.scatter(x_pos,
                average_time,
                color=color,
                marker='_',
                s=markers_config.get("average_size", DEFAULT_STYLE["markers"]["average_size"]),
                alpha=1.0,
                # edgecolors='black',
                linewidth=markers_config.get("average_line_width", DEFAULT_STYLE["markers"]["average_line_width"]),
                zorder=2)

    for _, row in group.iterrows():
        plt.scatter(x_pos + jitter[n],
                    row['time'],
                    color=color,
                    marker='o',
                    s=markers_config.get("individual_size", DEFAULT_STYLE["markers"]["individual_size"]),
                    alpha=layout_config.get("marker_alpha", DEFAULT_STYLE["layout"]["marker_alpha"]),
                    edgecolors='black',
                    linewidth=markers_config.get("individual_line_width", DEFAULT_STYLE["markers"]["individual_line_width"]),
                    zorder=3)
        n += 1

# Set x-axis ticks and labels using the mapped labels.
plt.xticks(list(x_positions.values()), [class_labels_map.get(cls, cls) for cls in all_solution_classes])

# Set y-axis limits from config.
y_limits = config.get("y_axis_from_to", None)
if y_limits:
    plt.ylim(y_limits)

# Set y-axis to logarithmic scale if configured
use_log_scale = figure_config.get("log_scale", DEFAULT_STYLE["figure"]["log_scale"])
if use_log_scale:
    plt.yscale('log')

# Plot reference data as horizontal lines across the plot area.
ax = plt.gca()
ref_lines = []  # For storing line objects.
ref_labels = []  # For storing labels.
for i, (_, row) in enumerate(df_references.iterrows()):
    y_val = row['time']
    color_idx = i % len(reference_colors)
    style_idx = i % len(reference_styles)
    x_min, x_max = ax.get_xlim()
    line = plt.axhline(y=y_val,
                       color=reference_colors[color_idx],
                       linestyle=reference_styles[style_idx],
                       linewidth=markers_config.get("reference_line_width", DEFAULT_STYLE["markers"]["reference_line_width"]),
                       alpha=layout_config.get("reference_line_alpha", DEFAULT_STYLE["layout"]["reference_line_alpha"]),
                       zorder=1)
    # Use the reference label from the config if available.
    ref_label_map = config.get("references_labels", {})
    ref_key = row['experiment_id']
    label = ref_label_map.get(ref_key, f"Ref {row['solution_number']} ({row['solution_class']}): {y_val:.1f}")
    ref_lines.append(line)
    ref_labels.append(label)

# Configure y-axis position and labels
use_y_label = figure_config.get("use_y_label", DEFAULT_STYLE["figure"]["use_y_label"])
ticks_on_right = figure_config.get("ticks_on_right", DEFAULT_STYLE["figure"]["ticks_on_right"])

# Handle y-tick position (left or right)
if ticks_on_right:
    # Move ticks to right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

if use_y_label:
    plt.ylabel(config.get("y_label", "Time"))
else:
    plt.ylabel(config.get("y_label", "Time"), color=(1, 1, 1, 0))

plt.grid(True, which='both', linestyle='--',
         linewidth=layout_config.get("grid_line_width", DEFAULT_STYLE["layout"]["grid_line_width"]),
         alpha=layout_config.get("grid_alpha", DEFAULT_STYLE["layout"]["grid_alpha"]))

# Add legend only if configured to do so
use_legend = figure_config.get("use_legend", DEFAULT_STYLE["figure"]["use_legend"])
if use_legend:
    # Add legend for reference lines
    for line, label in zip(ref_lines, ref_labels):
        line.set_label(label)

    # Get legend y offset value (0 means no offset)
    # legend_y_offset = layout_config.get("legend_y_offset", DEFAULT_STYLE["layout"]["legend_y_offset"])

    position = layout_config.get("legend_position", DEFAULT_STYLE["layout"]["legend_position"])

    # Position the legend with optional y-axis offset (1 is the top, lower numbers move it down)
    plt.legend(loc=position,
            #    bbox_to_anchor=(1, 1-legend_y_offset),
               fontsize=font_sizes.get("legend", DEFAULT_STYLE["font_sizes"]["legend"]))

plt.tight_layout(pad=layout_config.get("padding", DEFAULT_STYLE["layout"]["padding"]))
plt.subplots_adjust(right=0.85)

# Generate output filename from the experiment file path
output_filename = tag + os.path.basename(experiments_path).replace(".csv", ".graph.pdf")
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Plot saved as {output_filename}")
