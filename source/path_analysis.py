import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from typing import Optional, List


def plot_tool_path_data(file_path: str, cutoff: int = None) -> None:
    """Reads the CSV data and plots the x, y, z components for each round."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    plt.figure(figsize=(12, 8))

    for idx, row in data.iterrows():
        x_vals = row[::3]
        y_vals = row[1::3]
        z_vals = row[2::3]

        if cutoff is not None:
            x_vals = x_vals[:cutoff]
            y_vals = y_vals[:cutoff]
            z_vals = z_vals[:cutoff]

        plt.plot(x_vals, label=f"Round {idx+1} - X")
        plt.plot(y_vals, label=f"Round {idx+1} - Y")
        plt.plot(z_vals, label=f"Round {idx+1} - Z")

    plt.xlabel("Sample [n]")
    plt.ylabel("Tool Point Coordinate [units]")
    plt.title("Tool Path Data Visualization (X, Y, Z)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_single_tool_path_data(file_path: str, run_number: int = 2, cutoff: int = None) -> None:
    """Plots a single run (e.g., run number 2) of the tool path data in X, Y, Z components."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    if run_number < 1 or run_number > len(data):
        print(f"Invalid run number: {run_number}")
        return

    row = data.iloc[run_number - 1]  # Convert to zero-based index

    x_vals = row[::3]
    y_vals = row[1::3]
    z_vals = row[2::3]

    if cutoff is not None:
        x_vals = x_vals[:cutoff]
        y_vals = y_vals[:cutoff]
        z_vals = z_vals[:cutoff]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, label="X axis")
    plt.plot(y_vals, label="Y axis")
    plt.plot(z_vals, label="Z axis")
    plt.xlabel("Sample [n]")
    plt.ylabel("Tool Point Coordinate [units]")
    plt.title(f"Tool Path Data Visualization for Round {run_number}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tool_path_data_by_axis_color(file_path: str, cutoff: int = None) -> None:
    """Plots all tool paths with X, Y, Z components colored consistently across runs."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    plt.figure(figsize=(12, 8))

    # Predefined consistent colors
    color_map = {
        'X': 'blue',
        'Y': 'orange',
        'Z': 'green'
    }

    for idx, row in data.iloc[1:].iterrows():
        x_vals = row[::3]
        y_vals = row[1::3]
        z_vals = row[2::3]

        if cutoff is not None:
            x_vals = x_vals[:cutoff]
            y_vals = y_vals[:cutoff]
            z_vals = z_vals[:cutoff]

        # Plot with consistent colors across runs
        plt.plot(x_vals, color=color_map['X'], alpha=0.3)
        plt.plot(y_vals, color=color_map['Y'], alpha=0.3)
        plt.plot(z_vals, color=color_map['Z'], alpha=0.3)

    # Create dummy lines for the legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color=color_map['X'], lw=2, label='X Axis'),
        Line2D([0], [0], color=color_map['Y'], lw=2, label='Y Axis'),
        Line2D([0], [0], color=color_map['Z'], lw=2, label='Z Axis')
    ]

    plt.xlabel("Sample [n]")
    plt.ylabel("Tool Point Coordinate [units]")
    plt.title("Tool Path Data Visualization (Colored by Axis)")
    plt.legend(handles=legend_lines)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_3d_tool_path_data(file_path: str, cutoff: int = None) -> None:
    """Reads the CSV data and plots the path in 3D space, marking start and end points."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.colormaps['rainbow']
    num_lines = len(data)

    for idx, row in data.iloc[1:].iterrows():  # Skipping first row
        x_vals = row[::3].values
        y_vals = row[1::3].values
        z_vals = row[2::3].values

        if cutoff is not None:
            x_vals = x_vals[:cutoff]
            y_vals = y_vals[:cutoff]
            z_vals = z_vals[:cutoff]

        color = cmap(idx / num_lines)
        ax.plot(x_vals, y_vals, z_vals, color='grey', alpha=0.6)

        # Mark starting point
        ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='green', marker='o', s=50)
        # Mark ending point
        ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='red', marker='X', s=50)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Tool Path'),
        Line2D([0], [0], marker='o', color='w', label='Start Point', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='End Point', markerfacecolor='red', markersize=10)
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel("X [units]")
    ax.set_ylabel("Y [units]")
    ax.set_zlabel("Z [units]")
    ax.set_title("3D Tool Path Data Visualization with Start/End Points")
    plt.tight_layout()
    plt.show()

def plot_all_3d_paths(folder_name: str,
                      cutoff: Optional[int]   = None,
                      run_index: Optional[int] = None,
                      custom_steps:  Optional[List[int]] = None,
                      custom_marker: str      = 'D',
                      custom_color:  str      = 'black',
                      custom_text:   str      = 'Custom marker') -> None:
    """
    Plots 3D paths for tooltip, pin, pipe, and center from a folder of CSVs.
    If run_index is specified, only that run will be plotted for each object.
    Optionally mark custom steps along each trajectory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path    = os.path.join(project_root, "data", folder_name)

    file_map = {
        "tooltip_positions.csv": "Tooltip",
        "pin_positions.csv":     "Pin",
        "pipe_positions.csv":    "Pipe",
        "center_positions.csv":  "Center"
    }
    colors = {"Tooltip":'gray',"Pin":'blue',"Pipe":'purple',"Center":'orange'}

    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(111, projection='3d')

    # plot each trajectory
    for filename, label in file_map.items():
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        data = pd.read_csv(file_path, header=None)
        rows = ([run_index+1] if run_index is not None
                else range(1, len(data)))  # skip header row

        for idx in rows:
            row    = data.iloc[idx]
            x_vals = row[::3].to_numpy()
            y_vals = row[1::3].to_numpy()
            z_vals = row[2::3].to_numpy()

            if cutoff is not None:
                x_vals, y_vals, z_vals = x_vals[:cutoff], y_vals[:cutoff], z_vals[:cutoff]

            # main line
            ax.plot(x_vals, y_vals, z_vals,
                    color=colors[label], alpha=0.6,
                    label=(f"{label} Path" if run_index is None and idx==1 else None))

            # start/end
            ax.scatter(x_vals[0], y_vals[0], z_vals[0],
                       color='green', marker='o', s=50)
            ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1],
                       color='red',   marker='X', s=50)

            # custom steps
            if custom_steps:
                for step in custom_steps:
                    if 0 <= step < len(x_vals):
                        ax.scatter(x_vals[step], y_vals[step], z_vals[step],
                                   color=custom_color,
                                   marker=custom_marker,
                                   s=60,
                                   label=( "Custom Point" if step==custom_steps[0] and idx==rows[0] else None)
                                  )

    # build legend once
    legend_elements = [
        Line2D([0],[0], color='gray',  lw=2, label='Tooltip Path'),
        Line2D([0],[0], color='blue',  lw=2, label='Pin Path'),
        Line2D([0],[0], color='purple',lw=2, label='Pipe Path'),
        Line2D([0],[0], color='orange',lw=2, label='Center Path'),
        Line2D([0],[0], marker='o',   color='w', label='Start Point',
               markerfacecolor='green',markersize=10),
        Line2D([0],[0], marker='X',   color='w', label='End Point',
               markerfacecolor='red',  markersize=10),
        Line2D([0],[0], marker=custom_marker,
               color='w', label=custom_text,
               markerfacecolor=custom_color,
               markersize=10)
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel("X [units]")
    ax.set_ylabel("Y [units]")
    ax.set_zlabel("Z [units]")
    title = f"3D Object Paths from data collected {folder_name}"
    #title = f"3D Object Paths"
    if run_index is not None:
        title += f" — Run {run_index+1}"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    ##Data analysis
    #plot_tool_path_data("../data/tooltip_positions_4.2.csv", 200)
    #plot_single_tool_path_data("../data/tooltip_positions_4.2.csv",3, 200)
    #plot_tool_path_data_by_axis_color("../data/tooltip_positions_4.2.csv", 200)
    #plot_3d_tool_path_data("../data/tooltip_positions_4.2.csv", 200)
    #plot_3d_tool_path_data("../data/2025-08-04_11-53-19/tooltip_positions.csv", 200)
    ##Syntetic data
    #plot_tool_path_data("../data/tool_path_data.csv")
    #plot_3d_tool_path_data("../data/tool_path_data.csv")

    ##Mega data(pint)
    plot_single_tool_path_data("../data/2025-08-06_10-59-00/pin_positions.csv", 10, 200)
    plot_all_3d_paths(
        folder_name="2025-08-06_10-59-00",
        cutoff=200,
        custom_steps=[42],
        custom_marker='D',
        custom_color='black',
        custom_text='Contact point'
    )


