import os
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D


def plot_all_3d_paths(folder_name: str,
                      mesh_map: dict[str, str],
                      cutoff: int = None,
                      run_index: int = None) -> None:
    """
    Plots 3D paths for tooltip, pin, pipe, and center from a folder of CSVs,
    and overlays each object’s CAD mesh at its start position.

    mesh_map: { "Tooltip": "/path/to/tooltip.stl", ... }
    """
    # locate data folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(project_root, "data", folder_name)

    file_map = {
        "tooltip_positions.csv": "Tooltip",
        "pin_positions.csv": "Pin",
        "pipe_positions.csv": "Pipe",
        "center_positions.csv": "Center"
    }
    colors = {"Tooltip": 'gray', "Pin": 'blue', "Pipe": 'purple', "Center": 'orange'}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # First pass: plot meshes at start positions
    for filename, label in file_map.items():
        mesh_path = mesh_map.get(label)
        if not mesh_path or not os.path.exists(mesh_path):
            continue

        # load one CSV just to read the start point of run_index (or first run)
        df = pd.read_csv(os.path.join(base_path, filename), header=None)
        row_idx = (run_index + 1) if run_index is not None else 1
        start_row = df.iloc[row_idx]
        x0, y0, z0 = start_row[::3].iloc[0], start_row[1::3].iloc[0], start_row[2::3].iloc[0]

        mesh = trimesh.load(mesh_path)
        mesh.apply_translation([x0, y0, z0])
        faces = mesh.vertices[mesh.faces]
        poly = Poly3DCollection(faces,
                                facecolor='lightgray',
                                edgecolor='k',
                                alpha=0.3)
        ax.add_collection3d(poly)
        # optional label at that point
        ax.text(x0, y0, z0, label, color='k')

    # Second pass: plot trajectories
    for filename, label in file_map.items():
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        data = pd.read_csv(file_path, header=None)
        rows = ([run_index + 1] if run_index is not None
                else range(1, len(data)))

        for idx in rows:
            row = data.iloc[idx]
            x_vals = row[::3].to_numpy()
            y_vals = row[1::3].to_numpy()
            z_vals = row[2::3].to_numpy()

            if cutoff is not None:
                x_vals, y_vals, z_vals = (x_vals[:cutoff],
                                          y_vals[:cutoff],
                                          z_vals[:cutoff])

            ax.plot(x_vals, y_vals, z_vals,
                    color=colors[label], alpha=0.6,
                    label=f"{label} Path" if run_index is None and idx == 1 else None)
            ax.scatter(x_vals[0], y_vals[0], z_vals[0],
                       color='green', marker='o', s=50)
            ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1],
                       color='red', marker='X', s=50)

    # legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Tooltip Path'),
        Line2D([0], [0], color='blue', lw=2, label='Pin Path'),
        Line2D([0], [0], color='purple', lw=2, label='Pipe Path'),
        Line2D([0], [0], color='orange', lw=2, label='Center Path'),
        Line2D([0], [0], marker='o', color='w', label='Start Pt',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='End Pt',
               markerfacecolor='red', markersize=10)
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel("X [units]")
    ax.set_ylabel("Y [units]")
    ax.set_zlabel("Z [units]")
    title = f"3D Object Paths from {folder_name}"
    if run_index is not None:
        title += f" — Run {run_index + 1}"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mesh_map = {
        "Tooltip": "/path/to/tooltip_frame.stl",
        "Pin": "/home/rhino/ENE701-Project/mesh/Pin.stl",
        "Pipe": "/home/rhino/ENE701-Project/mesh/Pipe.stl",
        "Center": "/home/rhino/ENE701-Project/mesh/Center.stl"
    }

    plot_all_3d_paths(
        "2025-08-06_10-59-00",
        mesh_map,
        cutoff=200,
        run_index=0  # or None to plot all runs
    )
