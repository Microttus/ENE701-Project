import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_tool_path_data(file_path: str) -> None:
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

        plt.plot(x_vals, label=f"Round {idx+1} - X")
        plt.plot(y_vals, label=f"Round {idx+1} - Y")
        plt.plot(z_vals, label=f"Round {idx+1} - Z")

    plt.xlabel("Trial")
    plt.ylabel("Tool Point Value")
    plt.title("Tool Path Data Visualization (X, Y, Z)")
    #plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_3d_tool_path_data(file_path: str) -> None:
    """Reads the CSV data and plots the path in 3D space."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, row in data.iterrows():
        x_vals = row[::3]
        y_vals = row[1::3]
        z_vals = row[2::3]

        ax.plot(x_vals, y_vals, z_vals, label=f"Round {idx+1}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Tool Path Data Visualization")
    #ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_tool_path_data("../data/tooltip_positions_3.csv")
    plot_3d_tool_path_data("../data/tooltip_positions_3.csv")
