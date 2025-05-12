import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os


def perform_regression_analysis(file_path: str) -> None:
    """Reads the CSV data and performs quadratic regression for each axis, plotting each axis in separate subplots with regression overlay."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes_titles = ['X Axis', 'Y Axis', 'Z Axis']
    light_blue = '#87CEEB'
    dark_blue = '#1E90FF'

    for axis_index in range(3):
        axis_data = data.iloc[:, axis_index::3]
        trial_count = axis_data.shape[1]
        time_steps = np.arange(trial_count)

        # Flatten data for regression
        X = np.tile(time_steps, axis_data.shape[0]).reshape(-1, 1)
        y = axis_data.values.flatten()

        # Perform quadratic regression
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Plot each round as a line
        for idx, row in axis_data.iterrows():
            axes[axis_index].plot(time_steps, row, alpha=0.5, color=light_blue)

        # Plot quadratic regression line
        axes[axis_index].plot(time_steps, y_pred[:trial_count], color=dark_blue, linewidth=2, label="Quadratic Regression")
        axes[axis_index].set_title(axes_titles[axis_index])
        axes[axis_index].grid(alpha=0.3)

    plt.xlabel("Trial")
    plt.suptitle("Quadratic Regression Analysis for Each Axis")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    perform_regression_analysis("../data/tool_path_data.csv")
