import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import os


def perform_regression_analysis(file_path: str, dof: int = 2, cutoff: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads the CSV data and performs quadratic regression for each axis, plotting each axis in separate subplots with regression overlay."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = pd.read_csv(file_path, header=None)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes_titles = ['X Axis', 'Y Axis', 'Z Axis']
    light_blue = '#87CEEB'
    dark_blue = '#1E90FF'

    residuals = []
    for axis_index in range(3):
        axis_data = data.iloc[:, axis_index::3]
        trial_count = axis_data.shape[1] if cutoff is None else min(cutoff, axis_data.shape[1])
        axis_data = axis_data.iloc[:, :trial_count]
        time_steps = np.arange(trial_count)

        # Flatten data for regression
        X = np.tile(time_steps, axis_data.shape[0]).reshape(-1, 1)
        y = axis_data.values.flatten()

        # Perform quadratic regression
        poly = PolynomialFeatures(degree=dof)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Plot each round as a line
        for idx, row in axis_data.iterrows():
            axes[axis_index].plot(time_steps, row, alpha=0.5, color=light_blue)

        # Plot quadratic regression line
        axes[axis_index].plot(time_steps, y_pred[:trial_count], color=dark_blue, linewidth=2, label="Regression Line")
        # Calculate residuals (noise)
        residuals.append(axis_data.values.flatten() - y_pred)
        axes[axis_index].set_title(axes_titles[axis_index])
        axes[axis_index].grid(alpha=0.3)

    plt.xlabel("Trial")
    plt.suptitle("Regression Analysis for Each Axis")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return tuple(residuals)


def plot_residuals(residuals: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Plots the residuals (noise) for each axis in separate subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes_titles = ['X Axis Noise', 'Y Axis Noise', 'Z Axis Noise']
    dark_blue = '#1E90FF'

    for i, axis_residuals in enumerate(residuals):
        axes[i].plot(axis_residuals, color=dark_blue, alpha=0.7)
        axes[i].set_title(axes_titles[i])
        axes[i].grid(alpha=0.3)

    plt.xlabel("Data Point Index")
    plt.suptitle("Residuals (Noise) for Each Axis")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_qq(residuals: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Plots Q-Q plots for the residuals to assess the noise distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    axes_titles = ['X Axis Q-Q Plot', 'Y Axis Q-Q Plot', 'Z Axis Q-Q Plot']

    for i, axis_residuals in enumerate(residuals):
        stats.probplot(axis_residuals, dist=stats.norm, plot=axes[i])
        #stats.probplot(axis_residuals, dist=stats.chi2, sparams=(5,), plot=axes[i])
        axes[i].set_title(axes_titles[i])
        axes[i].grid(alpha=0.3)

    plt.suptitle("Q-Q Plots for Residuals (Noise)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    residuals = perform_regression_analysis("../data/tooltip_positions_3.csv", 2,75)
    plot_residuals(residuals)
    plot_qq(residuals)
