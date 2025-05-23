import os
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.fft import fft, fftfreq
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA


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
        #stats.probplot(axis_residuals, dist=stats.t, sparams=(5,), plot=axes[i])
        axes[i].set_title(axes_titles[i])
        axes[i].grid(alpha=0.3)

    plt.suptitle("Q-Q Plots for Residuals (Noise)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_autocorrelation_and_fft(residuals: tuple[np.ndarray, np.ndarray, np.ndarray], sampling_rate: float = 1.0) -> None:
    """
    Plots autocorrelation and FFT (frequency spectrum) for each axis of the residuals.

    Args:
        residuals: A tuple of three numpy arrays (x_residuals, y_residuals, z_residuals).
        sampling_rate: The rate at which the data was sampled (used to calculate frequency axis).
    """
    axes_labels = ['X Axis', 'Y Axis', 'Z Axis']

    for i, axis_res in enumerate(residuals):
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f'Residual Analysis - {axes_labels[i]}')

        # Autocorrelation
        plot_acf(axis_res, ax=axs[0], lags=50)
        axs[0].set_title('Autocorrelation')
        axs[0].grid(alpha=0.3)

        # FFT
        n = len(axis_res)
        yf = fft(axis_res)
        xf = fftfreq(n, d=1.0 / sampling_rate)[:n // 2]  # positive frequencies
        axs[1].plot(xf, 2.0 / n * np.abs(yf[:n // 2]))
        axs[1].set_title('FFT - Frequency Spectrum')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

def plot_arima_fit(residuals: tuple[np.ndarray, np.ndarray, np.ndarray], order: tuple = (2, 0, 2)) -> list[Any]:
    """
    Fits and plots an ARIMA model for each axis of the residuals.

    Args:
        residuals: A tuple of three numpy arrays (x_residuals, y_residuals, z_residuals).
        order: ARIMA order (p, d, q) — default is (2, 0, 2).
    """
    axes_labels = ['X Axis', 'Y Axis', 'Z Axis']

    arima_model_fit = []

    for i, axis_res in enumerate(residuals):
        model = ARIMA(axis_res, order=order)
        model_fit = model.fit()

        arima_model_fit.append(model_fit.fittedvalues)

        # Plot original and fitted
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(axis_res, label="Original Residuals", alpha=0.5)
        ax.plot(model_fit.fittedvalues, label="ARIMA Fitted", color="darkblue")
        ax.set_title(f"ARIMA Fit on {axes_labels[i]} Residuals")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Residual Value")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"=== ARIMA({order}) Model Summary for {axes_labels[i]} ===")
        print(model_fit.summary())

    return arima_model_fit


if __name__ == "__main__":
    residuals = perform_regression_analysis("../data/tooltip_positions_4.2.csv", 5, cutoff=20)
    plot_residuals(residuals)
    plot_qq(residuals)
    plot_autocorrelation_and_fft(residuals, 1.0)
    arima_residuals = plot_arima_fit(residuals, order=(2, 0, 2))
    plot_autocorrelation_and_fft(arima_residuals, 1.0)
