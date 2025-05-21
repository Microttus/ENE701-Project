# Understanding Residual Analysis: Autocorrelation and FFT Plots

When analyzing residuals (differences between predicted and actual values), it's important to determine whether the noise is random or has some structure. Two powerful tools for this are:

---

## 🔁 1. Autocorrelation Plot

### ✅ What It Shows:
An autocorrelation plot shows the correlation of the signal with itself at different time lags.

### 📈 How to Read It:
- **X-axis:** Time lag (number of steps between points).
- **Y-axis:** Correlation coefficient (from -1 to 1).
- **Bars:** Each bar shows how correlated the residuals are with themselves at that lag.

### 🧠 Interpretation:
- **No significant bars (within confidence bounds):**
  - Noise is likely **white noise** (no autocorrelation).
- **Significant spikes at regular intervals:**
  - Indicates **periodicity** or **cyclical behavior** in the noise.
- **Long tail of bars:**
  - Suggests **trend** or **long-memory process**.

---

## 📊 2. FFT (Fast Fourier Transform) Plot

### ✅ What It Shows:
The FFT converts the residual signal from the time domain into the frequency domain, showing which frequencies are most dominant.

### 📈 How to Read It:
- **X-axis:** Frequency (in Hz, if sampling rate is provided).
- **Y-axis:** Amplitude (magnitude of each frequency component).

### 🧠 Interpretation:
- **Flat spectrum:** Noise is likely **white noise**.
- **Peak(s) at specific frequency:** Indicates **periodic components** in the noise.
  - For example, a peak at 0.1 Hz means a repeating pattern every 10 time steps.
- **Multiple peaks:** Might suggest **complex, multi-frequency patterns**.

---

## 🧪 Combining Both:
- If both plots suggest randomness → Your residuals are likely noise.
- If both show structure → Consider modeling it with **time series models** (e.g., ARIMA) or exploring **sources of systematic error**.

---

## 🛠️ Use Cases:
- Detect sensor drift, mechanical oscillation, or feedback loops.
- Validate assumption that noise is independent and identically distributed (i.i.d.).
- Refine your regression or filtering models.
