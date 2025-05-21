# 📊 Statistical Analysis of Robotic Tool Paths

This project is developed as part of the **ENE701 - Statistics** course and focuses on applying statistical methods to real-world data generated from a robotic system trained with reinforcement learning (RL). The goal is to analyze the motion of a robot arm's tooltip while performing a disassembly task.

---

## 🎯 Project Objective

Use statistical methods to explore, interpret, and model tooltip trajectories recorded during the execution of a robotic disassembly task. This includes generating synthetic data, analyzing real recorded data, evaluating residuals, and understanding the type and structure of the noise present.

---

## 📂 Project Structure

```
.
├── CHANGELOG.md                  # Summary of changes and updates
├── LICENSE                       # Project license
├── README.md                     # This file
├── data/                         # Real and synthetic data, plus visualization outputs
│   ├── tool_path_data.csv        # Generated synthetic path data
│   ├── tooltip_positions_*.csv   # Recorded RL-based robot tooltip positions
│   ├── images/                   # Generated plots (residuals, regressions, etc.)
├── docs/                         # Supplemental documents and markdown explanations
│   ├── autoffm.md                # Autocorrelation and FFT method explanation
│   └── qqplotguide.md            # Guide for interpreting Q-Q plots
├── source/                       # Python source code
│   ├── path-generation.py        # Code to generate synthetic data with noise
│   ├── path_analysis.py          # Code to visualize and explore real/synthetic paths
│   └── regression_ananlysis.py   # Regression modeling and residual analysis
```

---

## 📈 Techniques Used

* **Quadratic and Polynomial Regression**
* **Residual Noise Estimation**
* **Q-Q Plot Analysis**
* **Autocorrelation Function (ACF)**
* **Fast Fourier Transform (FFT)**
* **Time Series Visualization**

---

## 🧪 Dataset

The primary data used in this project consists of:

* **Synthetic trajectories** generated using second-order optimal paths plus Gaussian Process-based noise.
* **Recorded robot tooltip positions** during disassembly, produced by a trained RL agent performing a proof-of-concept task.

---

## 🎓 Course Context

This project is conducted within the scope of the ENE701 Statistics course, whose aim is:

> *To enable participants to perform statistical analysis on their own data and relate the results to practical applications using both frequentist and Bayesian concepts.*

Topics covered in this course include:

* Condensing and visualizing data
* Probability and statistical inference
* Hypothesis testing and confidence intervals
* Gaussian and Poisson processes
* Regression analysis (linear, multiple, logistic)
* Planning and interpreting experiments

This project forms a part of the practical component and will contribute toward the final oral examination.

---

## 📌 How to Run

1. Clone or download the project.
2. Install Python dependencies:

   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn statsmodels
   ```
3. Run individual analysis scripts in the `source/` folder to generate data or plots.

   ```bash
   python source/path_analysis.py
   ```

---

## 👨‍💻 Author

This work is part of the practical coursework by a student of Mechatronics Engineering, using real RL robot data for meaningful statistical evaluation.

---

## 📜 License

This project is licensed under the terms in the `LICENSE` file.

