# ENE701-Project
Repo for my Statistics project in the ENE701 subject for the PhD course on UiA

## Noise Ananlysis

**Q-Q Plot Interpretation:**

- If the points fall along the reference line, the residuals follow the specified distribution (e.g., normal).
- S-shaped curve: Indicates heavy or light tails in the data distribution.
- Upward/Downward bends:
  - Points above the line at the start: Positive skew.
  - Points below the line at the start: Negative skew.
- Outliers: Points far away from the line indicate potential deviations from the expected distribution.

## Types of Distributions for Q-Q Plot Analysis

### 1. Normal Distribution (`norm`)
- Symmetric, bell-shaped curve.
- Most common assumption for noise.
- Indicates normally distributed residuals.

---

### 2. Uniform Distribution (`uniform`)
- Equal probability for all values within a range.
- No central tendency.
- Residuals are evenly spread across the range.

---

### 3. Laplace Distribution (`laplace`)
- Heavy-tailed distribution.
- Indicates the presence of more extreme values or outliers.
- Steeper peak and fatter tails compared to a normal distribution.

---

### 4. Exponential Distribution (`expon`)
- Right-skewed distribution.
- Often used to model the time until an event occurs.
- Suitable for modeling waiting times or decay.

---

### 5. Gamma Distribution (`gamma`)
- Right-skewed, can model waiting times or decay processes.
- More flexible shape than the exponential distribution.

---

### 6. Beta Distribution (`beta`)
- Defined between 0 and 1.
- Useful for bounded data, such as proportions and probabilities.

---

### 7. Chi-Square Distribution (`chi2`)
- Right-skewed, dependent on degrees of freedom.
- Often used for variance analysis and goodness-of-fit tests.

---

### 8. Student's t-Distribution (`t`)
- Similar to normal distribution but with heavier tails.
- Useful for small sample sizes or data with outliers.