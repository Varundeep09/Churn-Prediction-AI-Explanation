# Phase 3B: Probability Calibration Audit & Validation Report

**Author:** Senior Machine Learning Engineer  
**Model Under Audit:** Balanced Random Forest (`model/experiments/selected_model_pipeline.pkl`)  
**Evaluation Scope:** Reliability Diagrams, Brier Scores, ECE/MCE, and Platt vs. Isotonic Calibration  
**Execution Date:** October 13, 2026  

---

## 1. Data Split & Leakage Audit

To prevent data snooping and circular validation, strict data partitioning boundaries were maintained:
1. **Total Cleaned Records:** $N = 7,032$ (26.58% churn prevalence).
2. **Training Partition ($N = 5,625$):**
   - Churn rate: $p_{\text{train}} = 26.5778\%$ (1,495 churners).
   - **Climatological Reference Brier Score:** $B_{\text{ref}} = p(1 - p) = 0.265778 \times 0.734222 = \mathbf{0.195140}$.
   - All candidate calibrators (Sigmoid / Platt scaling and Isotonic regression) were trained **strictly on the 5-fold out-of-fold (OOF) cross-validation predictions of this training partition**.
3. **Holdout Test Partition ($N = 1,407$):**
   - Churn rate: $p_{\text{test}} = 26.5814\%$ (374 churners).
   - Reference Brier Score: $B_{\text{ref}} = \mathbf{0.195157}$.
   - The test set was **completely untouched** during model fitting, out-of-fold probability generation, and calibrator selection. It was evaluated exactly once to measure generalization.

---

## 2. Existing Uncalibrated Calibration Findings

### The "0.25 Brier Score" Misconception Corrected:
- In earlier discussions, $0.25$ was cited as the random baseline. However, on an imbalanced dataset with $26.58\%$ churn prevalence, predicting $0.50$ for everyone produces a high error. A naive baseline that simply predicts the base rate $p = 0.2658$ achieves a **reference Brier score of 0.1951**.
- The uncalibrated Random Forest achieved a Brier score of **0.1596** (OOF) and **0.1673** (Test). This represents a positive **Brier Skill Score (BSS) of +14.27%** over the naive base rate.

### Root Cause of Probability Distortion:
- Because the Random Forest was trained with `class_weight='balanced'`, churn instances were upweighted by $2.76\times$ during tree splitting.
- Individual tree leaves estimate probability as the weighted sample proportion:
  $$\hat{P} = \frac{\sum w_i y_i}{\sum w_i}$$
- **Empirical Consequence:** The model exhibits severe probability inflation. On the holdout test set, the uncalibrated model's:
  - **Expected Calibration Error (ECE):** **0.1498** (an average $\approx 15\%$ probability gap across all deciles).
  - **Maximum Calibration Error (MCE):** **0.2648** (overestimating by up to $26.5\%$ in high-confidence bins).
  - **Mean Predicted Probability:** **40.77%** (vs true base rate of **26.58%**).
- When the uncalibrated model predicts a customer has a $55\%$ chance of churning, their true empirical likelihood is only $\approx 35\%$.
- **Audit Conclusion:** Raw Random Forest scores **cannot be presented to end users as frequentist churn probabilities**.

---

## 3. Calibration Methods Compared & Validation Results

Two standard post-hoc calibration methods were fit on out-of-fold training predictions and evaluated on the holdout test partition:

1. **Sigmoid Calibration (Platt Scaling):** Fits a logistic regression model $\sigma(A \cdot f(x) + B)$ on raw model predictions.
2. **Isotonic Regression:** Fits a non-parametric piecewise constant isotonic step function.

### Benchmark Comparison Table (Holdout Test Set $N = 1,407$):

| Architecture / Method | Test Brier Score | Brier Skill Score (BSS) | Test ECE (Avg Error) | Test MCE (Max Error) | Test Log Loss | Test ROC-AUC | Ranking Preserved? |
|---|---|---|---|---|---|---|---|
| **Uncalibrated Balanced RF** | 0.1673 | +14.27% | 0.1498 | 0.2648 | 0.4951 | **0.8390** | Yes (Baseline) |
| **Sigmoid (Platt Scaling)** | **0.1393** | **+28.64%** | **0.0260** | **0.1010** | 0.4272 | **0.8390** | **100% Exact** |
| **Isotonic Regression** | 0.1392 | +28.67% | 0.0298 | 0.1231 | **0.4259** | 0.8368 | Degraded (ties) |

### Decile Calibration Breakdown (Platt Scaling on Test Set):
- **Bin 1 (0–10% range):** Mean predicted probability = **4.80%**, Actual churn rate = **4.74%** (Error: **0.07%**).
- **Bin 3 (20–30% range):** Mean predicted probability = **24.99%**, Actual churn rate = **23.58%** (Error: **1.41%**).
- **Bin 6 (50–60% range):** Mean predicted probability = **55.37%**, Actual churn rate = **56.14%** (Error: **0.77%**).

---

## 4. Recommended Calibration Method & Rationale

**Selected Method:** **Sigmoid Calibration (Platt Scaling)**

### Engineering Rationale:
1. **Lowest Expected Calibration Error:** Slashes holdout ECE from **0.1498 down to 0.0260** (an **82.6% reduction in probability distortion**).
2. **Strict Monotonicity & ROC Preservation:** Because the sigmoid mapping $g(z) = \frac{1}{1 + e^{-(Az+B)}}$ with $A > 0$ is strictly increasing, it preserves the exact pairwise ranking of all customers. **ROC-AUC remains 0.8390**, unlike Isotonic regression which introduces flat step ties that degraded test ROC-AUC to 0.8368.
3. **Smooth Continuous Output:** Produces smooth probability distributions rather than step-function plateaus.

---

## 5. Impact on SHAP & Future GenAI Explanations

A critical architectural consideration is how calibration interacts with SHAP explainability:

### The Mathematical Reality:
- `shap.TreeExplainer` operates on tree structure and node weights. It explains the **underlying Random Forest output** $P_{\text{raw}}$, strictly guaranteeing:
  $$\text{base\_value} + \sum_{i=1}^{30} \phi_i = P_{\text{raw}}$$
- It **does not** directly explain the non-linear sigmoid transform $P_{\text{cal}} = \sigma(A \cdot P_{\text{raw}} + B)$.
- If one attempts to explain the calibrated wrapper directly, additivity is lost unless slow sampling approximations (`KernelExplainer`) are used.

### The Recommended Dual-Score Protocol:
We decouple the score presentation into two mathematically sound outputs:
1. **Calibrated Churn Probability ($P_{\text{cal}}$):**
   - Derived via Platt scaling ($P_{\text{cal}} \in [0.0, 1.0]$).
   - Used for frequentist customer communication, retention ROI calculations, and executive dashboards (ECE = 0.026).
2. **SHAP Feature Attributions ($\phi_i$):**
   - Derived from `TreeExplainer` on the underlying tree ensemble.
   - Retains exact additivity against the operational decision threshold ($t = 0.50$).
   - Because Platt scaling is strictly monotonic, any positive SHAP contributor that increases $P_{\text{raw}}$ **strictly increases** $P_{\text{cal}}$.
3. **GenAI Prompt Injection:**
   - The LLM prompt receives the true calibrated probability for narrative text ("Customer has an estimated 28% likelihood of churning"), alongside the top SHAP drivers ("Primary risk factor: Month-to-month contract adding +8.0% to risk score").

---

## 6. Artifacts Created

All deliverables are saved in `model/experiments/` without modifying any production code:
- `model/experiments/run_calibration_audit.py`: The executable audit script.
- `model/experiments/calibration_comparison.csv`: Tabular benchmark comparing Raw, Sigmoid, and Isotonic across Train and Test sets.
- `model/experiments/calibration_curves_comparison.png`: High-resolution reliability diagram and probability histogram.
- `model/experiments/calibrated_pipeline.pkl`: Serialized `CalibratedPipelineWrapper` combining the base pipeline and fitted Platt calibrator.
- `model/experiments/calibration_audit_results.json`: Full decile-level calibration metrics.

---

## 7. Status & Next Step

- **Status:** Probability calibration audit is complete and mathematically validated.
- **Unresolved Issues:** None.
- **Next Decision:** Awaiting approval before proceeding to GenAI integration or model promotion.
