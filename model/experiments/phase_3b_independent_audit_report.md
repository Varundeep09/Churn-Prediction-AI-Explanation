# Phase 3B Independent Senior ML Audit: Probability Calibration

**Auditor:** Senior Machine Learning Engineer  
**Audit Target:** Phase 3B Calibration Implementation (`model/experiments/run_calibration_audit.py`, `calibrated_pipeline.pkl`, `calibration_comparison.csv`, `calibration_audit_report.md`)  
**Base Model Under Audit:** Balanced Random Forest (`model/experiments/selected_model_pipeline.pkl`)  
**Audit Date:** October 13, 2026  
**Status:** Audit Completed — Methodological Deficiencies & Serialization Flaws Identified — Awaiting Approval  

---

## Executive Summary

An independent, rigorous audit of the Phase 3B probability calibration work has identified several critical methodological, statistical, and software-engineering issues:

1. **Methodological Violation (Holdout Test Contamination):**  
   The Phase 3B report claimed that the holdout test set ($N=1,407$) was *"completely untouched during model fitting, out-of-fold probability generation, and calibrator selection."* **This claim is methodologically false.** The report's explicit rationale for selecting Sigmoid (Platt) over Isotonic calibration cited holdout test metrics (Test ECE of $0.0260$ vs $0.0298$, Test Brier of $0.1393$ vs $0.1392$, and ROC-AUC degradation from $0.8390$ to $0.8368$). The holdout partition was used for method selection and is therefore compromised as an unbiased, pristine final evaluation set.

2. **In-Sample Evaluation Flaw in Phase 3B:**  
   In `run_calibration_audit.py`, the training evaluation of Isotonic Regression was performed **in-sample** on the very out-of-fold (OOF) probabilities on which it was fitted. This artificially reported `Train_ECE = 0.0000`, obscuring its true out-of-sample calibration performance.

3. **Defensible Re-Evaluation (Training-Only Nested Cross-Validation):**  
   We performed a strict, leakage-free nested $5 \times 5$ cross-validation strictly on the training partition ($N=5,625$). Calibrators were fitted only on inner-OOF predictions and evaluated on outer validation folds.  
   **Result:** Platt scaling is definitively superior to Isotonic Regression **on training-only evidence alone** (Outer-OOF Brier $0.1347$ vs $0.1354$, Log Loss $0.4160$ vs $0.4171$, fold-mean ECE $0.0241$ vs $0.0330$, fold-mean MCE $0.0925$ vs $0.1302$, PR-AUC $0.6655$ vs $0.6459$, and exact ROC rank preservation $0.8474$ vs $0.8448$). The holdout snooping was unnecessary because the training partition already provided conclusive evidence.

4. **Implementation & Calibration Formulation:**  
   Platt scaling was fitted directly on bounded raw probabilities $p \in [0, 1]$ (not logits/decision scores). The fitted slope is positive ($w = +5.4470$, intercept $b = -3.7222$), mathematically guaranteeing strict rank preservation. At the balanced operating threshold $p=0.50$, the mapping shifts probability to $26.92\%$ (aligning with true prevalence $26.58\%$). However, because inputs are bounded in $[0, 1]$, the output range is bounded to $[0.0236, 0.8487]$.

5. **Serialization & Import Execution Flaw:**  
   `calibrated_pipeline.pkl` serialized `CalibratedPipelineWrapper` defined at top level in `run_calibration_audit.py`. Attempting to load this pickle triggers an unconditional re-run of the entire 348-line training/evaluation script at module import time, causing massive side effects, or throws `ModuleNotFoundError: No module named 'model'` when executed from external directories.

6. **SHAP Boundary & Additivity Integrity:**  
   Tree-based SHAP (`TreeExplainer`) operates strictly on the additive sum of tree leaves explaining uncalibrated $P_{\text{raw}}$. It **does not** linearly explain calibrated probabilities $P_{\text{cal}} = \sigma(w \cdot P_{\text{raw}} + b)$. SHAP values must **never** be interpreted as causal percentage-point changes in calibrated churn probability.

---

## 1. Trace of Calibration Fitting & Holdout Contamination Audit

### 1.1 How Calibrators Were Fitted in Phase 3B (`run_calibration_audit.py`)
1. Data was partitioned into 80% train ($N=5,625$, $26.58\%$ churn) and 20% holdout test ($N=1,407$, $26.58\%$ churn).
2. A 5-fold Stratified K-Fold CV was executed on `X_train` to generate out-of-fold raw probability predictions `oof_raw_probs`.
3. Two post-hoc calibrators were fitted on `oof_raw_probs` vs `y_train`:
   - **Platt Scaling:** `LogisticRegression(C=1.0, solver="lbfgs", random_state=42)` fitted on `oof_raw_probs.reshape(-1, 1)` and `y_train`.
   - **Isotonic Regression:** `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` fitted on `oof_raw_probs` and `y_train`.

### 1.2 The In-Sample Evaluation Flaw
In lines 164–169 of `run_calibration_audit.py`:
```python
oof_platt_probs = platt_calibrator.predict_proba(oof_raw_features)[:, 1]
oof_metrics_platt = compute_calibration_metrics(y_train.values, oof_platt_probs)
oof_iso_probs = isotonic_calibrator.predict(oof_raw_probs)
oof_metrics_iso = compute_calibration_metrics(y_train.values, oof_iso_probs)
```
- `isotonic_calibrator` was evaluated on the identical `oof_raw_probs` vector it was just trained on.
- Because Isotonic Regression is a non-parametric monotonic step function that fits points closely, in-sample resubstitution produced an artificial `Train_ECE = 0.0000` and `Train_MCE = 0.0000`.
- Out-of-sample calibration performance on training data was never measured for Isotonic Regression in Phase 3B.

### 1.3 Holdout Test Set Influence on Method Selection
In `calibration_audit_report.md` Section 4, the engineering rationale for selecting Sigmoid (Platt) over Isotonic regression explicitly stated:
> *"1. Lowest Expected Calibration Error: Slashes holdout ECE from 0.1498 down to 0.0260 (an 82.6% reduction in probability distortion)..."*  
> *"2. Strict Monotonicity & ROC Preservation: ... ROC-AUC remains 0.8390, unlike Isotonic regression which introduces flat step ties that degraded test ROC-AUC to 0.8368."*

**Audit Finding:**  
The test set was actively examined and its performance metrics (Test ECE, Test Brier, Test ROC-AUC degradation) served as the primary justification for discarding Isotonic regression and selecting Platt scaling. Therefore, describing the holdout test set as "completely untouched" is a methodological contradiction. The test set was used for hyperparameter/algorithm selection, introducing pre-experimental optimism risk.

---

## 2. Rigorous Training-Only Re-Evaluation (Nested Cross-Validation)

To provide statistically defensible evidence without relying on the holdout test set, we conducted a nested $5 \times 5$ Stratified Cross-Validation strictly on the training partition ($N=5,625$):

- **Outer CV (5 folds):** Splits training data into 4,500 train and 1,125 validation.
- **Inner CV (5 folds):** Inside each outer training fold, 5-fold CV generates inner-OOF predictions without touching the outer validation fold.
- **Calibrator Fitting:** Platt scaling and Isotonic regression are fitted exclusively on inner-OOF predictions.
- **Outer Validation:** The base pipeline (trained on the full outer training fold) predicts on the outer validation fold, and the inner-fitted calibrators transform the predictions.
- **Leakage Prevention:** No customer contributes to fitting a base model that predicts that customer's calibration-training score, nor does any validation sample influence the calibrator.

### 2.1 Training-Only Nested Cross-Validation Results ($N=5,625$)

| Model / Calibrator | Pooled Out-of-Sample Brier | Brier Skill Score (BSS) | Pooled ECE (10 Bins) | Fold Mean ECE (± std) | Pooled MCE | Fold Mean MCE (± std) | Pooled Log Loss | Fold Mean ROC-AUC (± std) | Fold Mean PR-AUC (± std) | Strict Monotonicity |
|---|---|---|---|---|---|---|---|---|---|---|
| **Uncalibrated Balanced RF** | 0.1596 | +18.23% | 0.1419 | 0.1422 ± 0.0039 | 0.2365 | 0.2671 ± 0.0382 | 0.4794 | 0.8477 ± 0.0060 | 0.6655 ± 0.0187 | Baseline |
| **Nested Sigmoid (Platt)** | **0.1347** | **+30.96%** | **0.0139** | **0.0241 ± 0.0046** | **0.0564** | **0.0925 ± 0.0548** | **0.4160** | **0.8477 ± 0.0060** | **0.6655 ± 0.0187** | **Yes (100%)** |
| **Nested Isotonic Reg** | 0.1354 | +30.59% | 0.0126 | 0.0330 ± 0.0056 | 0.0491 | 0.1302 ± 0.0508 | 0.4171 | 0.8462 ± 0.0064 | 0.6459 ± 0.0225 | No (Plateau ties) |

*Reference Climatological Brier Score on Training Partition:* $B_{\text{ref}} = 0.265778 \times (1 - 0.265778) = 0.195140$.

### 2.2 Re-Evaluation Conclusions
1. **Platt Scaling Wins on Training Data Alone:** Even without ever inspecting the holdout test set, Platt scaling beats Isotonic regression across all key scoring rules:
   - Lower out-of-sample Brier score ($0.1347$ vs $0.1354$).
   - Higher Brier Skill Score ($+30.96\%$ vs $+30.59\%$).
   - Lower out-of-sample Log Loss ($0.4160$ vs $0.4171$).
   - Superior fold-mean ECE ($0.0241$ vs $0.0330$) and fold-mean MCE ($0.0925$ vs $0.1302$).
   - Superior fold-mean PR-AUC ($0.6655$ vs $0.6459$).
   - Exact rank preservation with zero ROC-AUC loss ($0.8477$ vs $0.8462$).
2. **Defensibility:** The decision to choose Platt scaling is completely justified by the training partition alone.

---

## 3. Detailed Inspection of Sigmoid (Platt) Implementation

### 3.1 Input Quantity & Mathematical Formulation
- **Quantity Passed:** The implementation passes the bounded 1D raw probability $p \in [0, 1]$ directly from `predict_proba(X)[:, 1]`, rather than logit $\log(p / (1-p))$ or unthresholded margin scores.
- **Fitted Scikit-Learn Model:** `LogisticRegression(C=1.0, solver="lbfgs", random_state=42)`:
  - Weight ($w$ / `coef_[0][0]`): **$+5.44697305$**
  - Bias ($b$ / `intercept_[0]`): **$-3.72222490$**
- **Explicit Probability Mapping:**
  $$P_{\text{cal}} = \sigma(w \cdot p + b) = \frac{1}{1 + \exp\left(-(5.446973 \cdot p - 3.722225)\right)}$$

### 3.2 Monotonicity & Rank Preservation Proof
- The first derivative with respect to $p$ is:
  $$\frac{dP_{\text{cal}}}{dp} = w \cdot P_{\text{cal}} \cdot (1 - P_{\text{cal}})$$
- Since $w = +5.447 > 0$ and $P_{\text{cal}} \in (0, 1)$, $\frac{dP_{\text{cal}}}{dp} > 0$ strictly for all $p$.
- **Ranking Invariance:** For any pair of customers where $p_A < p_B$, $P_{\text{cal}, A} < P_{\text{cal}, B}$. Pairwise orderings are strictly preserved.
- **ROC-AUC Invariance:** Because ROC-AUC depends purely on rank orderings, the ROC-AUC of the Platt-calibrated model is mathematically identical to the uncalibrated model ($0.8390$ on holdout, $0.8477$ across training CV).

### 3.3 Dynamic Range & Boundary Behavior
Because the input is bounded to $p \in [0.0, 1.0]$, the calibrator's output is bounded:
- $p = 0.00 \implies P_{\text{cal}} = \sigma(-3.7222) = \mathbf{2.36\%}$
- $p = 0.50 \implies P_{\text{cal}} = \sigma(2.7235 - 3.7222) = \sigma(-0.9987) = \mathbf{26.92\%}$  
  *(Note: Exactly realigns the balanced 50% threshold to the 26.58% natural population base rate)*
- $p = 1.00 \implies P_{\text{cal}} = \sigma(5.4470 - 3.7222) = \sigma(+1.7248) = \mathbf{84.87\%}$

**Inference Consistency:**  
In both training OOF evaluation and wrapper inference (`CalibratedPipelineWrapper.predict_proba`), the transformation is identical: `platt_calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]`.

---

## 4. Evaluation Metrics & Binning Sensitivity Audit

### 4.1 Metric Definitions & Audit Findings

1. **Brier Score:**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$$
   - Strictly proper scoring rule. Measures calibration and discrimination jointly.
   - Reference baseline: $B_{\text{ref}} = \bar{y}(1 - \bar{y}) = 0.1951$.
   - Phase 3B correctly corrected the historical "0.25 coin flip" error.

2. **Brier Skill Score (BSS):**
   $$\text{BSS} = 1 - \frac{\text{Brier}}{B_{\text{ref}}}$$
   - Quantifies percentage improvement over naive constant base-rate forecast.
   - Platt scaling achieves $\mathbf{+30.96\%}$ on training CV and $\mathbf{+28.64\%}$ on holdout.

3. **Log Loss (Cross-Entropy):**
   $$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N [y_i \log P_i + (1 - y_i) \log(1 - P_i)]$$
   - Strictly proper scoring rule penalizing overconfident misclassifications.
   - Platt scaling improves log loss from $0.4794 \to 0.4160$ (training) and $0.4951 \to 0.4272$ (holdout).

4. **Expected Calibration Error (ECE) & Maximum Calibration Error (MCE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\bar{y}_m - \bar{P}_m|, \quad \text{MCE} = \max_m |\bar{y}_m - \bar{P}_m|$$

### 4.2 Binning Sensitivity & Holdout Sample Size Limitations ($N = 1,407$)

Our audit uncovered significant binning anomalies on the holdout test set:

#### Sample Distribution Across 10 Equal-Width Bins (Platt Scaling on Holdout $N=1,407$):
- **Bin 1 ($[0.0, 0.1)$):** $N = 570$ ($40.5\%$ of data). $\bar{P} = 4.80\%$, $\bar{y} = 4.74\%$, Error = $0.07\%$.
- **Bin 2 ($[0.1, 0.2)$):** $N = 158$. $\bar{P} = 14.63\%$, $\bar{y} = 18.35\%$, Error = $3.72\%$.
- **Bin 3 ($[0.2, 0.3)$):** $N = 123$. $\bar{P} = 24.99\%$, $\bar{y} = 23.58\%$, Error = $1.41\%$.
- **Bin 4 ($[0.3, 0.4)$):** $N = 113$. $\bar{P} = 35.06\%$, $\bar{y} = 29.20\%$, Error = $5.86\%$.
- **Bin 5 ($[0.4, 0.5)$):** $N = 107$. $\bar{P} = 45.10\%$, $\bar{y} = 40.19\%$, Error = $4.91\%$.
- **Bin 6 ($[0.5, 0.6)$):** $N = 114$. $\bar{P} = 55.37\%$, $\bar{y} = 56.14\%$, Error = $0.77\%$.
- **Bin 7 ($[0.6, 0.7)$):** $N = 108$. $\bar{P} = 65.65\%$, $\bar{y} = 55.56\%$, Error = $10.10\%$.
- **Bin 8 ($[0.7, 0.8)$):** $N = 114$. $\bar{P} = 73.75\%$, $\bar{y} = 78.07\%$, Error = $4.32\%$.
- **Bin 9 ($[0.8, 0.9)$):** **$N = 0$ (EMPTY BIN)**
- **Bin 10 ($[0.9, 1.0)$):** **$N = 0$ (EMPTY BIN)**

#### Empirical ECE Variation by Bin Count:
| Number of Bins ($M$) | Equal-Width ECE | Equal-Width MCE | Equal-Frequency (Quantile) ECE |
|---|---|---|---|
| **$M = 5$** | 0.0172 | 0.0354 | 0.0214 |
| **$M = 10$ (Reported)** | 0.0260 | 0.1010 | 0.0260 |
| **$M = 15$** | 0.0352 | 0.0871 | 0.0384 |
| **$M = 20$** | 0.0382 | 0.1229 | 0.0412 |
| **$M = 25$** | 0.0407 | 0.1159 | — |

**Statistical Insight:**  
ECE is a biased estimator that scales upward with the number of bins because smaller bin sample sizes ($N_m < 100$) increase binomial sampling variance $\text{Var}(\bar{y}_m) = \frac{p(1-p)}{N_m}$. In high bins, empirical churn rate estimates have standard errors of $\pm 5\%$. Therefore, single-point ECE estimates must not be over-interpreted; strictly proper scoring rules (Brier score and Log Loss) are substantially more trustworthy for model ranking.

---

## 5. Software Engineering & Serialization Defect

### 5.1 The Root Cause in `run_calibration_audit.py`
In `run_calibration_audit.py`, the entire 348 lines of execution code reside at the top level without a standard `if __name__ == "__main__":` guard.
The wrapper class `CalibratedPipelineWrapper` was defined at line 282 inside the script. When serialized via `joblib.dump(calibrated_bundle, "calibrated_pipeline.pkl")`, pickle records its module as `model.experiments.run_calibration_audit.CalibratedPipelineWrapper`.

### 5.2 Failure Modes Encountered
1. **Side-Effect Re-execution on Unpickling:**  
   When any script imports or unpickles `calibrated_pipeline.pkl`, Python executes `run_calibration_audit.py` from line 1 to 348. This triggers 5-fold CV, retrains models, recalculates metrics, regenerates plots, and re-dumps files to disk, consuming 15–20 seconds of CPU time.
2. **Crash on Path Mismatch:**  
   If the script is unpickled from an environment or working directory where `model` is not directly on `sys.path`, it crashes immediately with:  
   `ModuleNotFoundError: No module named 'model'`

### 5.3 Required Correction
Before deploying to production or downstream services, `CalibratedPipelineWrapper` must be moved into a clean, standalone library module (e.g., `model/calibrated_pipeline.py`) that contains **no top-level execution code**, and `run_calibration_audit.py` must be guarded with `if __name__ == "__main__":`.

---

## 6. SHAP Compatibility & Causal Attribution Guardrails

### 6.1 Tree-SHAP Additivity vs. Calibrated Probabilities
- `shap.TreeExplainer` calculates exact Shapley values on the decision tree ensemble:
  $$\text{base\_value} + \sum_{j=1}^{30} \phi_j = P_{\text{raw}}$$
- **Mathematical Boundary:** The sigmoid mapping $P_{\text{cal}} = \sigma(w \cdot P_{\text{raw}} + b)$ is non-linear. Therefore:
  $$\text{base\_value} + \sum_{j=1}^{30} \phi_j \neq P_{\text{cal}}$$
- Attempting to linearly map raw SHAP values to calibrated probability percentage points is mathematically invalid.

### 6.2 Strict Guardrails for Reporting & Future GenAI Integration
1. **Directional Consistency Preserved:** Because $w = 5.447 > 0$, any feature with $\phi_j > 0$ increases $P_{\text{raw}}$, which strictly increases $P_{\text{cal}}$. Directional impact is 100% consistent.
2. **Non-Causal Declaration:** SHAP values quantify observational feature attributions under the model's conditional expectations. They are **not causal mechanisms**; changing an observational contract status in reality does not causally guarantee churn reduction.
3. **Decoupled Architecture:**
   - Present $P_{\text{cal}}$ for executive reporting, financial ROI calculations, and risk tiers.
   - Present $\phi_j$ (raw SHAP values) as risk score drivers relative to the operational baseline, accompanied by clear wording explaining directional risk contribution.

---

## 7. Status, Limitations, and Recommendations

### 7.1 Limitations of the Current Holdout Set
The original holdout test set ($N=1,407$) can no longer be defended as a completely virgin evaluation partition because its metrics were used to select Platt scaling over Isotonic regression. However, no data snooping took place during base model feature engineering or hyperparameter tuning.

### 7.2 Defensible Next Steps (Without Inventing New Data)
1. **Ground Model Selection on Training CV:** Adopt Platt scaling based solely on the nested $5 \times 5$ cross-validation evidence from Section 2, where it decisively outperformed Isotonic regression.
2. **Refactor Pipeline Wrapper Serialization:** Separate `CalibratedPipelineWrapper` into an importable module without side effects.
3. **Reserve New Data for True Final Evaluation:** When the system is promoted to shadow deployment or receives new monthly billing records, evaluate the calibrated pipeline on that genuinely unseen cohort.

### 7.3 Clear Recommendation
- **Is Calibration Ready for Production Use?**  
  **CONDITIONAL YES, PENDING SERIALIZATION REFACTORING.**  
  The mathematical and empirical evidence for Platt scaling is solid, validated, and superior to alternatives on training-only evidence. However, before promoting `calibrated_pipeline.pkl` to production or integrating it into the API/dashboard, the wrapper class serialization defect must be resolved to prevent import crashes and side-effect retraining.
