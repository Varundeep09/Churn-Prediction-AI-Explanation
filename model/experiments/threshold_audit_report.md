# Operational Threshold Audit & Validation Report: Calibrated Random Forest Pipeline

**Auditor:** Senior Machine Learning Engineer  
**Audit Target:** Operational Decision Threshold in `CalibratedPipelineWrapper` (`model/calibrated_pipeline.py`, `model/experiments/calibrated_pipeline_refactored.pkl`)  
**Base Benchmark Source:** Phase 2 Benchmarks (`threshold_optimization.csv`, `selected_model_config.json`)  
**Audit Date:** October 13, 2026  
**Status:** Audit Completed — Awaiting Approval  

---

## 1. Executive Summary & Source of Threshold `0.27`

### 1.1 The Source & Origin of `0.27`
In `model/calibrated_pipeline.py` and `run_calibration_audit.py`, the default threshold was set to `threshold=0.27`.  
Our audit determined the exact source and rationale:
1. **Prevalence Alignment:** The natural population churn prevalence in the Telco dataset is **$26.58\%$** ($p_{\text{train}} = 0.265778$, $p_{\text{test}} = 0.265814$). In decision theory, when probabilities are calibrated to true frequentist likelihoods, flagging instances where $P_{\text{cal}} \ge \text{base\_rate}$ is a canonical base-rate decision policy.
2. **Mathematical Projection of Raw 0.50:** The Balanced Random Forest was trained with `class_weight='balanced'`, which inflated tree probabilities by $2.76\times$ (the inverse positive class frequency). Under the fitted Platt calibrator ($w = 5.44697305$, $b = -3.72222490$), the uncalibrated balanced operating threshold of **$p_{\text{raw}} = 0.5000$ maps directly to $P_{\text{cal}} = 0.269190 \approx \mathbf{0.27}$**.
3. **Validation Source:** Threshold `0.27` was **not** selected via test-set data snooping or holdout metric optimization; it was derived analytically from the population base rate and the calibrated projection of raw $0.50$.

---

## 2. Distinction: Raw Random Forest Score vs. Calibrated Probability Thresholds

A critical finding of this audit is that threshold numbers cannot be compared across different probability scales:

| Dimension | Raw Random Forest Score ($p_{\text{raw}}$) | Platt-Calibrated Probability ($P_{\text{cal}}$) |
|---|---|---|
| **Mathematical Nature** | Ranking risk score / leaf sample proportion under balanced re-weighting ($w_1 = 2.76, w_0 = 1.0$). | Frequentist estimate of empirical churn probability: $P(Y=1 \mid X)$. |
| **Population Mean** | Artificially inflated: mean prediction is **$40.77\%$**. | Aligned to true reality: mean prediction is **$26.58\%$**. |
| **Meaning of 0.50** | The natural decision boundary of the balanced model (equal-loss boundary under synthetic balancing). | A customer with an empirical 1-in-2 chance of churning ($p_{\text{raw}} = 0.6834$). |
| **Effect of Applying Threshold 0.27** | **DISASTROUS MISCLASSIFICATION:** Flags **60.66%** of all customers ($3,412 / 5,625$), producing **2,017 false positives** and precision of only $40.89\%$. | **INTENDED BASE-RATE RULE:** Flags **39.72%** of customers ($2,234 / 5,625$), producing **1,043 false positives** and precision of $53.31\%$. |

**Key Guardrail:** If an engineer or dashboard user mistakenly applies threshold `0.27` to raw scores or threshold `0.50` to calibrated scores without understanding the scale, the intervention volume shifts by tens of percentage points.

---

## 3. Mathematical Inversion & Classification Parity

Because the logistic function is strictly monotonic, we can analytically invert the Platt calibrator:
$$P_{\text{cal}} = \sigma(w \cdot p_{\text{raw}} + b) \iff p_{\text{raw}} = \frac{\text{logit}(P_{\text{cal}}) - b}{w} = \frac{\ln\left(\frac{P_{\text{cal}}}{1 - P_{\text{cal}}}\right) - b}{w}$$

Given fitted parameters $w = 5.44697305$ and $b = -3.72222490$:
- For $P_{\text{cal}} = 0.2700$:
  $$p_{\text{raw}} = \frac{\text{logit}(0.27) - (-3.722225)}{5.446973} = \frac{-0.994623 + 3.722225}{5.446973} = \mathbf{0.500756}$$
- For $p_{\text{raw}} = 0.5000$:
  $$P_{\text{cal}} = \sigma(5.446973 \cdot 0.50 - 3.722225) = \sigma(-0.998738) = \mathbf{0.269190}$$
- For $p_{\text{raw}} = 0.5200$ (Phase 2 Optimal F1):
  $$P_{\text{cal}} = \sigma(5.446973 \cdot 0.52 - 3.722225) = \sigma(-0.889799) = \mathbf{0.291151}$$

### Empirical Classification Parity on Training OOF ($N = 5,625$):
- **Rule 1 vs Rule 2 ($p_{\text{raw}} \ge 0.5000$ vs $P_{\text{cal}} \ge 0.2692$):** **0 mismatches / 5,625 (100.0% IDENTICAL)**.
- **Rule 1 vs Rule 3 ($p_{\text{raw}} \ge 0.5000$ vs $P_{\text{cal}} \ge 0.2700$):** **5 mismatches / 5,625 (99.91% MATCH)**.  
  *(Only 5 borderline customers with $p_{\text{raw}} \in [0.5000, 0.500756]$ differ).*
- **Rule 4 vs Rule 5 ($p_{\text{raw}} \ge 0.5200$ vs $P_{\text{cal}} \ge 0.2912$):** **0 mismatches / 5,625 (100.0% IDENTICAL)**.

---

## 4. Training-Only Candidate Threshold Comparisons ($N = 5,625$)

To avoid using the previously exposed holdout test set, all candidate thresholds were evaluated strictly across the 5-fold out-of-fold cross-validation predictions of the training partition ($N=5,625$, $1,495$ churners):

| Strategy / Scenario | Calibrated Threshold ($t_{\text{cal}}$) | Mapped Raw Score ($p_{\text{raw}}$) | Recall | Precision | F1-Score | Accuracy | False Positives (FP) | False Negatives (FN) | Flagged Customers | Flagged % |
|---|---|---|---|---|---|---|---|---|---|---|
| **High Recall (Max Coverage)** | 0.1500 | 0.3649 | 89.43% | 45.95% | 0.6070 | 69.21% | 1,573 | 158 | 2,910 | 51.73% |
| **High Recall / Retention Focused** | 0.2000 | 0.4288 | 85.42% | 49.29% | 0.6251 | 72.80% | 1,314 | 218 | 2,591 | 46.06% |
| **Moderate Retention** | 0.2500 | 0.4817 | 81.07% | 52.15% | 0.6347 | 75.20% | 1,112 | 283 | 2,324 | 41.32% |
| **Prevalence-Aligned (Exact Base Rate)** | 0.2658 | 0.4968 | 80.40% | 53.21% | 0.6404 | 76.00% | 1,057 | 293 | 2,259 | 40.16% |
| **Exact Raw 0.50 Equivalent** | 0.2692 | 0.5000 | 79.80% | 53.28% | 0.6390 | 76.04% | 1,046 | 302 | 2,239 | 39.80% |
| **Wrapper Default (Rounded)** | **0.2700** | **0.5008** | **79.67%** | **53.31%** | **0.6388** | **76.05%** | **1,043** | **304** | **2,234** | **39.72%** |
| **Phase 2 Maximum F1 Equivalent** | 0.2912 | 0.5200 | 77.99% | 54.41% | **0.6410** | 76.85% | 977 | 329 | 2,143 | 38.10% |
| **Balanced Precision/Recall** | 0.3500 | 0.5697 | 71.51% | 57.29% | 0.6361 | 78.31% | 797 | 426 | 1,866 | 33.17% |
| **Precision-Prioritized (Capacity Limit)** | 0.4000 | 0.6089 | 66.15% | 59.72% | 0.6277 | 79.16% | 667 | 506 | 1,656 | 29.44% |
| **High Precision (Expensive Outreach)** | 0.5000 | 0.6834 | 56.25% | 64.94% | 0.6029 | 80.30% | 454 | 654 | 1,295 | 23.02% |

---

## 5. Trade-Off Analysis Without Inventing Arbitrary Cost Functions

Because real-world financial cost per false positive (cost of retention discount/call) vs. cost per false negative (lost customer LTV) has not been provided by business leadership, we present the non-parametric trade-offs:

1. **At $t_{\text{cal}} = 0.27$ (Current Default):**
   - **Trade-off:** Captures nearly $80\%$ of all churners ($1,191 / 1,495$), but flags $39.72\%$ of all customers. For every $100$ customers flagged, $\approx 53$ are actual churners and $\approx 47$ would not have churned.
   - **Best suited for:** Low-cost interventions (automated emails, in-app retention prompts, survey outreach).

2. **At $t_{\text{cal}} = 0.2912$ (Maximum F1):**
   - **Trade-off:** Captures $78.0\%$ of churners while trimming $91$ false positives ($1,043 \to 977$), improving precision from $53.3\% \to 54.4\%$.
   - **Best suited for:** Operational teams seeking the mathematical harmonic balance between recall and precision.

3. **At $t_{\text{cal}} = 0.40$ (Capacity-Constrained):**
   - **Trade-off:** Cuts flagged volume down to $29.4\%$ (saving $376$ false positive interventions compared to $0.27$), but misses $506$ churners (Recall drops to $66.15\%$).
   - **Best suited for:** High-touch manual call campaigns where customer support capacity is limited.

---

## 6. Recommendations & Implementation Status

### 6.1 Recommendation on Default Threshold
1. **Retain $t_{\text{cal}} = 0.27$ as the Default Decision Threshold:**
   - It is mathematically justifiable: directly corresponds to the balanced model's $0.50$ baseline that was benchmarked and approved in Phase 2.
   - It captures $79.7\%$ of churners with $53.3\%$ precision.
   - Changing to $0.2912$ provides a tiny $+0.2\%$ F1 gain but drops recall by $1.7\%$. In telecom churn, missing churners is generally considered more expensive than sending proactive retention offers.
2. **Mandate Threshold Configurability:**
   - The wrapper already exposes `predict(X, threshold=...)`. The API or downstream service should accept an optional `threshold` query parameter so marketing teams can target different operational tiers:
     - Tier 1 (High Risk / VIP): $P_{\text{cal}} \ge 0.40$ (High Precision)
     - Tier 2 (Standard Risk): $P_{\text{cal}} \ge 0.27$ (Prevalence-Aligned Default)
     - Tier 3 (Proactive Outreach): $P_{\text{cal}} \ge 0.20$ (High Recall)

### 6.2 Files Created in `model/experiments/`
- [`model/experiments/run_threshold_audit.py`](file:///f:/Projects/Customer%20Churn%20Prediction/customer-churn-prediction/model/experiments/run_threshold_audit.py): Reproducible audit script.
- [`model/experiments/calibrated_threshold_candidates.csv`](file:///f:/Projects/Customer%20Churn%20Prediction/customer-churn-prediction/model/experiments/calibrated_threshold_candidates.csv): Complete candidate trade-off matrix.
- [`model/experiments/threshold_audit_results.json`](file:///f:/Projects/Customer%20Churn%20Prediction/customer-churn-prediction/model/experiments/threshold_audit_results.json): Structured audit parameters and parity checks.
- [`model/experiments/threshold_audit_report.md`](file:///f:/Projects/Customer%20Churn%20Prediction/customer-churn-prediction/model/experiments/threshold_audit_report.md): Formal markdown report.

---

*This audit is complete. Production files and APIs remain untouched. Stopping and awaiting user approval.*
