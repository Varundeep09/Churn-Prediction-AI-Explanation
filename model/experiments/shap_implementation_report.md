# Phase 3A: SHAP Customer-Level Explainability Implementation Report

**Module:** `model/explain.py`  
**Explainer Backend:** `shap.TreeExplainer` (v0.53.0)  
**Target Pipeline:** `model/experiments/selected_model_pipeline.pkl` (Balanced Random Forest)  
**Execution Date:** October 13, 2026  

---

## 1. Files Added & Modified

| File | Type | Changes / Purpose |
|---|---|---|
| `requirements.txt` | Modified | Appended `shap>=0.45.0` to dependencies. |
| `model/explain.py` | Added (New) | Production-ready customer explanation service with `CustomerExplainer` class and `explain_customer()` helper. |
| `model/experiments/validate_shap.py` | Added (New) | Validation script running 4 representative customer profiles with mathematical additivity assertions. |
| `model/experiments/shap_validation_results.json` | Added (New) | Serialized structured JSON output from the 4 validated test customer profiles. |
| `model/experiments/shap_implementation_report.md` | Added (New) | Technical report documenting explainability design, validation, and limitations. |

*Original production artifacts (`churn_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `model_info.json`) remain untouched.*

---

## 2. SHAP Version & Explainer Architecture

- **Library Version:** `shap==0.53.0`
- **Explainer Architecture:** `shap.TreeExplainer(model.named_steps['classifier'])`
- **Execution Mechanism:**
  1. Incoming raw customer features (dict or DataFrame) pass through the fitted `ColumnTransformer` (`model.named_steps['preprocessor']`).
  2. The transformed 30-feature vector is passed to `explainer(X_transformed)`.
  3. `TreeExplainer` leverages the exact tree paths of the 200 estimators in the Balanced Random Forest, computing local Shapley values in **$< 8\text{ ms}$**.

---

## 3. Churn Class Identification & Output Scale

- In binary classification for Scikit-Learn `RandomForestClassifier`, SHAP returns an explanation object of shape `(n_samples, n_features, 2)`:
  - Slice `[:, :, 0]`: Non-churn class ($y = 0$).
  - Slice `[:, :, 1]`: **Churn class ($y = 1$)**.
- `model/explain.py` explicitly targets the positive churn class by inspecting `explainer.expected_value[1]`.
- **Output Scale:** The explanation operates on the **raw probability scale** ($0.0 \to 1.0$), satisfying:
  $$\hat{P}(\text{Churn} = 1) = \text{base\_value} + \sum_{i=1}^{30} \phi_i$$
- **Base Value:** $\text{base\_value} = 0.499847$ (the prior expected value under balanced class weighting).

---

## 4. Feature Name Mapping

Encoded column names from `preprocessor.get_feature_names_out()` are mapped to human-readable labels via `FEATURE_DISPLAY_MAP`:
- `num__tenure` $\to$ **Customer Tenure (Months)**
- `num__MonthlyCharges` $\to$ **Monthly Charges ($)**
- `cat__InternetService_Fiber optic` $\to$ **Internet Service: Fiber Optic**
- `cat__Contract_Two year` $\to$ **Contract: Two-Year Commitment**
- `cat__PaymentMethod_Electronic check` $\to$ **Payment Method: Electronic Check**
- `cat__OnlineSecurity_Yes` $\to$ **Online Security: Active**
- `cat__TechSupport_Yes` $\to$ **Tech Support: Active**

---

## 5. Mathematical Validation & Additivity Checks

Across all 4 test profiles, mathematical additivity was validated:
$$\left| \left( \text{base\_value} + \sum_{i=1}^{30} \phi_i \right) - \text{model.predict\_proba}(X)[:, 1] \right| < 10^{-14}$$

### Validation Summary Across Profiles:
1. **High-Risk Profile (Female, Senior, Month-to-month, Fiber optic, tenure=2m):**
   - Predicted Probability: **90.0%** (Risk: High)
   - Lead Risk Driver: Short Tenure (+8.03%), Fiber Optic (+4.50%), Absence of Two-Year Contract (+3.61%).
   - Additivity Discrepancy: $0.00 \times 10^{-15}$ (Exact).
2. **Low-Risk Profile (Male, tenure=68m, Two-year contract, full security/support):**
   - Predicted Probability: **6.0%** (Risk: Low)
   - Lead Protective Factor: Two-Year Contract (-12.73%), Long Tenure (-8.35%), Absence of Fiber Optic (-5.21%).
   - Additivity Discrepancy: $0.00 \times 10^{-15}$ (Exact).
3. **Categorical Validation Profile (24m tenure, DSL, One-year contract, Bank transfer):**
   - Predicted Probability: **28.1%** (Risk: Low)
   - Lead Protective Factor: One-Year Contract (-9.54%), Absence of Fiber Optic (-5.46%), Active Online Security (-4.86%).
   - Additivity Discrepancy: $0.00 \times 10^{-15}$ (Exact).
4. **Edge-Case Profile (tenure=0, minimum charges $18.25, missing TotalCharges fallback):**
   - Predicted Probability: **42.1%** (Risk: Medium)
   - Handled `tenure=0` and calculated `TotalCharges = 0.0` gracefully.
   - Additivity Discrepancy: $0.00 \times 10^{-15}$ (Exact).

---

## 6. Runtime Performance & Production Readiness

- **Initialization Overhead:** Creating `CustomerExplainer()` takes $\sim 150\text{ ms}$ (loading the 4.3 MB pipeline and initializing the tree structure).
- **Inference Latency:** Generating full customer predictions + 30-feature SHAP attributions + sorting top drivers takes **$< 10\text{ ms}$ per call**.
- **Singleton Caching:** `get_explainer()` provides a pre-warmed singleton instance, ensuring zero per-request disk I/O when integrated into FastAPI.

---

## 7. Known Limitations & Scientific Disclaimers

1. **Correlation vs. Causation:** SHAP values represent the internal feature weights of the decision trees. Having a positive attribution on `PaymentMethod: Electronic check` does **not** prove that changing payment method causes a customer to stay.
2. **Probability Shift:** Because balanced class weighting was applied, the base value is $\approx 0.50$ (rather than the unweighted base rate of $0.2658$). Feature contributions explain deviations around this balanced decision threshold.

---

## 8. Instructions for Manual Verification

Run the verification suite from terminal:
```powershell
.\venv\Scripts\python.exe model/experiments/validate_shap.py
```

Or generate an explanation programmatically in Python:
```python
from model.explain import explain_customer

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.00,
    "TotalCharges": 255.00,
}

explanation = explain_customer(customer)
print(explanation["churn_probability"])
print(explanation["top_risk_drivers"])
```
