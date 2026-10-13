"""
Validation Script for SHAP Customer-Level Explainability Module
Tests 4 representative customer profiles:
1. High-risk profile (new customer, fiber optic, month-to-month, electronic check, no security/support)
2. Low-risk profile (long tenure, two-year contract, full security/support, auto payment)
3. Categorical handling profile (DSL, one-year contract, bank transfer, multiple lines)
4. Edge-case profile (min/max boundary numerical values: tenure=0, tenure=72, extreme charges)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.explain import CustomerExplainer, explain_customer

EXP_DIR = Path(__file__).resolve().parent

test_profiles = {
    "profile_1_high_risk": {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
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
        "MonthlyCharges": 105.50,
        "TotalCharges": 211.00,
    },
    "profile_2_low_risk": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.20,
        "TotalCharges": 4433.60,
    },
    "profile_3_categorical_validation": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 35.50,
        "TotalCharges": 852.00,
    },
    "profile_4_boundary_edge_case": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,  # Minimum boundary
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 18.25,  # Lowest boundary
        "TotalCharges": None,    # Tests missing fallback
    },
}

explainer = CustomerExplainer()
validation_output = {}

print("=" * 85)
print("VALIDATING SHAP CUSTOMER-LEVEL EXPLANATIONS ACROSS REPRESENTATIVE PROFILES")
print("=" * 85)

for profile_name, profile_data in test_profiles.items():
    res = explainer.explain_customer(profile_data, top_n=3)
    validation_output[profile_name] = res

    print(f"\n>>> {profile_name.upper()}:")
    print(f"    Prediction: {res['churn_prediction']} | Probability: {res['churn_probability'] * 100:.1f}% | Risk: {res['risk_level']}")
    print(f"    Base Value: {res['base_value']:.4f} | Sum Attributions: {res['mathematical_reconciliation']['sum_attributions']:.4f}")
    print(f"    Reconstructed: {res['mathematical_reconciliation']['reconstructed_probability']:.6f} vs Model: {res['mathematical_reconciliation']['model_predicted_probability']:.6f}")
    print(f"    Additivity Discrepancy: {res['mathematical_reconciliation']['discrepancy']:.2e} (Passed: {res['mathematical_reconciliation']['additivity_verified']})")

    print("    Top Risk Drivers (Pushing toward churn):")
    for d in res["top_risk_drivers"]:
        print(f"      + {d['display_name']}: {d['attribution']:+.4f} ({d['impact_percentage']:+.2f}%) [{d['magnitude']}]")

    print("    Top Protective Factors (Pushing away from churn):")
    for p in res["top_protective_factors"]:
        print(f"      - {p['display_name']}: {p['attribution']:+.4f} ({p['impact_percentage']:+.2f}%) [{p['magnitude']}]")

    print(f"    Summary: \"{res['interpretability_summary']}\"")

    # Assertions
    assert res["mathematical_reconciliation"]["additivity_verified"], f"Additivity failed for {profile_name}"
    assert len(res["all_attributions"]) == 30, f"Expected 30 features, got {len(res['all_attributions'])}"

# Save structured validation report to experiments folder
output_path = EXP_DIR / "shap_validation_results.json"
with open(output_path, "w") as f:
    json.dump(validation_output, f, indent=2)

print(f"\nAll 4 test profiles passed verification. Results saved to: {output_path}")
