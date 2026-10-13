"""
SHAP-Based Customer-Level Explainability Module
Customer Churn Prediction & Retention Intelligence System

Provides local, customer-level feature attributions using shap.TreeExplainer
on the fitted Balanced Random Forest pipeline (selected_model_pipeline.pkl).

Key Properties:
- Uses the fitted ColumnTransformer from the selected pipeline.
- Explains the positive class (Churn = 1) on the exact probability scale.
- Strictly satisfies the SHAP efficiency / additivity property:
  base_value + sum(attributions) == predicted_churn_probability
- Returns clean, JSON-serializable structured explanations.
- Distinguishes statistical model attribution from real-world causality.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap

# Paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_PIPELINE_PATH = MODULE_DIR / "experiments" / "selected_model_pipeline.pkl"

# Human-readable feature naming map
FEATURE_DISPLAY_MAP = {
    "num__tenure": "Customer Tenure (Months)",
    "num__MonthlyCharges": "Monthly Charges ($)",
    "num__TotalCharges": "Total Billed Charges ($)",
    "cat__gender_Male": "Gender: Male",
    "cat__SeniorCitizen_1": "Senior Citizen Status: Yes",
    "cat__Partner_Yes": "Has Partner: Yes",
    "cat__Dependents_Yes": "Has Dependents: Yes",
    "cat__PhoneService_Yes": "Phone Service: Subscribed",
    "cat__MultipleLines_No phone service": "Multiple Lines: No Phone Service",
    "cat__MultipleLines_Yes": "Multiple Lines: Multiple Lines Active",
    "cat__InternetService_Fiber optic": "Internet Service: Fiber Optic",
    "cat__InternetService_No": "Internet Service: None",
    "cat__OnlineSecurity_No internet service": "Online Security: No Internet Service",
    "cat__OnlineSecurity_Yes": "Online Security: Active",
    "cat__OnlineBackup_No internet service": "Online Backup: No Internet Service",
    "cat__OnlineBackup_Yes": "Online Backup: Active",
    "cat__DeviceProtection_No internet service": "Device Protection: No Internet Service",
    "cat__DeviceProtection_Yes": "Device Protection: Active",
    "cat__TechSupport_No internet service": "Tech Support: No Internet Service",
    "cat__TechSupport_Yes": "Tech Support: Active",
    "cat__StreamingTV_No internet service": "Streaming TV: No Internet Service",
    "cat__StreamingTV_Yes": "Streaming TV: Active",
    "cat__StreamingMovies_No internet service": "Streaming Movies: No Internet Service",
    "cat__StreamingMovies_Yes": "Streaming Movies: Active",
    "cat__Contract_One year": "Contract: One-Year Commitment",
    "cat__Contract_Two year": "Contract: Two-Year Commitment",
    "cat__PaperlessBilling_Yes": "Paperless Billing: Active",
    "cat__PaymentMethod_Credit card (automatic)": "Payment Method: Automatic Credit Card",
    "cat__PaymentMethod_Electronic check": "Payment Method: Electronic Check",
    "cat__PaymentMethod_Mailed check": "Payment Method: Mailed Check",
}


def _get_magnitude_tier(val: float) -> str:
    """Categorize absolute SHAP attribution into human-readable impact tiers."""
    abs_v = abs(val)
    if abs_v >= 0.08:
        return "HIGH"
    elif abs_v >= 0.03:
        return "MODERATE"
    elif abs_v >= 0.01:
        return "LOW"
    return "NEGLIGIBLE"


class CustomerExplainer:
    """
    Production-ready customer explanation service using SHAP TreeExplainer.
    Loads and caches the selected pipeline and precomputes the TreeExplainer.
    """

    def __init__(self, pipeline_path: Optional[Union[str, Path]] = None, threshold: float = 0.50):
        self.pipeline_path = Path(pipeline_path) if pipeline_path else DEFAULT_PIPELINE_PATH
        self.threshold = threshold

        if not self.pipeline_path.exists():
            raise FileNotFoundError(
                f"Selected model pipeline not found at {self.pipeline_path}. "
                "Ensure Phase 2 benchmarking has been executed."
            )

        # Load fitted pipeline
        self.pipeline = joblib.load(self.pipeline_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        # Cache encoded feature names from preprocessor
        self.feature_names: List[str] = list(self.preprocessor.get_feature_names_out())

        # Initialize SHAP TreeExplainer once on the fitted tree classifier
        self.explainer = shap.TreeExplainer(self.classifier)

        # Identify positive class (churn = 1) expected value
        # In multi-class/binary RandomForestClassifier, expected_value is array [ev_0, ev_1]
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
            self.base_value_churn = float(ev[1])
            self.churn_class_idx = 1
        else:
            self.base_value_churn = float(ev)
            self.churn_class_idx = 0

    def _prepare_dataframe(self, customer_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Standardize raw customer inputs into a valid pandas DataFrame."""
        if isinstance(customer_data, dict):
            raw = dict(customer_data)
            # Default TotalCharges if missing
            if raw.get("TotalCharges") is None:
                tenure = float(raw.get("tenure", 0))
                monthly = float(raw.get("MonthlyCharges", 0.0))
                raw["TotalCharges"] = tenure * monthly
            df = pd.DataFrame([raw])
        elif isinstance(customer_data, pd.DataFrame):
            df = customer_data.copy()
            if "TotalCharges" in df.columns:
                null_mask = df["TotalCharges"].isna()
                if null_mask.any():
                    df.loc[null_mask, "TotalCharges"] = (
                        df.loc[null_mask, "tenure"] * df.loc[null_mask, "MonthlyCharges"]
                    )
        else:
            raise TypeError("customer_data must be a dict or a pandas DataFrame.")

        # Clean customerID if passed
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])

        return df

    def explain_customer(
        self,
        customer_data: Union[Dict[str, Any], pd.DataFrame],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate a structured, JSON-serializable customer churn explanation.

        Args:
            customer_data: Single customer attributes as dict or DataFrame row.
            top_n: Number of top risk drivers and protective factors to extract.

        Returns:
            Dictionary containing model predictions, top contributors,
            mathematical reconciliation, and attribution descriptions.
        """
        df_input = self._prepare_dataframe(customer_data)
        if len(df_input) != 1:
            raise ValueError(f"explain_customer expects a single customer record, got {len(df_input)} rows.")

        # 1. Transform raw customer features via fitted preprocessor
        X_trans = self.preprocessor.transform(df_input)

        # 2. Compute model predictions
        predict_proba_arr = self.classifier.predict_proba(X_trans)
        churn_prob = float(predict_proba_arr[0, 1])
        prediction = "Yes" if churn_prob >= self.threshold else "No"

        # Risk tier matching project convention
        if churn_prob >= 0.66:
            risk_tier = "High"
        elif churn_prob >= 0.33:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        # 3. Compute SHAP values via TreeExplainer
        shap_explanation = self.explainer(X_trans)

        # Extract churn class (index 1) attributions
        if len(shap_explanation.shape) == 3:
            # Shape: (1, n_features, 2)
            shap_values_churn = shap_explanation.values[0, :, self.churn_class_idx]
        else:
            # Shape: (1, n_features)
            shap_values_churn = shap_explanation.values[0, :]

        # 4. Mathematical Additivity Check
        sum_attributions = float(np.sum(shap_values_churn))
        reconstructed_prob = float(self.base_value_churn + sum_attributions)
        discrepancy = float(abs(reconstructed_prob - churn_prob))

        # 5. Build structured feature attribution list
        feature_attributions: List[Dict[str, Any]] = []
        for feat_name, shap_val, input_val in zip(self.feature_names, shap_values_churn, X_trans[0]):
            display_name = FEATURE_DISPLAY_MAP.get(feat_name, feat_name)
            val_float = float(shap_val)
            direction = "INCREASES_CHURN_RISK" if val_float > 0 else "DECREASES_CHURN_RISK"
            tier = _get_magnitude_tier(val_float)

            feature_attributions.append({
                "feature_name": feat_name,
                "display_name": display_name,
                "encoded_value": float(input_val),
                "attribution": round(val_float, 4),
                "direction": direction,
                "magnitude": tier,
                "impact_percentage": round(val_float * 100, 2),
            })

        # Separate risk drivers (positive SHAP) vs protective factors (negative SHAP)
        risk_drivers = sorted(
            [f for f in feature_attributions if f["attribution"] > 0],
            key=lambda x: x["attribution"],
            reverse=True,
        )
        protective_factors = sorted(
            [f for f in feature_attributions if f["attribution"] < 0],
            key=lambda x: x["attribution"],
        )

        top_drivers = risk_drivers[:top_n]
        top_protective = protective_factors[:top_n]

        # 6. Generate non-causal interpretability summary
        summary_lines = []
        if top_drivers:
            lead_driver = top_drivers[0]
            summary_lines.append(
                f"Primary factor elevating churn risk is {lead_driver['display_name']} "
                f"(+{lead_driver['impact_percentage']}% probability contribution)."
            )
        if top_protective:
            lead_prot = top_protective[0]
            summary_lines.append(
                f"Strongest protective factor against churn is {lead_prot['display_name']} "
                f"({lead_prot['impact_percentage']}% probability contribution)."
            )

        plain_summary = " ".join(summary_lines) if summary_lines else "Customer attributes balance near baseline risk."

        return {
            "churn_prediction": prediction,
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_tier,
            "decision_threshold": self.threshold,
            "base_value": round(self.base_value_churn, 4),
            "score_scale": "probability",
            "mathematical_reconciliation": {
                "base_value": round(self.base_value_churn, 6),
                "sum_attributions": round(sum_attributions, 6),
                "reconstructed_probability": round(reconstructed_prob, 6),
                "model_predicted_probability": round(churn_prob, 6),
                "discrepancy": round(discrepancy, 12),
                "additivity_verified": bool(discrepancy < 1e-5),
            },
            "top_risk_drivers": top_drivers,
            "top_protective_factors": top_protective,
            "all_attributions": feature_attributions,
            "interpretability_summary": plain_summary,
            "scientific_disclaimer": (
                "SHAP values quantify statistical feature importance assigned by the "
                "Random Forest model and do not constitute proven causal mechanisms."
            ),
        }


# Singleton helper instance for module-level usage
_cached_explainer: Optional[CustomerExplainer] = None


def get_explainer() -> CustomerExplainer:
    """Return a cached singleton instance of CustomerExplainer."""
    global _cached_explainer
    if _cached_explainer is None:
        _cached_explainer = CustomerExplainer()
    return _cached_explainer


def explain_customer(customer_data: Union[Dict[str, Any], pd.DataFrame], top_n: int = 5) -> Dict[str, Any]:
    """Convenience helper function for generating single-customer SHAP explanations."""
    explainer = get_explainer()
    return explainer.explain_customer(customer_data=customer_data, top_n=top_n)
