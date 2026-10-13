"""
Calibrated Model Pipeline Wrapper
Customer Churn Prediction & Retention Intelligence System

Provides a modular, production-ready inference wrapper that pairs a base
feature-engineering + classifier pipeline with a post-hoc probability calibrator
(Platt scaling / Sigmoid or Isotonic Regression).

Key Architectural Decoupling:
- predict_raw_proba(X): Exposes uncalibrated base-model tree probabilities,
  enabling exact SHAP TreeExplainer additivity (base_value + sum(shap) == raw_p).
- predict_proba(X): Exposes calibrated frequentist probabilities for customer-facing
  risk communication, financial expected-value calculations, and executive dashboards.
- predict(X, threshold): Applies decision threshold (default 0.27 aligned to population base rate).

This module contains NO executable training, evaluation, plotting, or file I/O logic.
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd


class CalibratedPipelineWrapper:
    """
    Unified inference wrapper that decouples uncalibrated tree scores (for SHAP)
    from calibrated probabilities (for frequentist risk & financial analysis).

    Parameters
    ----------
    base_pipeline : Any
        Fitted Scikit-Learn Pipeline or estimator providing `predict_proba`.
    calibrator : Any
        Fitted calibrator (e.g., LogisticRegression for Platt scaling or IsotonicRegression).
    method : str, default="sigmoid"
        Calibration method name ("sigmoid" or "isotonic").
    threshold : float, default=0.27
        Operational decision threshold on the calibrated probability scale.
    """

    def __init__(
        self,
        base_pipeline: Any,
        calibrator: Any,
        method: str = "sigmoid",
        threshold: float = 0.27,
    ):
        self.base_pipeline = base_pipeline
        self.calibrator = calibrator
        self.method = method.lower()
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_raw_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return the 1D uncalibrated probability for the positive class (Churn = 1).

        This score directly preserves the underlying decision-tree split logic
        and is required for additive SHAP TreeExplainer local explanations.
        """
        raw_proba_2d = self.base_pipeline.predict_proba(X)
        return raw_proba_2d[:, 1]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return the 2D calibrated probability array of shape (n_samples, 2).
        Column 0: P(Churn = 0)
        Column 1: P(Churn = 1) (Calibrated)
        """
        raw_p = self.predict_raw_proba(X)

        if self.method == "sigmoid":
            # Platt scaling: Logistic regression on raw probability feature
            raw_features = raw_p.reshape(-1, 1)
            cal_p = self.calibrator.predict_proba(raw_features)[:, 1]
        elif self.method == "isotonic":
            # Isotonic regression: Piecewise constant monotonic mapping
            cal_p = self.calibrator.predict(raw_p)
        else:
            raise ValueError(f"Unsupported calibration method: '{self.method}'. Expected 'sigmoid' or 'isotonic'.")

        # Clip for numerical stability within [0.0, 1.0]
        cal_p = np.clip(cal_p, 0.0, 1.0)
        return np.column_stack([1.0 - cal_p, cal_p])

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict binary churn class (0 or 1) at the specified decision threshold.
        If threshold is None, self.threshold (default 0.27) is used.
        """
        eff_threshold = threshold if threshold is not None else self.threshold
        p_cal = self.predict_proba(X)[:, 1]
        return (p_cal >= eff_threshold).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        """Return parameters for Scikit-Learn compatibility."""
        return {
            "base_pipeline": self.base_pipeline,
            "calibrator": self.calibrator,
            "method": self.method,
            "threshold": self.threshold,
        }

    def __repr__(self) -> str:
        return (
            f"CalibratedPipelineWrapper("
            f"base_pipeline={self.base_pipeline.__class__.__name__}, "
            f"calibrator={self.calibrator.__class__.__name__}, "
            f"method='{self.method}', "
            f"threshold={self.threshold})"
        )
