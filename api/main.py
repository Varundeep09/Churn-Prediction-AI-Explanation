from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import math
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.calibrated_pipeline import CalibratedPipelineWrapper
from model.explain import CustomerExplainer, get_explainer
from model.genai_explainer import (
    GeminiRetentionExplainer,
    RetentionExplanation,
    GenAIConfigurationError,
    GenAIProviderError,
    GenAIValidationError,
)

# Logger setup
logger = logging.getLogger("ChurnAPI")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# LOAD PRODUCTION MODEL ARTIFACTS
# ---------------------------------------------------------------------------
model_dir = project_root / "model"
experiments_dir = model_dir / "experiments"

# Baseline Production Model: Logistic Regression
model = joblib.load(model_dir / "churn_model.pkl")
scaler = joblib.load(model_dir / "scaler.pkl")
feature_columns = joblib.load(model_dir / "feature_columns.pkl")

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

# ---------------------------------------------------------------------------
# LAZY SINGLETONS FOR EXPERIMENTAL ARTIFACTS & GENAI
# ---------------------------------------------------------------------------
_calibrated_pipeline: Optional[CalibratedPipelineWrapper] = None
_explainer_service: Optional[GeminiRetentionExplainer] = None


def get_calibrated_model() -> CalibratedPipelineWrapper:
    """Return cached CalibratedPipelineWrapper for experimental inference."""
    global _calibrated_pipeline
    if _calibrated_pipeline is None:
        pipe_path = experiments_dir / "calibrated_pipeline_refactored.pkl"
        if not pipe_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experimental calibrated pipeline artifact is not available.",
            )
        try:
            _calibrated_pipeline = joblib.load(pipe_path)
            # Verify interface
            if not hasattr(_calibrated_pipeline, "predict_raw_proba") or not hasattr(_calibrated_pipeline, "predict_proba"):
                raise TypeError("Loaded pipeline does not implement CalibratedPipelineWrapper interface.")
        except Exception as e:
            logger.error("Failed to load experimental calibrated pipeline: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to load experimental calibrated model pipeline.",
            ) from None
    return _calibrated_pipeline


def get_gemini_service() -> GeminiRetentionExplainer:
    """Return cached GeminiRetentionExplainer service instance."""
    global _explainer_service
    if _explainer_service is None:
        _explainer_service = GeminiRetentionExplainer(tier_convention="calibrated")
    return _explainer_service


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["Yes"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[70.35])
    TotalCharges: Optional[float] = Field(default=None, examples=[845.25])


class ExplainRequest(CustomerData):
    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer identifier for tracking.",
        examples=["CUST-10492"],
    )
    threshold: Optional[float] = Field(
        default=0.27,
        description="Operational decision boundary on calibrated scale [0.0, 1.0]. Defaults to 0.27.",
        examples=[0.27],
    )
    allow_fallback: Optional[bool] = Field(
        default=True,
        description="If True, returns a safe deterministic explanation if Gemini API is unavailable.",
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold_override(cls, v: Optional[float]) -> float:
        if v is None:
            return 0.27
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Operational threshold must be a finite float.")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Operational threshold must be bounded in [0.0, 1.0], got {v}.")
        return round(float(v), 4)


class FeatureAttributionItem(BaseModel):
    feature_name: str
    display_name: str
    attribution: float
    magnitude: str
    impact_percentage: float


class RecommendedActionItem(BaseModel):
    action: str
    rationale: str


class ExplainResponse(BaseModel):
    customer_id: Optional[str] = None
    raw_model_score: float = Field(
        ...,
        description="Uncalibrated Random Forest base score [0.0, 1.0].",
    )
    calibrated_churn_probability: float = Field(
        ...,
        description="Platt-calibrated frequentist churn probability [0.0, 1.0].",
    )
    operational_threshold: float = Field(
        ...,
        description="Operational decision threshold on calibrated probability scale.",
    )
    risk_classification: str = Field(
        ...,
        description="Assigned risk tier: 'Low', 'Medium', or 'High'.",
    )
    shap_risk_drivers: List[FeatureAttributionItem] = Field(
        ...,
        description="Top observational factors elevating churn risk from SHAP analysis.",
    )
    shap_protective_factors: List[FeatureAttributionItem] = Field(
        ...,
        description="Top observational factors mitigating churn risk from SHAP analysis.",
    )
    explanation: str = Field(
        ...,
        description="Plain-language retention brief generated by Google Gemini.",
    )
    recommended_actions: List[RecommendedActionItem] = Field(
        ...,
        description="Evidence-grounded retention initiatives tailored to model drivers.",
    )
    disclaimer: str = Field(
        ...,
        description="Non-causal statistical interpretability disclaimer.",
    )
    fallback_used: bool = Field(
        default=False,
        description="Indicates whether a deterministic rule fallback was used instead of live Gemini generation.",
    )
    model_version: Optional[str] = Field(
        None,
        description="Underlying LLM or fallback version identifier.",
    )


# ---------------------------------------------------------------------------
# FASTAPI APP DEFINITION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction & Retention Intelligence API",
    version="1.1.0",
    description=(
        "Production API serving baseline Logistic Regression churn predictions "
        "and experimental GenAI retention explanations powered by Platt-calibrated Random Forest, "
        "SHAP TreeExplainer, and Google Gemini."
    ),
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# PRODUCTION HELPER FUNCTIONS (PRESERVED)
# ---------------------------------------------------------------------------
def prepare_input_data(customer: CustomerData) -> pd.DataFrame:
    raw_data = customer.model_dump()
    if raw_data["TotalCharges"] is None:
        raw_data["TotalCharges"] = raw_data["tenure"] * raw_data["MonthlyCharges"]

    input_df = pd.DataFrame([raw_data])
    encoded_df = pd.get_dummies(input_df, drop_first=False)
    aligned_df = encoded_df.reindex(columns=feature_columns, fill_value=0)
    aligned_df = aligned_df.astype({column: float for column in numeric_columns})
    aligned_df.loc[:, numeric_columns] = scaler.transform(aligned_df[numeric_columns])
    return aligned_df


def get_risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "name": "Customer Churn Prediction & Retention Intelligence API",
        "version": "1.1.0",
        "description": "FastAPI service for predicting and explaining telecom customer churn.",
        "endpoints": {
            "/predict": "Production baseline Logistic Regression prediction",
            "/explain": "Experimental calibrated Random Forest, SHAP, and Gemini retention explanation",
            "/health": "Service health check",
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "production_model": "LogisticRegression",
        "experimental_pipeline": "CalibratedPipelineWrapper (Random Forest + Platt Scaling)",
        "genai_explainer": "GeminiRetentionExplainer",
    }


@app.post("/predict")
def predict(customer: CustomerData) -> dict:
    """Production baseline Logistic Regression prediction endpoint (unchanged)."""
    prepared_df = prepare_input_data(customer)
    prediction = model.predict(prepared_df)[0]
    probability = float(model.predict_proba(prepared_df)[0][1])

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "risk_level": get_risk_level(probability),
    }


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Experimental GenAI retention explanation endpoint.
    Combines calibrated Random Forest probabilities, SHAP TreeExplainer local attributions,
    and Google Gemini structured retention briefs.
    """
    # 1. Extract customer attributes
    customer_dict = request.model_dump()
    customer_id = customer_dict.pop("customer_id", None)
    threshold = customer_dict.pop("threshold", 0.27)
    if threshold is None:
        threshold = 0.27
    allow_fallback = customer_dict.pop("allow_fallback", True)

    # 2. Estimate TotalCharges if missing
    if customer_dict.get("TotalCharges") is None:
        customer_dict["TotalCharges"] = float(customer_dict["tenure"]) * float(customer_dict["MonthlyCharges"])

    input_df = pd.DataFrame([customer_dict])

    # 3. Load experimental calibrated pipeline
    pipeline = get_calibrated_model()

    # 4. Load SHAP customer explainer
    try:
        explainer = get_explainer()
    except Exception as e:
        logger.error("Failed to initialize SHAP CustomerExplainer: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainability engine is currently unavailable.",
        ) from None

    # 5. Predict calibrated probability and uncalibrated raw score
    try:
        cal_prob = float(pipeline.predict_proba(input_df)[0, 1])
        raw_score = float(pipeline.predict_raw_proba(input_df)[0])
    except Exception as e:
        logger.error("Experimental pipeline inference error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate predictions from experimental calibrated pipeline.",
        ) from None

    # 6. Compute exact customer-level SHAP attributions
    try:
        shap_res = explainer.explain_customer(input_df, top_n=5)
        top_risk_drivers = shap_res.get("top_risk_drivers", [])
        top_protective = shap_res.get("top_protective_factors", [])
    except Exception as e:
        logger.error("SHAP attribution computation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute SHAP feature attributions.",
        ) from None

    # 7. Generate Gemini structured explanation
    gemini_service = get_gemini_service()
    explanation_obj: Optional[RetentionExplanation] = None
    fallback_used = False

    try:
        explanation_obj = gemini_service.explain(
            customer_features=customer_dict,
            calibrated_prob=cal_prob,
            raw_score=raw_score,
            threshold=threshold,
            risk_drivers=top_risk_drivers,
            protective_factors=top_protective,
            customer_id=customer_id,
            allow_fallback=allow_fallback,
        )
        if explanation_obj.model_version == "deterministic-fallback":
            fallback_used = True

    except GenAIConfigurationError:
        if allow_fallback:
            fallback_used = True
            explanation_obj = gemini_service.build_deterministic_fallback(
                customer_features=customer_dict,
                calibrated_prob=cal_prob,
                raw_score=raw_score,
                threshold=threshold,
                risk_drivers=top_risk_drivers,
                protective_factors=top_protective,
                customer_id=customer_id,
                reason="GEMINI_API_KEY is not configured; using deterministic rule-based retention analysis.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini GenAI service is not configured (missing or invalid GEMINI_API_KEY).",
            ) from None

    except GenAIValidationError as val_err:
        if allow_fallback:
            fallback_used = True
            explanation_obj = gemini_service.build_deterministic_fallback(
                customer_features=customer_dict,
                calibrated_prob=cal_prob,
                raw_score=raw_score,
                threshold=threshold,
                risk_drivers=top_risk_drivers,
                protective_factors=top_protective,
                customer_id=customer_id,
                reason=f"LLM grounding validation failed; using deterministic rule-based retention analysis.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini output validation failed: {gemini_service._sanitize_message(str(val_err))}",
            ) from None

    except GenAIProviderError as prov_err:
        if allow_fallback:
            fallback_used = True
            explanation_obj = gemini_service.build_deterministic_fallback(
                customer_features=customer_dict,
                calibrated_prob=cal_prob,
                raw_score=raw_score,
                threshold=threshold,
                risk_drivers=top_risk_drivers,
                protective_factors=top_protective,
                customer_id=customer_id,
                reason=f"Gemini provider failure; using deterministic rule-based retention analysis.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini provider request failed: {gemini_service._sanitize_message(str(prov_err))}",
            ) from None

    except Exception as e:
        sanitized = gemini_service._sanitize_message(str(e))
        logger.error("Unexpected error during GenAI explanation generation: %s", sanitized)
        if allow_fallback:
            fallback_used = True
            explanation_obj = gemini_service.build_deterministic_fallback(
                customer_features=customer_dict,
                calibrated_prob=cal_prob,
                raw_score=raw_score,
                threshold=threshold,
                risk_drivers=top_risk_drivers,
                protective_factors=top_protective,
                customer_id=customer_id,
                reason="Unexpected error occurred; using deterministic rule-based retention analysis.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate customer explanation: {sanitized}",
            ) from None

    # 8. Format structured items for response
    formatted_actions = [
        RecommendedActionItem(action=a.action, rationale=a.rationale)
        for a in explanation_obj.recommended_actions
    ]

    formatted_drivers = [
        FeatureAttributionItem(
            feature_name=d.get("feature_name", ""),
            display_name=d.get("display_name", d.get("feature_name", "")),
            attribution=round(float(d.get("attribution", 0.0)), 4),
            magnitude=d.get("magnitude", "MODERATE"),
            impact_percentage=round(float(d.get("impact_percentage", d.get("attribution", 0.0) * 100)), 2),
        )
        for d in top_risk_drivers
    ]

    formatted_protective = [
        FeatureAttributionItem(
            feature_name=p.get("feature_name", ""),
            display_name=p.get("display_name", p.get("feature_name", "")),
            attribution=round(float(p.get("attribution", 0.0)), 4),
            magnitude=p.get("magnitude", "MODERATE"),
            impact_percentage=round(float(p.get("impact_percentage", p.get("attribution", 0.0) * 100)), 2),
        )
        for p in top_protective
    ]

    return ExplainResponse(
        customer_id=customer_id,
        raw_model_score=round(raw_score, 4),
        calibrated_churn_probability=round(cal_prob, 4),
        operational_threshold=round(threshold, 4),
        risk_classification=explanation_obj.risk_tier,
        shap_risk_drivers=formatted_drivers,
        shap_protective_factors=formatted_protective,
        explanation=explanation_obj.risk_summary,
        recommended_actions=formatted_actions,
        disclaimer=explanation_obj.disclaimer,
        fallback_used=fallback_used,
        model_version=explanation_obj.model_version,
    )
