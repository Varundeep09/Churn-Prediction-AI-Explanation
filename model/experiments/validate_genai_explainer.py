"""
Validation Suite for GenAI Retention Explanation Engine
Customer Churn Prediction & Retention Intelligence System

Tests and validates all Phase 3C security remediations and engine capabilities:
1. Module imports cleanly without environment dependencies, API keys, or network activity.
2. Missing or placeholder API key raises a clear, actionable GenAIConfigurationError.
3. Placeholder-only .env.example verification and gitignore enforcement.
4. Robust dummy-key redaction across arbitrary prefixes (AIza, AQ., sk-, Bearer, query params).
5. Real HTTP request timeout configuration and conversion to milliseconds in SDK.
6. Mocked provider timeout and API failure handling.
7. Pre-flight input validation (rejection of NaN, infinite, negative, >1.0, non-float inputs).
8. Incompatible pipeline interface rejection (predict_raw_proba check).
9. Strict evidence grounding (rejection of unsupported SHAP driver labels).
10. Rejection of unsupported recommendation claims (guarantees, 50% discounts, fabricated complaints).
11. Deterministic fallback mode verification when allow_fallback=True.
12. Scale separation & prompt construction (calibrated prob vs raw score vs threshold vs SHAP).
13. Risk-tier isolation and convention verification (calibrated scale vs legacy tertiles).
14. Successful end-to-end mocked structured output parsing.
15. Immutability check: zero side-effects on model artifacts, calibrator weights, or files.
"""

import json
import math
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd


def run_validation():
    print("=" * 85)
    print("PHASE 3C SECURITY & CAPABILITY VALIDATION SUITE: GEMINI RETENTION ENGINE")
    print("=" * 85)

    test_counter = 0

    def test_header(title: str):
        nonlocal test_counter
        test_counter += 1
        print(f"\n[Test {test_counter}] {title}")

    # -----------------------------------------------------------------------
    # TEST 1: Clean Import without API Key and without Network Activity
    # -----------------------------------------------------------------------
    test_header("Clean module import without API key or network activity...")
    orig_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        from model.genai_explainer import (
            GeminiRetentionExplainer,
            RetentionExplanation,
            RetentionAction,
            GenAIConfigurationError,
            GenAIProviderError,
            GenAIValidationError,
            DEFAULT_GEMINI_MODEL,
        )
        print("  -> PASS: Module imported cleanly without GEMINI_API_KEY.")
        print(f"  -> Default Gemini model identified as: '{DEFAULT_GEMINI_MODEL}'")
    finally:
        if orig_key:
            os.environ["GEMINI_API_KEY"] = orig_key

    # -----------------------------------------------------------------------
    # TEST 2: Missing or Placeholder API Key Error Handling
    # -----------------------------------------------------------------------
    test_header("Missing or placeholder API key handling...")
    for dummy_val in [None, "", "your_gemini_api_key_here"]:
        explainer = GeminiRetentionExplainer(api_key=dummy_val)
        with patch.dict(os.environ, {}, clear=True):
            try:
                explainer.explain(
                    customer_features={"tenure": 12, "Contract": "Month-to-month"},
                    calibrated_prob=0.28,
                    raw_score=0.52,
                    threshold=0.27,
                )
                assert False, f"Failed to raise GenAIConfigurationError for api_key='{dummy_val}'"
            except GenAIConfigurationError as e:
                assert "GEMINI_API_KEY is not configured" in str(e)
    print("  -> PASS: GenAIConfigurationError raised appropriately for None, empty, and placeholder keys.")

    # -----------------------------------------------------------------------
    # TEST 3: Verification of .env.example and .gitignore
    # -----------------------------------------------------------------------
    test_header("Verification of .env.example template and .gitignore...")
    env_example_path = PROJECT_ROOT / ".env.example"
    gitignore_path = PROJECT_ROOT / ".gitignore"

    assert env_example_path.exists(), ".env.example file must exist"
    env_example_content = env_example_path.read_text(encoding="utf-8")

    # Verify no real key or active credential pattern is in .env.example
    for line in env_example_content.splitlines():
        if line.startswith("GEMINI_API_KEY"):
            assert "your_new_gemini_api_key_here" in line or "your_gemini_api_key_here" in line or "placeholder" in line, (
                f"Active credential found in .env.example: {line}"
            )
            assert not re.search(r"AQ\.[0-9A-Za-z\-_]{20,}", line), (
                "Real Google AI Studio key pattern found in .env.example!"
            )
            assert not re.search(r"AIza[0-9A-Za-z\-_]{20,}", line), (
                "Real AIza key pattern found in .env.example!"
            )

    # Verify .env is listed in .gitignore
    assert gitignore_path.exists(), ".gitignore file must exist"
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert re.search(r"^\.env\b", gitignore_content, re.MULTILINE), (
        "CRITICAL: .env is not present in .gitignore!"
    )
    print("  -> PASS: .env.example contains only safe placeholders; .env is strictly ignored by Git.")

    # -----------------------------------------------------------------------
    # TEST 4: Dummy-Key Redaction across Multiple Prefixes
    # -----------------------------------------------------------------------
    test_header("Dummy-key redaction across arbitrary prefixes and formats...")
    dummy_configured_key = "MY_CUSTOM_SECRET_KEY_9876543210"
    dummy_aiza_key = "AIzaSyDummySecretKey1234567890abcdef"
    dummy_aq_key = "AQ.DummyKeyAb8RN6J5x9uEViJ5jSZmge9SeyOjXNlghyCcBb3"
    dummy_sk_key = "sk-proj-DummyOpenAIKey1234567890abcdef123456"
    dummy_bearer = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_token_12345"
    dummy_url_param = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecretParam12345"

    explainer_sanitize = GeminiRetentionExplainer(api_key=dummy_configured_key)

    test_strings = [
        (f"Error connecting with configured key={dummy_configured_key}", dummy_configured_key),
        (f"Exception from endpoint: {dummy_aiza_key} invalid", dummy_aiza_key),
        (f"Provider failed for key AQ.{dummy_aq_key[3:]}", dummy_aq_key),
        (f"Failed authentication with token {dummy_sk_key}", dummy_sk_key),
        (f"Authorization header was {dummy_bearer}", dummy_bearer),
        (f"Request failed at URL {dummy_url_param}", "AIzaSySecretParam12345"),
    ]

    for raw_msg, secret in test_strings:
        sanitized = explainer_sanitize._sanitize_message(raw_msg)
        assert secret not in sanitized, f"CRITICAL LEAK: '{secret}' was not redacted in: '{sanitized}'"
        assert "[REDACTED" in sanitized, f"Expected [REDACTED] in sanitized output: '{sanitized}'"

    print("  -> PASS: All dummy secret patterns (configured, AIza, AQ., sk-, Bearer, query params) redacted safely.")

    # -----------------------------------------------------------------------
    # TEST 5: Real Timeout Configuration & Millisecond Conversion in SDK
    # -----------------------------------------------------------------------
    test_header("Real HTTP timeout verification in google-genai SDK...")
    timeout_target_seconds = 45.0
    explainer_timeout = GeminiRetentionExplainer(
        api_key="dummy_test_key_for_timeout",
        timeout_seconds=timeout_target_seconds,
    )

    client = explainer_timeout._get_client()

    # 1. Verify client-level HttpOptions timeout is in milliseconds
    client_http_opts = client._api_client.get_read_only_http_options()
    expected_ms = int(timeout_target_seconds * 1000)
    assert client_http_opts["timeout"] == expected_ms, (
        f"Client HttpOptions timeout mismatch: expected {expected_ms} ms, got {client_http_opts['timeout']}"
    )

    # 2. Verify SDK _build_request converts milliseconds back to seconds on the resulting HttpRequest
    req = client._api_client._build_request(
        "post",
        "models/gemini-2.5-flash:generateContent",
        {"contents": "test"},
        http_options=client._api_client._http_options,
    )
    assert req.timeout == timeout_target_seconds, (
        f"HttpRequest.timeout mismatch: expected {timeout_target_seconds}s, got {req.timeout}s"
    )
    print(f"  -> PASS: Configured timeout ({timeout_target_seconds}s -> {expected_ms}ms) verified on client and request.")

    # -----------------------------------------------------------------------
    # TEST 6: Mocked Provider Timeout and API Failures
    # -----------------------------------------------------------------------
    test_header("Mocked provider timeout and network failure handling...")
    explainer_net = GeminiRetentionExplainer(api_key="dummy_test_key")
    sample_features = {
        "tenure": 3,
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.5,
    }

    # Simulate network timeout
    with patch.object(explainer_net, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("HTTP request timed out after 30000ms")
        mock_get_client.return_value = mock_client

        try:
            explainer_net.explain(
                customer_features=sample_features,
                calibrated_prob=0.35,
                raw_score=0.52,
                threshold=0.27,
            )
            assert False, "Should have raised GenAIProviderError on timeout"
        except GenAIProviderError as e:
            assert "HTTP request timed out" in str(e)
            print(f"  -> PASS: Provider timeout correctly caught and wrapped in GenAIProviderError: '{e}'")

    # -----------------------------------------------------------------------
    # TEST 7: Pre-Flight Input Validation (NaN, Inf, Out-of-bounds, Types)
    # -----------------------------------------------------------------------
    test_header("Pre-flight input validation (NaN, Inf, Out-of-bounds, Types)...")
    invalid_cases = [
        ("calibrated_prob NaN", {"calibrated_prob": float("nan"), "raw_score": 0.5, "threshold": 0.27}),
        ("calibrated_prob Inf", {"calibrated_prob": float("inf"), "raw_score": 0.5, "threshold": 0.27}),
        ("calibrated_prob > 1.0", {"calibrated_prob": 1.25, "raw_score": 0.5, "threshold": 0.27}),
        ("calibrated_prob < 0.0", {"calibrated_prob": -0.15, "raw_score": 0.5, "threshold": 0.27}),
        ("calibrated_prob Boolean", {"calibrated_prob": True, "raw_score": 0.5, "threshold": 0.27}),
        ("calibrated_prob String", {"calibrated_prob": "0.50", "raw_score": 0.5, "threshold": 0.27}),
        ("raw_score NaN", {"calibrated_prob": 0.35, "raw_score": float("nan"), "threshold": 0.27}),
        ("raw_score > 1.0", {"calibrated_prob": 0.35, "raw_score": 1.05, "threshold": 0.27}),
        ("threshold < 0.0", {"calibrated_prob": 0.35, "raw_score": 0.52, "threshold": -0.05}),
        ("threshold > 1.0", {"calibrated_prob": 0.35, "raw_score": 0.52, "threshold": 1.01}),
    ]

    for label, kwargs in invalid_cases:
        with patch.object(explainer_net, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            try:
                explainer_net.explain(
                    customer_features=sample_features,
                    **kwargs,
                )
                assert False, f"Failed to reject invalid input: {label}"
            except GenAIValidationError as e:
                # Ensure no client or API call was attempted
                mock_client.models.generate_content.assert_not_called()
                assert "Invalid input" in str(e)

    print("  -> PASS: All NaN, infinite, out-of-bounds, and non-numeric inputs rejected locally prior to API calls.")

    # -----------------------------------------------------------------------
    # TEST 8: Incompatible Pipeline Objects Rejection
    # -----------------------------------------------------------------------
    test_header("Incompatible pipeline objects rejection in explain_from_pipeline()...")

    # Mock uncalibrated pipeline (e.g. standard LogisticRegression without predict_raw_proba)
    class DummyUncalibratedPipeline:
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])
        # Deliberately missing 'predict_raw_proba'

    dummy_pipe = DummyUncalibratedPipeline()
    mock_explainer = MagicMock()
    mock_explainer.explain_customer.return_value = {"top_risk_drivers": [], "top_protective_factors": []}

    try:
        explainer_net.explain_from_pipeline(
            customer_data={"tenure": 12, "MonthlyCharges": 50.0},
            calibrated_pipeline=dummy_pipe,
            explainer=mock_explainer,
            threshold=0.27,
        )
        assert False, "Should have rejected pipeline lacking predict_raw_proba"
    except GenAIValidationError as e:
        assert "Incompatible pipeline object" in str(e)
        assert "predict_raw_proba" in str(e)
        print(f"  -> PASS: Incompatible pipeline safely rejected before inference: '{e}'")

    # Incompatible explainer
    try:
        explainer_net.explain_from_pipeline(
            customer_data={"tenure": 12, "MonthlyCharges": 50.0},
            calibrated_pipeline=MagicMock(predict_raw_proba=MagicMock(), predict_proba=MagicMock()),
            explainer=object(),  # missing explain_customer
            threshold=0.27,
        )
        assert False, "Should have rejected explainer lacking explain_customer"
    except GenAIValidationError as e:
        assert "Incompatible explainer object" in str(e)
        print("  -> PASS: Incompatible explainer safely rejected.")

    # -----------------------------------------------------------------------
    # TEST 9: Strict Evidence Grounding (Unsupported SHAP Drivers)
    # -----------------------------------------------------------------------
    test_header("Deterministic evidence grounding: rejecting ungrounded drivers...")
    sample_drivers = [
        {"feature_name": "cat__Contract_Month-to-month", "display_name": "Contract: Month-to-month", "attribution": 0.0825, "magnitude": "HIGH"},
        {"feature_name": "cat__InternetService_Fiber optic", "display_name": "Internet Service: Fiber optic", "attribution": 0.0412, "magnitude": "MODERATE"},
    ]
    sample_protective = [
        {"feature_name": "num__tenure", "display_name": "Customer Tenure (Months)", "attribution": -0.0210, "magnitude": "LOW"},
    ]

    # Mock response returning an ungrounded risk driver not present in evidence
    hallucinated_payload = {
        "customer_id": "TEST-CUST-001",
        "calibrated_probability": 0.3421,
        "raw_model_score": 0.5512,
        "decision_threshold": 0.27,
        "risk_tier": "Medium",
        "risk_summary": "Customer exhibits elevated churn probability.",
        "risk_drivers": [
            "Contract: Month-to-month",
            "Customer had frequent network outages and poor service quality",  # UNGROUNDED!
        ],
        "protective_factors": ["Customer Tenure (Months)"],
        "recommended_actions": [
            {"action": "Offer annual contract upgrade.", "rationale": "Addresses month-to-month volatility."}
        ],
        "disclaimer": "Feature attributions are statistical associations.",
    }

    mock_resp = MagicMock()
    mock_resp.text = json.dumps(hallucinated_payload)

    with patch.object(explainer_net, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client

        try:
            explainer_net.explain(
                customer_features=sample_features,
                calibrated_prob=0.3421,
                raw_score=0.5512,
                threshold=0.27,
                risk_drivers=sample_drivers,
                protective_factors=sample_protective,
                allow_fallback=False,
            )
            assert False, "Should have raised GenAIValidationError on ungrounded driver"
        except GenAIValidationError as e:
            assert "Ungrounded risk driver detected" in str(e)
            print(f"  -> PASS: Ungrounded risk driver safely rejected: '{e}'")

    # -----------------------------------------------------------------------
    # TEST 10: Rejection of Unsupported Recommendation Claims
    # -----------------------------------------------------------------------
    test_header("Rejection of unsupported recommendation claims (guarantees, 50% discount, fabricated facts)...")
    predatory_recommendations = [
        ("Retention guarantee", {"action": "Guarantee that the customer will stay.", "rationale": "Model proves they will remain."}),
        ("Steep ungrounded discount", {"action": "Offer a 50% discount on all bills.", "rationale": "Special price reduction."}),
        ("Fabricated customer complaint", {"action": "Resolve billing dispute.", "rationale": "Customer complained about fiber service."}),
        ("Fabricated competitor offer", {"action": "Match competing rate.", "rationale": "Competitor offered a cheaper bundle."}),
    ]

    for label, bad_rec in predatory_recommendations:
        bad_payload = dict(hallucinated_payload)
        bad_payload["risk_drivers"] = ["Contract: Month-to-month"]
        bad_payload["recommended_actions"] = [bad_rec]

        mock_resp_bad = MagicMock()
        mock_resp_bad.text = json.dumps(bad_payload)

        with patch.object(explainer_net, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_resp_bad
            mock_get_client.return_value = mock_client

            try:
                explainer_net.explain(
                    customer_features=sample_features,
                    calibrated_prob=0.3421,
                    raw_score=0.5512,
                    threshold=0.27,
                    risk_drivers=sample_drivers,
                    protective_factors=sample_protective,
                    allow_fallback=False,
                )
                assert False, f"Should have raised GenAIValidationError for {label}"
            except GenAIValidationError as e:
                assert "Unsupported claim detected in recommendation" in str(e)

    print("  -> PASS: All unsupported promises, steep discounts, and fabricated customer facts rejected deterministically.")

    # -----------------------------------------------------------------------
    # TEST 11: Deterministic Fallback Mode when allow_fallback=True
    # -----------------------------------------------------------------------
    test_header("Deterministic fallback mode when allow_fallback=True...")
    with patch.object(explainer_net, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp_bad  # Bad recommendation payload
        mock_get_client.return_value = mock_client

        fallback_result = explainer_net.explain(
            customer_features=sample_features,
            calibrated_prob=0.3421,
            raw_score=0.5512,
            threshold=0.27,
            risk_drivers=sample_drivers,
            protective_factors=sample_protective,
            allow_fallback=True,
        )

        assert isinstance(fallback_result, RetentionExplanation)
        assert fallback_result.model_version == "deterministic-fallback"
        assert "deterministic rule fallback" in fallback_result.disclaimer
        assert fallback_result.risk_tier == "Medium"
        assert len(fallback_result.recommended_actions) >= 1
        print("  -> PASS: Transparent deterministic fallback executed cleanly with verified SHAP evidence.")

    # -----------------------------------------------------------------------
    # TEST 12: Prompt Construction & Metric Scale Separation
    # -----------------------------------------------------------------------
    test_header("Prompt construction and metric scale separation...")
    prompt = explainer_net.build_evidence_prompt(
        customer_features=sample_features,
        calibrated_prob=0.3421,
        raw_score=0.5512,
        threshold=0.2700,
        risk_drivers=sample_drivers,
        protective_factors=sample_protective,
        customer_id="TEST-CUST-001",
    )

    assert "Calibrated Churn Likelihood: 0.3421" in prompt
    assert "Uncalibrated Base Score:     0.5512" in prompt
    assert "Operational Threshold:       0.2700" in prompt
    assert "Risk Classification Tier:    Medium" in prompt
    assert "Contract: Month-to-month" in prompt
    assert "Customer Tenure (Months)" in prompt
    assert "TEST-CUST-001" in prompt
    print("  -> PASS: Prompt strictly segregates calibrated probability, raw score, threshold, and SHAP scales.")

    # -----------------------------------------------------------------------
    # TEST 13: Risk-Tier Convention Isolation
    # -----------------------------------------------------------------------
    test_header("Risk-tier convention isolation ('calibrated' vs 'legacy_tertiles')...")
    explainer_cal = GeminiRetentionExplainer(api_key="dummy", tier_convention="calibrated")
    explainer_leg = GeminiRetentionExplainer(api_key="dummy", tier_convention="legacy_tertiles")

    # In calibrated convention:
    assert explainer_cal._determine_risk_tier(0.52, 0.27) == "High"
    assert explainer_cal._determine_risk_tier(0.35, 0.27) == "Medium"
    assert explainer_cal._determine_risk_tier(0.15, 0.27) == "Low"

    # In legacy tertiles convention (matching api/main.py & dashboard):
    assert explainer_leg._determine_risk_tier(0.52, 0.27) == "Medium"
    assert explainer_leg._determine_risk_tier(0.67, 0.27) == "High"
    assert explainer_leg._determine_risk_tier(0.28, 0.27) == "Low"
    print("  -> PASS: Risk tier mappings isolated cleanly without altering production API conventions.")

    # -----------------------------------------------------------------------
    # TEST 14: Successful Mocked Structured Response
    # -----------------------------------------------------------------------
    test_header("Successful mocked structured response end-to-end...")
    valid_payload = {
        "customer_id": "TEST-CUST-001",
        "calibrated_probability": 0.3421,
        "raw_model_score": 0.5512,
        "decision_threshold": 0.27,
        "risk_tier": "Medium",
        "risk_summary": "Customer exhibits an estimated 34% likelihood of churn, exceeding the standard retention boundary.",
        "risk_drivers": [
            "Contract: Month-to-month",
            "Internet Service: Fiber optic",
        ],
        "protective_factors": [
            "Customer Tenure (Months)",
        ],
        "recommended_actions": [
            {
                "action": "Offer annual contract transition with rate stability.",
                "rationale": "Directly targets month-to-month contract volatility identified in model evidence."
            },
            {
                "action": "Introduce complementary Tech Support consultation.",
                "rationale": "Addresses higher risk associated with fiber optic connections."
            }
        ],
        "disclaimer": "Feature attributions represent observational statistical associations and do not prove causation.",
    }

    mock_good_response = MagicMock()
    mock_good_response.text = json.dumps(valid_payload)

    with patch.object(explainer_net, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_good_response
        mock_get_client.return_value = mock_client

        result = explainer_net.explain(
            customer_features=sample_features,
            calibrated_prob=0.3421,
            raw_score=0.5512,
            threshold=0.2700,
            risk_drivers=sample_drivers,
            protective_factors=sample_protective,
            customer_id="TEST-CUST-001",
        )

        assert isinstance(result, RetentionExplanation)
        assert result.customer_id == "TEST-CUST-001"
        assert result.calibrated_probability == 0.3421
        assert result.raw_model_score == 0.5512
        assert result.decision_threshold == 0.27
        assert result.risk_tier == "Medium"
        assert len(result.recommended_actions) == 2
        assert "observational statistical associations" in result.disclaimer
        print("  -> PASS: End-to-end schema validation passed completely!")

    # -----------------------------------------------------------------------
    # TEST 15: Immutability Check (No Side Effects on Model Artifacts)
    # -----------------------------------------------------------------------
    test_header("Verifying zero side-effects on model artifacts...")
    exp_dir = PROJECT_ROOT / "model" / "experiments"
    artifacts_to_check = [
        exp_dir / "calibrated_pipeline_refactored.pkl",
        exp_dir / "selected_model_pipeline.pkl",
        exp_dir / "calibrated_threshold_candidates.csv",
        exp_dir / "threshold_audit_results.json",
    ]
    mtimes_before = {a: a.stat().st_mtime for a in artifacts_to_check if a.exists()}

    # Run pipeline integration helper with mock to test explain_from_pipeline
    cal_pipe = joblib.load(exp_dir / "calibrated_pipeline_refactored.pkl")
    from model.explain import get_explainer
    shap_explainer = get_explainer()

    # Load a full, valid customer row from data
    data_csv = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df_raw = pd.read_csv(data_csv)
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
    full_sample_row = df_raw.dropna(subset=["TotalCharges"]).drop(columns=["customerID", "Churn"]).iloc[0].to_dict()

    with patch.object(explainer_net, "explain") as mock_explain_method:
        mock_explain_method.return_value = result
        pipe_result = explainer_net.explain_from_pipeline(
            customer_data=full_sample_row,
            calibrated_pipeline=cal_pipe,
            explainer=shap_explainer,
            threshold=0.27,
            customer_id="PIPE-TEST-001",
        )
        assert pipe_result == result

    for a, mtime_orig in mtimes_before.items():
        assert a.stat().st_mtime == mtime_orig, f"Artifact {a.name} was modified by GenAI service!"
    print("  -> PASS: Model weights, calibrator scores, and experiment artifacts remain strictly untouched.")

    print("\n" + "=" * 85)
    print("ALL 15 VALIDATION SUITE CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_validation()
