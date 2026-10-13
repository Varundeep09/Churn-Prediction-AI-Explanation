# Customer Churn Prediction & Retention Intelligence System

## Project Overview

This project is an end-to-end Machine Learning and Decision Intelligence platform designed to identify telecom customers at risk of churn and automate data-driven retention interventions. Using the IBM Telco Customer Churn dataset (7,043 records), the system combines rigorous statistical root-cause analysis, a dedicated SQL analytics data warehouse, and a calibrated classification model with a real-time FastAPI inference engine and interactive web dashboard.

---

## Key Results at a Glance

* **Production Model Performance:** Logistic Regression achieved **80.45% Accuracy** and **60.88% F1-Score**, selected for production deployment for superior precision (65.05%) and fewer false positives compared to uncalibrated tree baselines.
* **Primary Statistical Association:** Month-to-month contracts have the strongest statistical association with attrition (χ² = 1179.55, p < 0.0001, Cramér's V = 0.410; associative correlation, not causal determination).
* **Primary Protective Factor:** Customer tenure exhibits the strongest inverse statistical association with churn (r = -0.3540, p < 0.0001).
* **Critical SQL Revenue Finding:** First-year customers churn at **47.68%**; 814 high-value at-risk accounts represent **$67,418.90/month** in immediate revenue at risk (out of **$323,148.10/month** across all 3,581 high-value accounts).
* **Actionable Retention Playbook:** 6 data-backed customer segments mapped to specific retention workflows and wired directly into the live prediction UI.

---

## Tech Stack

- **Machine Learning & Statistics:** Python, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, Joblib
- **Explainability & Calibration:** SHAP (TreeExplainer), Platt Scaling (Logistic Sigmoid Calibrator)
- **Generative AI & LLM:** Google GenAI SDK (`google-genai`), Gemini 2.5 Flash (`gemini-2.5-flash`)
- **Database & SQL Analytics:** SQLite 3, Structured `.sql` Query Suite
- **API & Backend:** FastAPI, Uvicorn, Pydantic, Python-dotenv
- **Frontend & UI:** HTML5, CSS3 (Modern Glassmorphic Design System), JavaScript (ES2020), Chart.js

---

## Project Structure

```text
customer-churn-prediction/
├── api/
│   └── main.py                          # FastAPI service with /predict & /explain endpoints
├── dashboard/
│   ├── index.html                       # Web dashboard UI with baseline & AI Dossier tabs
│   ├── style.css                        # Glassmorphic CSS design system with risk badges
│   └── script.js                        # Form handling, dual API dispatch & Chart.js charts
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # IBM Telco raw dataset (7,043 rows)
├── model/
│   ├── churn_model.pkl                  # Production Logistic Regression pipeline
│   ├── churn_playbook.json              # Structured business retention action rules
│   ├── feature_columns.pkl              # Saved training feature schema
│   ├── feature_importance.png           # Feature importance visualization
│   ├── model_info.json                  # Model metadata and evaluation metrics
│   ├── preprocess.py                    # Cleaning, encoding, scaling & split pipeline
│   ├── root_cause_analysis.py           # Statistical hypothesis testing (Chi-Square & Point-Biserial)
│   ├── root_cause_findings.json         # Statistically ranked churn drivers
│   ├── scaler.pkl                       # Fitted StandardScaler artifact
│   ├── train.py                         # Model training and benchmark script
│   ├── calibrated_pipeline.py           # Self-contained Platt Calibrated Classifier wrapper
│   ├── explain.py                       # SHAP TreeExplainer & customer risk driver extraction
│   ├── genai_explainer.py               # Google Gemini GenAI executive retention explainer
│   ├── experiments/                     # Experimental models, validation & benchmark suites
│   │   ├── calibrated_pipeline_refactored.pkl # Calibrated Balanced RF artifact (threshold 0.27)
│   │   ├── validate_genai_explainer.py  # Standalone unit test suite for GenAI explainer
│   │   └── validate_api_integration.py  # End-to-end integration test suite for /explain & /predict
│   ├── X_test.pkl, X_train.pkl          # Feature train/test splits
│   └── y_test.pkl, y_train.pkl          # Target train/test splits
├── notebooks/
│   └── 01_eda.ipynb                     # Exploratory Data Analysis & visualizations
├── sql/
│   ├── build_database.py                # Database creation script (ingests to churn.db)
│   ├── churn.db                         # SQLite database with `customers` table
│   ├── queries/                         # Dedicated analytical SQL queries
│   │   ├── churn_by_tenure_cohort.sql
│   │   ├── churn_by_contract_type.sql
│   │   ├── churn_by_payment_method.sql
│   │   ├── churn_by_customer_value.sql
│   │   ├── churn_by_service_combo.sql
│   │   └── high_value_at_risk.sql
│   ├── query_results/                   # Exported query results (CSV)
│   └── run_all_queries.py               # SQL query execution & export runner
├── .env.example                         # Template for GEMINI_API_KEY configuration
├── .gitignore                           # Git ignore rules (protects .env and binary models)
├── requirements.txt                     # Python dependencies
├── PROJECT_DOCUMENTATION.md             # Comprehensive technical documentation
└── README.md
```

---

## Dataset

- **Dataset:** IBM Telco Customer Churn
- **Source:** Kaggle / IBM Sample Data
- **File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 customer records, 21 attributes (demographic, subscribed services, contract type, billing channels, and churn label).

---

## 1. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted in `notebooks/01_eda.ipynb`:
- **Class Distribution:** Moderately imbalanced: `73.46%` retained vs `26.54%` churned.
- **Contract Risk:** Month-to-month contracts exhibit high churn (~42.7%) compared to multi-year contracts (~2.8%).
- **Tenure Risk:** Customers in their first year (0–12 months) account for the majority of churn events (~47.4%).
- **Data Quality:** Identified 11 blank string entries in `TotalCharges` requiring numeric coercion and cleaning.

---

## 2. Preprocessing Pipeline (`model/preprocess.py`)

1. Coerces `TotalCharges` to numeric and drops the 11 `NaN` records (retaining 7,032 clean rows).
2. Drops non-predictive `customerID`.
3. Encodes binary target: `Churn` (`Yes=1`, `No=0`).
4. One-hot encodes categoricals using `pd.get_dummies(..., drop_first=True)`.
5. Performs stratified 80/20 train/test split (`random_state=42`) to preserve class proportions.
6. Scales continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` fitted on the training split only.
7. Persists artifacts (`scaler.pkl`, `feature_columns.pkl`, datasets) via `joblib`.

---

## 3. Modeling & Benchmark Results (`model/train.py`)

Two classification models were trained and benchmarked:
- **Logistic Regression** (`max_iter=1000`)
- **Random Forest Classifier** (`n_estimators=200`, `random_state=42`)

### Benchmark Comparison (Test Set: N = 1,407)

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|---|---:|---:|---:|---:|---|
| **Logistic Regression** | **0.8045** | **0.6505** | **0.5722** | **0.6088** | **Selected / Deployed** |
| Random Forest Classifier | 0.7868 | 0.6225 | 0.5027 | 0.5562 | Baseline Comparison |

* **Selection Rationale:** Logistic Regression won decisively across all 4 evaluation metrics. It captured 26 more true churners than Random Forest (yielding a +6.95% higher Recall) with superior precision and calibration.

---

## 4. Root Cause Analysis (`model/root_cause_analysis.py`)

To identify *why* customers churn beyond tree-based importance heuristics, formal statistical hypothesis testing was conducted:
- **Categorical Features:** Pearson's Chi-Square (χ²) Test of Independence with Cramér's V effect size.
- **Continuous Features:** Point-Biserial Correlation (r) with exact two-tailed p-values.

### Top Statistically Significant Churn Drivers

| Rank | Feature | Type | Test | Statistic | p-value | Effect Size | Finding |
|---|---|---|---|---|---|---|---|
| **1** | `Contract` | Categorical | Chi-Square (χ²) | χ² = 1179.55 | < 0.0001 | Cramér's V = 0.410 | Month-to-month contracts have the highest statistical dependency with churn. |
| **2** | `tenure` | Numeric | Point-Biserial | r = -0.3540 | < 0.0001 | \|r\| = 0.354 | Strong protective factor: each month of tenure significantly lowers churn risk. |
| **3** | `OnlineSecurity` | Categorical | Chi-Square (χ²) | χ² = 846.68 | < 0.0001 | Cramér's V = 0.347 | Absence of online security sharply increases customer attrition. |
| **4** | `TechSupport` | Categorical | Chi-Square (χ²) | χ² = 824.93 | < 0.0001 | Cramér's V = 0.343 | Lack of tech support leads to unresolved service friction and departures. |
| **5** | `PaymentMethod` | Categorical | Chi-Square (χ²) | χ² = 645.43 | < 0.0001 | Cramér's V = 0.303 | Electronic check users churn at 45.3% vs ~15% for automatic payments. |
| **-** | `PhoneService` | Categorical | Chi-Square (χ²) | χ² = 0.87 | 0.3499 | V = 0.011 | *Not statistically significant (p > 0.05).* |
| **-** | `gender` | Categorical | Chi-Square (χ²) | χ² = 0.48 | 0.4905 | V = 0.008 | *Not statistically significant (p > 0.05).* |

---

## 5. SQL Analytics Layer (`sql/`)

A dedicated SQLite data warehouse (`sql/churn.db`) was constructed to run production SQL queries answering key business questions:

1. **`churn_by_tenure_cohort.sql`**: Cohort analysis showing first-year churn (47.68%) dropping to 6.61% for 5+ year customers.
2. **`churn_by_contract_type.sql`**: Month-to-month churn (42.71%) vs Two-year contracts (2.85%).
3. **`churn_by_payment_method.sql`**: Electronic check churn (45.29%) vs Auto Credit Card (15.25%).
4. **`churn_by_customer_value.sql`**: High-value (>$70/mo) accounts represent $323,148.10/month in total recurring revenue with a 35.38% churn rate.
5. **`churn_by_service_combo.sql`**: Fiber optic users without Tech Support or Security churn at 55.01% (vs 17.05% baseline).
6. **`high_value_at_risk.sql`**: Actionable target list of **814 accounts** (>$70/mo, month-to-month, <12m tenure) representing **$67,418.90/month** in immediate revenue at risk for retention campaigns.

Run all SQL queries via:
```powershell
python sql/build_database.py
python sql/run_all_queries.py
```

---

## 6. Churn Retention Playbook (`model/churn_playbook.json`)

Translates statistical findings and SQL insights into an actionable decision matrix:

1. **Critical Risk — High-Value New Customer:** Month-to-month AND tenure < 12 months AND MonthlyCharges > $70. Priority outreach within 48h; offer contract upgrade discount.
2. **Critical Risk — Unsupported Fiber Customer:** Fiber Optic AND No Tech Support AND No Security. Proactive 90-day free trial of Support & Security bundle.
3. **High Risk — Manual Payment, Month-to-Month:** Electronic check AND Month-to-month. Offer $5–$10 bill credit to migrate to automated payment.
4. **Moderate Risk — Early Tenure, Standard Plan:** Tenure 13–24 months AND not on a 2-year contract. Annual renewal discount before entering high-churn window.
5. **Low Risk — Long-Term Contract Holder:** Two-year contract AND tenure > 48 months. Loyalty perks & premium upsells.
6. **Low Risk — Fully Protected Customer:** Tech Support AND Online Security AND Multi-year contract. Case study/testimonial candidates.

---

## 7. Prediction & Retention Intelligence API (`api/main.py`)

Built with **FastAPI** to serve both the production baseline model and the experimental explainable GenAI retention pipeline.

### Production Baseline Endpoint: `POST /predict`
- **Model:** Logistic Regression (`model/churn_model.pkl`).
- **Purpose:** Fast, lightweight production churn classification.
- **Independence:** Completely isolated from GenAI dependencies; operates without any API keys.
- **Request / Response:**
```json
// Request POST /predict
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "Yes",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.45, "TotalCharges": 190.90
}

// Response
{
  "churn_prediction": "Yes",
  "churn_probability": 0.8232,
  "risk_level": "High"
}
```

### Experimental GenAI Retention Endpoint: `POST /explain`
- **Pipeline:** Platt-calibrated Balanced Random Forest (`model/experiments/calibrated_pipeline_refactored.pkl`).
- **Explainability:** SHAP `TreeExplainer` (`model/explain.py`) generating verified customer-level risk drivers and protective mitigators.
- **GenAI Reasoning:** Google Gemini 2.5 Flash (`model/genai_explainer.py`) via the official `google-genai` SDK.
- **Decoupled Scoring Architecture:**
  - `raw_model_score`: Underlying Random Forest probability ($0.5393$).
  - `calibrated_churn_probability`: Platt-scaled empirical probability ($0.3133$).
  - `operational_threshold`: Default `0.27` (selected via out-of-sample cross-validation), supports validated overrides in $[0.0, 1.0]$.
- **Graceful Fallback & Error Resilience:**
  - If `allow_fallback: true` (default), missing API keys or provider timeouts return HTTP 200 with deterministic rule-based retention actions clearly marked (`fallback_used: true`).
  - If `allow_fallback: false`, strict HTTP 503 (missing key) or HTTP 502 (provider error) is returned with zero credential leakage.

```json
// Request POST /explain
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "Yes",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.45, "TotalCharges": 190.90,
  "threshold": 0.27,
  "allow_fallback": true,
  "customer_id": "CUST-9842"
}

// Response
{
  "customer_id": "CUST-9842",
  "raw_model_score": 0.5393,
  "calibrated_churn_probability": 0.3133,
  "operational_threshold": 0.27,
  "risk_classification": "High Risk",
  "risk_drivers": [
    {
      "feature": "Contract",
      "shap_value": 0.0812,
      "feature_value": "Month-to-month",
      "direction": "increases_risk",
      "description": "Increases churn risk (SHAP +0.0812)"
    }
  ],
  "protective_factors": [
    {
      "feature": "tenure",
      "shap_value": -0.0451,
      "feature_value": 2.0,
      "direction": "decreases_risk",
      "description": "Decreases churn risk (SHAP -0.0451)"
    }
  ],
  "ai_explanation": "Customer CUST-9842 exhibits an elevated calibrated churn likelihood of 31.33% (exceeding the 27.00% operational threshold). The primary risk drivers are a Month-to-month contract and lack of Tech Support...",
  "recommended_actions": [
    "Present a targeted 12-month contract upgrade offer with guaranteed pricing.",
    "Provision a complimentary 60-day trial of Tech Support and Device Protection."
  ],
  "disclaimer": "This explanation was generated by Google Gemini using SHAP attribution from an experimental calibrated Random Forest pipeline. SHAP values reflect model feature attribution on the uncalibrated scale and do not imply causal relationships.",
  "model_type": "BalancedRandomForestClassifier (Experimental)",
  "calibration_method": "Platt Scaling (Logistic Sigmoid)",
  "fallback_used": false
}
```

---

## 8. Interactive Dashboard (`dashboard/`)

A responsive single-page application built with modern HTML5, CSS3, and ES2020 JavaScript:
- **19-Field Input Form:** Full customer profile specification with auto-calculation of total charges and optional threshold tuning (default `0.27`).
- **Dual-Action Inference Dispatch:**
  - **`Predict Churn` Button:** Invokes production `/predict` endpoint (Logistic Regression baseline, doughnut chart, risk badge, static playbook mapping).
  - **`AI Retention Brief` Button:** Invokes experimental `/explain` endpoint (Calibrated Random Forest + SHAP + Gemini GenAI).
- **Interactive Tab Switcher (`#result-tabs`):** Toggle between **Baseline Prediction** and **AI Retention Dossier** instantly without form re-entry or data loss.
- **AI Retention Dossier Layout:**
  - **4-Card Metric Grid:** Calibrated Risk Probability, Raw RF Tree Score, Operational Threshold, and Operational Status badge.
  - **Gemini Executive Retention Brief:** Grounded, plain-language executive narrative.
  - **SHAP Evidence Lists:** Top risk drivers (+SHAP) and protective mitigators (-SHAP) showing exact feature values and attribution impacts.
  - **Retention Intervention Cards:** Prioritized retention roadmap items.
  - **Model Provenance & Non-Causal Disclaimers:** Explicitly states model type, calibration method, and non-causal interpretation.
- **UX Protections:** Buttons lock during requests to prevent duplicate submissions; animated shimmer skeletons communicate active processing.

---

## 9. How to Run the Project

Follow these exact steps in **Windows PowerShell**:

### Step 1: Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Configure Gemini API Key (Optional for Local AI Analysis)
```powershell
# Copy the environment template
Copy-Item .env.example .env

# Edit .env and supply your API key from Google AI Studio (https://aistudio.google.com/)
# Example .env content:
# GEMINI_API_KEY=your_actual_key_here
# GEMINI_MODEL=gemini-2.5-flash
```
> **Security Note:** `.env` is listed in `.gitignore` and will never be committed to git. If no key is configured, the application runs seamlessly in deterministic fallback mode or serves the baseline production endpoint.

### Step 3: Run Validation & Integration Test Suites
```powershell
# Run the Phase 3C GenAI explainer test suite (15 unit tests)
python model/experiments/validate_genai_explainer.py

# Run the Phase 3D Full-Stack API integration test suite (12 tests)
python model/experiments/validate_api_integration.py
```

### Step 4: Start the FastAPI Backend
```powershell
uvicorn api.main:app --reload --port 8000
```
- Real-time API: `http://127.0.0.1:8000`
- Interactive OpenAPI / Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Step 5: Launch the Web Dashboard
Open `dashboard/index.html` directly in any modern browser (or serve via VS Code Live Server / `python -m http.server 3000`).

---

## 10. Responsible AI Guardrails & Architecture Integrity

1. **Decoupled Scoring Scales:**
   - The raw Random Forest tree score ($\in [0, 1]$) and Platt-calibrated empirical probability ($\in [0, 1]$) are presented separately.
   - Raw SHAP values describe feature attributions on the tree model's uncalibrated score, not percentage point shifts on the calibrated scale.
2. **Strict Non-Causal Grounding:**
   - Prompts and UI disclaimers explicitly instruct stakeholders that SHAP values and model associations do not establish causal relationships (e.g., forcing a customer onto a two-year contract does not automatically eliminate underlying dissatisfaction).
3. **Zero Secret Leakage:**
   - Exception handlers across backend endpoints and GenAI services automatically scrub API keys, credentials, and raw stack traces before returning client responses.
4. **Production Baseline Immutability:**
   - `model/churn_model.pkl` and `/predict` remain 100% frozen and operational. Experiments are completely encapsulated in `/explain` and `model/experiments/`.
