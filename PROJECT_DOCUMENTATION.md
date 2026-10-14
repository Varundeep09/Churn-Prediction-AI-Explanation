# Customer Churn Prediction — Project Documentation

> Generated from direct inspection of all project source files.
> All values, numbers, and descriptions are verified against the actual code.
> Nothing is assumed or filled in from general knowledge.

---

## 1. Project Overview

This project builds an end-to-end customer churn prediction system for a telecom business using the IBM Telco Customer Churn dataset (7,043 customers, 21 raw columns). The business problem it solves is identifying customers who are likely to cancel their service before they actually leave, so retention teams can intervene early. The system covers the full ML lifecycle: exploratory data analysis in a Jupyter notebook, a preprocessing pipeline that cleans and encodes the raw data, a training script that compares two classifiers and selects the better one by F1-score, a FastAPI inference service that accepts raw customer attributes and returns a churn probability with a risk label, and a static HTML/CSS/JS dashboard that lets a user fill in a customer profile, call the API live, and see the result alongside EDA-derived insights and model performance metrics.

---

## 2. Full Tech Stack

Verified from `requirements.txt` and actual `import` statements in each source file.

### Python Backend & AI Engine (`requirements.txt` + Imports)

| Library | Version Constraint | Used For |
|---|---|---|
| `pandas` | Unpinned | DataFrame operations throughout preprocessing, training, SQL query execution, and API inference |
| `numpy` | Unpinned | Numeric array manipulation and sigmoid math in `calibrated_pipeline.py`, `explain.py`, and `genai_explainer.py` |
| `scikit-learn` | Unpinned | `train_test_split`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, and calibration estimators |
| `scipy` | Unpinned | Statistical hypothesis testing (`chi2_contingency`, `pointbiserialr`) in `model/root_cause_analysis.py` |
| `shap` | `>=0.45.0` | `TreeExplainer` feature attribution and customer-level waterfall explanations in `model/explain.py` |
| `google-genai` | `>=2.0.0` | Official Google GenAI Python SDK (`from google import genai`) powering `model/genai_explainer.py` with Gemini 2.5 Flash |
| `fastapi` | Unpinned | Web framework serving `/predict` and `/explain` endpoints, request validation, and OpenAPI documentation |
| `uvicorn` | Unpinned | Production ASGI web server running the FastAPI application |
| `joblib` | Unpinned | Serializing and loading production and calibrated pipeline artifacts (`.pkl` files) |
| `python-dotenv` | `>=1.0.0` | Loading local `.env` configuration securely without committing secrets |
| `pydantic` | Unpinned (FastAPI dep) | Pydantic v2 schemas (`CustomerData`, `ExplainRequest`, `ExplainResponse`, `RetentionExplanation`) |
| `python-multipart` | Unpinned | Form data parsing support in FastAPI |
| `matplotlib` | Unpinned | Saving feature importance chart (`model/feature_importance.png`) and calibration curve figures |
| `seaborn` | Unpinned | Used in `notebooks/01_eda.ipynb` for exploratory data analysis charts |
| `pathlib` | stdlib | Robust, cross-platform filesystem path resolution relative to repository root |
| `typing` | stdlib | Type annotations (`Optional`, `Dict`, `List`, `Union`, `Any`) |
| `json` | stdlib | Serializing metadata and evaluation metrics |
| `datetime` | stdlib | Stamping ISO date formats |

### Frontend Dashboard (`dashboard/`)

| Tool | Source | Used For |
|---|---|---|
| HTML5 | Static file (`index.html`) | Semantic page structure, dual-action buttons, tab navigation, and result containers |
| CSS3 | Static file (`style.css`) | Modern glassmorphic theme, responsive breakpoints, metric cards, and shimmer animations |
| Vanilla JavaScript (ES2020) | Static file (`script.js`) | Dynamic API URL resolution, dual API dispatch, tab switching, and count-up animations |
| Chart.js `4.4.0` | CDN (`cdn.jsdelivr.net`) | Interactive doughnut prediction chart and three EDA benchmark bar charts |
| Inter (Font) | Google Fonts CDN | High-legibility typography across body text, badges, and headers |

---

## 3. Complete Workflow

Steps in the exact order the pipeline runs.

### Step 1 — Exploratory Data Analysis (`notebooks/01_eda.ipynb`)

Performed in a Jupyter notebook using pandas, matplotlib, and seaborn. The notebook is a binary `.ipynb` file; its findings are documented in `README.md` and reflected in the dashboard's hardcoded chart values. Key findings recorded:

- Dataset is moderately imbalanced: 73.46% no-churn, 26.54% churn
- `TotalCharges` contained 11 blank-string values requiring coercion to `NaN`
- Contract type is a strong churn predictor: month-to-month customers churn at ~42.7%, one-year at ~11.3%, two-year at ~2.8%
- Tenure is a strong predictor: 0–12 month customers churn at ~47.4%, declining to ~6.6% at 61–72 months
- Fiber optic internet customers churn at ~41.9% vs DSL at ~19.0% vs no service at ~7.4%
- Churning customers have higher average monthly charges and lower average tenure

### Step 2 — Preprocessing (`model/preprocess.py`)

Runs as a standalone script (`python model/preprocess.py`). Exact steps from the code:

1. Loads `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` using a path relative to the script's location via `Path(__file__).resolve().parents[1]`
2. Converts `TotalCharges` to numeric with `pd.to_numeric(..., errors="coerce")`, turning blank strings into `NaN`
3. Drops rows where `TotalCharges` is `NaN` using `dropna(subset=["TotalCharges"])`
4. Drops the `customerID` column
5. Maps `Churn` to binary integers: `Yes → 1`, `No → 0`
6. Separates features (`X`) from target (`y`)
7. One-hot encodes all `object`/`category`/`string` dtype columns using `pd.get_dummies(..., drop_first=True)`
8. Splits into train/test with `test_size=0.2`, `stratify=y`, `random_state=42`
9. Casts `tenure`, `MonthlyCharges`, `TotalCharges` to `float`
10. Fits `StandardScaler` on training numeric columns only, then transforms both train and test
11. Saves to `model/` via `joblib.dump`: `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`, `scaler.pkl`, `feature_columns.pkl`

### Step 3 — Model Training (`model/train.py`)

Runs as a standalone script (`python model/train.py`). Exact steps from the code:

1. Loads the four split artifacts from `model/` via `joblib.load`
2. Trains `LogisticRegression(max_iter=1000)` — no other hyperparameters set
3. Trains `RandomForestClassifier(n_estimators=200, random_state=42)` — no other hyperparameters set
4. Evaluates both models on `X_test`/`y_test` using `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `confusion_matrix`
5. Prints a side-by-side comparison table
6. Computes top-10 feature importances from the Random Forest and saves a horizontal bar chart to `model/feature_importance.png` at 150 DPI using `matplotlib.use("Agg")`
7. Selects the final model using: `if random_forest_results["f1_score"] > logistic_results["f1_score"]` — the model with the strictly higher F1-score wins; ties go to Logistic Regression
8. Saves the winning model to `model/churn_model.pkl`
9. Writes `model/model_info.json` with the winning model's name, four rounded metrics, and `date.today().isoformat()`

### Step 4 — API (`api/main.py`)

Started with `uvicorn api.main:app --reload` from the project root. On startup:

- Loads `churn_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` once into module-level variables
- Defines `numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]`

On each `POST /predict` request, `prepare_input_data()` runs:

1. Calls `customer.model_dump()` to get a plain dict
2. If `TotalCharges` is `None`, sets it to `tenure * MonthlyCharges`
3. Converts the dict to a one-row `pd.DataFrame`
4. Encodes with `pd.get_dummies(input_df, drop_first=False)` — **not** `drop_first=True`
5. Reindexes to `feature_columns` with `fill_value=0`
6. Casts numeric columns to `float`
7. Applies `scaler.transform()` to the three numeric columns only
8. Returns the aligned DataFrame to `predict()`

### Step 5 — Dashboard (`dashboard/`)

Opened directly as a static file in a browser (`dashboard/index.html`). No local server required. On load, `initEDACharts()` renders three static bar charts. On form submit, `buildPayload()` reads the form, casts types, and calls `fetch("http://127.0.0.1:8000/predict", ...)`. On response, `renderResult(data, payload)` updates the result panel.

---

## 4. Exact Model Results

Pulled directly from `model/model_info.json` (winning model) and `model/train.py` (comparison logic and README table for the losing model).

### From `model_info.json` (Logistic Regression — selected model)

```json
{
  "chosen_model": "Logistic Regression",
  "accuracy": 0.8045,
  "precision": 0.6505,
  "recall": 0.5722,
  "f1_score": 0.6088,
  "date_trained": "2026-07-29"
}
```

### From `README.md` model comparison table (Random Forest — not selected)

| Metric | Value |
|---|---|
| Accuracy | 0.7868 |
| Precision | 0.6225 |
| Recall | 0.5027 |
| F1-score | 0.5562 |

> Note: The Random Forest metrics in the README were recorded at training time. They are not stored in `model_info.json` (which only records the winning model). The `train.py` code prints both sets of metrics to stdout but only persists the winner.

### Selection criterion

From `train.py` line: `if random_forest_results["f1_score"] > logistic_results["f1_score"]`. Since `0.5562 < 0.6088`, the condition was false and Logistic Regression was selected. The rationale stated in comments: F1-score is preferred over accuracy for imbalanced churn data because it balances false positives and false negatives.

---

## 5. The One-Hot Encoding Bug

### What the bug was

During training in `preprocess.py`, `pd.get_dummies(..., drop_first=True)` is applied to the full 7,000+ row dataset. With many rows, every categorical column has multiple distinct values present, so `drop_first=True` drops one reference category per column and produces a stable, complete feature schema.

The original API code applied the same `drop_first=True` to a single-row inference DataFrame. With only one row, each categorical column has exactly one value present. `drop_first=True` on a single-category column drops that one value entirely — it has nothing to compare against. This silently erased dummy columns for whichever category happened to be submitted, including high-signal churn indicators such as:

- `InternetService_Fiber optic`
- `PaymentMethod_Electronic check`
- `PaperlessBilling_Yes`
- `MultipleLines_Yes`

The subsequent `reindex(columns=feature_columns, fill_value=0)` then filled those missing columns back in as zeros, making the model treat a Fiber optic / Electronic check / Month-to-month customer as if they had none of those attributes. High-risk customers were scored artificially low.

### How it was fixed

The fix is documented in `api/main.py` comments and the README. The change was:

**Before:**
```python
encoded_df = pd.get_dummies(input_df, drop_first=True)
```

**After:**
```python
encoded_df = pd.get_dummies(input_df, drop_first=False)
```

With `drop_first=False`, all present dummy columns are created. The `reindex` step that follows still handles alignment to the training schema — adding zeros for any categories not present in this single request, and dropping any unexpected columns. The fix also confirmed that `reindex` must happen before `scaler.transform`, which is the order in the current code.

A second fix added at the same time: `TotalCharges` was made `Optional[float]` in the `CustomerData` Pydantic model, with a fallback of `tenure * MonthlyCharges` when it is `None`.

---

## 6. Dashboard Features

Verified from `dashboard/index.html` and `dashboard/script.js`.

### Header
- Title: "Customer Churn Prediction"
- Subtitle: "Telco Customer Churn — Logistic Regression Model"
- "ML Dashboard" pill badge (hidden on mobile via CSS)

### Stats strip (below header, above main content)
Five fact chips separated by dividers: "7,043 customers analyzed", "19 input features", "80.4% accuracy", "Logistic Regression", "26.5% base churn rate". Horizontally scrollable on mobile.

### Left panel — Customer input form (19 fields)

Eyebrow label: "STEP 1". Section heading: "Customer Details".

**Account Info fieldset (8 fields):**
- `gender` — dropdown: Female, Male
- `SeniorCitizen` — dropdown: No (0), Yes (1) — sent as integer
- `Partner` — dropdown: No, Yes
- `Dependents` — dropdown: No, Yes
- `tenure` — number input, min=0, max=72 — sent as integer
- `Contract` — dropdown: Month-to-month, One year, Two year
- `PaperlessBilling` — dropdown: Yes, No
- `PaymentMethod` — dropdown: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)

**Services fieldset (9 fields):**
- `PhoneService` — dropdown: Yes, No
- `MultipleLines` — dropdown: No, Yes, No phone service
- `InternetService` — dropdown: DSL, Fiber optic, No
- `OnlineSecurity` — dropdown: No, Yes, No internet service
- `OnlineBackup` — dropdown: No, Yes, No internet service
- `DeviceProtection` — dropdown: No, Yes, No internet service
- `TechSupport` — dropdown: No, Yes, No internet service
- `StreamingTV` — dropdown: No, Yes, No internet service
- `StreamingMovies` — dropdown: No, Yes, No internet service

**Billing fieldset (2 fields):**
- `MonthlyCharges` — number input, step=0.01 — sent as float
- `TotalCharges` — number input, step=0.01, optional — omitted from payload if blank; API estimates as `tenure × MonthlyCharges`

**Submit button:** "Predict Churn" with inline CSS spinner (border animation) that replaces the text during loading. Button is disabled while loading.

**Error display:** `#form-error` div shown with red background when fetch fails or API returns non-200. Exact message for connection failure: "Cannot reach the API. Make sure the FastAPI server is running on http://127.0.0.1:8000". Cleared on each new submission attempt.

### Right panel — Result display

**Empty state (shown before first prediction):**
- Icon + "Churn Risk Analyzer" title + one-sentence model description
- "How it works" tinted card with a 3-step numbered stepper: (1) Enter customer details, (2) Model analyzes patterns, (3) Get churn probability
- CTA line: "← Fill in the form and click Predict Churn"

**Result state (shown after prediction, replaces empty state):**
- Eyebrow label: "RESULTS"
- Heading: "Prediction Result"
- Churn probability percentage — animated with ease-out cubic count-up over 350ms using `requestAnimationFrame`; animates from the previous value on repeated predictions
- Risk badge — pill shape with icon prefix (✓ for Low, ▲ for Medium/High) and color coding: green (`#dcfce7` / `#16a34a`) for Low, amber (`#fef3c7` / `#b45309`) for Medium, red (`#fee2e2` / `#dc2626`) for High
- Chart.js doughnut chart — two segments: Churn (red `#dc2626`) and Retention (green `#16a34a`), `cutout: "70%"`, legend at bottom, tooltip shows percentage. Previous chart instance is destroyed before each new render via `doughnutChart.destroy()`
- Verdict line — "⚠️ This customer is likely to churn (X% probability)." or "✅ This customer is likely to stay (X% churn probability)."
- "Key risk factors" detail block — rule-based text derived from submitted form values; checks 9 conditions (month-to-month contract, tenure ≤ 12, Fiber optic, no OnlineSecurity, no TechSupport, Electronic check, PaperlessBilling=Yes, MonthlyCharges ≥ 70, SeniorCitizen=1); returns up to 4 matching factors as a comma-separated string
- "Recommended action" detail block — three fixed strings keyed by risk level: High → "Consider proactive retention outreach — offer a contract upgrade or loyalty discount.", Medium → "Monitor engagement over the next billing cycle and consider a check-in call.", Low → "No immediate action needed — this customer shows low churn risk."
- Fade-slide-in animation (300ms ease) on the result panel on each new prediction, re-triggered via forced reflow (`void resultContent.offsetWidth`)

### Model metrics strip

Eyebrow: "MODEL PERFORMANCE". Heading: "Logistic Regression — Test Set Results". Four metric cards with hover lift effect:
- Accuracy: 80.45%
- Precision: 65.05%
- Recall: 57.22%
- F1-Score: 60.88%

Values are hardcoded in HTML, sourced from `model_info.json`.

### Key Insights section

Eyebrow: "EDA FINDINGS". Three Chart.js bar charts with hover lift on cards, labeled "based on training data analysis":

1. **Churn Rate by Contract Type** — bars: Month-to-month 42.7% (red), One year 11.3% (amber), Two year 2.8% (green)
2. **Churn Rate by Tenure Group** — bars: 0–12 mo 47.4%, 13–24 mo 35.2%, 25–48 mo 22.1%, 49–60 mo 14.8%, 61–72 mo 6.6% (all indigo `#4f46e5`)
3. **Churn Rate by Internet Service** — bars: Fiber optic 41.9% (red), DSL 19.0% (amber), No service 7.4% (green)

All three charts use `responsive: true`, `maintainAspectRatio: false`, no legend, percentage y-axis ticks.

### Footer

Two-column layout: left shows project name + training date; right shows "Built with Python · Scikit-learn · FastAPI · Chart.js" and a GitHub icon button (links to `#` — placeholder only).

### Responsive behavior

- ≤ 960px: two-column top grid collapses to single column; metrics and insights grids go to 2 columns
- ≤ 768px: field rows stack label above input; result main stacks vertically; insights grid goes to 1 column; header badge hidden
- ≤ 480px: font size reduced to 14px; padding tightened; metrics grid stays at 2 columns

---

## 7. API Endpoints

All three endpoints verified from `api/main.py`.

### `GET /`

```python
@app.get("/")
def root() -> dict:
    return {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "description": "FastAPI service for predicting telecom customer churn.",
    }
```

Returns a static dict with API name, version string, and description. No parameters.

### `GET /health`

```python
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "API is running"}
```

Returns a static dict. Used to confirm the server is up. No parameters.

### `POST /predict`

Accepts a JSON body matching the `CustomerData` Pydantic model. All fields required except `TotalCharges` which is `Optional[float]` defaulting to `None`.

**Request body fields:**

| Field | Type | Required |
|---|---|---|
| `gender` | `str` | Yes |
| `SeniorCitizen` | `int` | Yes |
| `Partner` | `str` | Yes |
| `Dependents` | `str` | Yes |
| `tenure` | `int` | Yes |
| `PhoneService` | `str` | Yes |
| `MultipleLines` | `str` | Yes |
| `InternetService` | `str` | Yes |
| `OnlineSecurity` | `str` | Yes |
| `OnlineBackup` | `str` | Yes |
| `DeviceProtection` | `str` | Yes |
| `TechSupport` | `str` | Yes |
| `StreamingTV` | `str` | Yes |
| `StreamingMovies` | `str` | Yes |
| `Contract` | `str` | Yes |
| `PaperlessBilling` | `str` | Yes |
| `PaymentMethod` | `str` | Yes |
| `MonthlyCharges` | `float` | Yes |
| `TotalCharges` | `Optional[float]` | No — estimated as `tenure × MonthlyCharges` if omitted |

**Response:**

```json
{
  "churn_prediction": "Yes" | "No",
  "churn_probability": float (rounded to 4 decimal places),
  "risk_level": "Low" | "Medium" | "High"
}
```

**Risk level thresholds** (from `get_risk_level()`):
- `probability >= 0.66` → `"High"`
- `probability >= 0.33` → `"Medium"`
- `probability < 0.33` → `"Low"`

**CORS:** `allow_origins=["*"]`, all methods and headers allowed. Configured for development use.

**Auto-generated docs:** Available at `http://127.0.0.1:8000/docs` (Swagger UI) when the server is running.

---

## 8. Automated Testing & Verification Framework

The platform includes three dedicated automated test suites located in `model/experiments/` that validate machine learning explainability, GenAI guardrails, and full-stack API integration without modifying model weights or re-training:

### 1. SHAP Customer-Level Explainability Suite (`model/experiments/validate_shap.py`)
- Evaluates `CustomerExplainer` across 4 diverse customer profiles (High Risk, Low Risk, Categorical Profile, Boundary Edge Case).
- **Additivity Verification:** Verifies that model base value plus the sum of all feature SHAP attributions exactly reconstructs the model output:
  $$\text{Reconstructed Output} = \phi_0 + \sum_{i=1}^M \phi_i = p_{\text{raw}}$$
  Verified additivity discrepancy: **$0.00\text{e}+00$ across all 4 profiles**.
- **Output Artifact:** Persists evaluation results to `model/experiments/shap_validation_results.json`.

### 2. GenAI Unit & Guardrail Test Suite (`model/experiments/validate_genai_explainer.py`)
Executes 15 focused capability, reliability, and security checks:
1. Clean module import without requiring an API key or triggering network calls.
2. Graceful rejection of missing, empty, or placeholder API keys (`GenAIConfigurationError`).
3. Template validation verifying `.env.example` contains only placeholders and `.gitignore` ignores `.env`.
4. Robust dummy-key redaction across arbitrary secret formats (`AIza`, `AQ.`, `sk-`, `Bearer`, `?key=`).
5. HTTP timeout propagation to the official `google-genai` client options (`timeout_seconds = 45.0s`).
6. Mocked provider timeout handling (`GenAIProviderError`).
7. Pre-flight input validation catching NaNs, infinities, and out-of-bounds probabilities before API requests.
8. Defensive pipeline type checking (rejection of uncalibrated pipelines or incompatible explainers).
9. Semantic evidence grounding (rejection of hallucinated drivers not present in input SHAP evidence).
10. Rejection of unsupported claims (unauthorized discounts, guarantees, or fabricated customer facts).
11. Deterministic fallback generation when `allow_fallback=True`.
12. Strict prompt metric separation (preventing scale mixing between calibrated and uncalibrated scores).
13. Risk-tier boundary isolation (`calibrated` vs legacy conventions).
14. End-to-end schema validation against `RetentionExplanation` using mocked structured responses.
15. Verification of zero side-effects on serialized model artifacts.

### 3. Full-Stack FastAPI Integration Suite (`model/experiments/validate_api_integration.py`)
A comprehensive 12-point automated test suite running against the live FastAPI application:
1. Service root (`/`) and health check (`/health`) endpoints.
2. Baseline `POST /predict` input/output contract preservation (verifying exact Logistic Regression predictions).
3. `POST /explain` input validation (rejecting missing attributes or invalid types with HTTP 422).
4. Operational decision threshold validation (defaults to `0.27`, accepts valid float overrides, rejects $\notin [0.0, 1.0]$).
5. Strict metric scale decoupling (confirming raw Random Forest score $\ne$ Platt-calibrated probability).
6. Customer-level SHAP feature attribution extraction (validating risk driver and protective factor vectors).
7. End-to-end execution with mocked successful Gemini structured responses.
8. Missing API key handling (`allow_fallback=True` returns HTTP 200 with fallback; `allow_fallback=False` returns HTTP 503).
9. Simulated provider network failure/timeout handling (`allow_fallback=True` returns HTTP 200 with fallback; `allow_fallback=False` returns HTTP 502).
10. Complete secret sanitization verifying zero credentials or stack traces in HTTP responses and logs.
11. Incompatible pipeline object defensive rejection.
12. Model artifact immutability verification.

---

## 9. Phase 3C — GenAI Retention Explanation Engine (`model/genai_explainer.py`)

A standalone, production-ready GenAI explanation service powered by the official `google-genai` Python SDK (`GeminiRetentionExplainer`). It converts calibrated model predictions, raw Random Forest risk scores, operational decision thresholds, and SHAP feature attributions into concise, evidence-grounded customer retention briefs.

### Key Specifications:
- **Module:** `model/genai_explainer.py`
- **SDK:** `google-genai` (official Google GenAI Python SDK v2.x)
- **Configuration:** Reads `GEMINI_API_KEY` and optional `GEMINI_MODEL` (default: `gemini-2.5-flash`) via `python-dotenv` or environment variables.
- **Output Schema (`RetentionExplanation`):** Validated Pydantic v2 model containing `risk_summary`, `risk_drivers`, `protective_factors`, `recommended_actions` (action + rationale), and mandatory `disclaimer`.
- **Safety & Evidence Guardrails:**
  - Treats customer features as untrusted input data, preventing prompt injection.
  - Strictly grounded in supplied SHAP attributions; no hallucinated promotions, discounts, or causes.
  - Explicitly declares that feature attributions are observational statistical associations, not causal mechanisms.
  - Maintains strict metric separation between calibrated frequentist probabilities ($P_{\text{cal}}$) and uncalibrated raw scores ($p_{\text{raw}}$).
- **Validation Suite:** `model/experiments/validate_genai_explainer.py` validates imports without API keys, configuration errors on missing credentials, malformed JSON handling, mock-tested schema validation, secret sanitization, and artifact immutability.

---

## 10. Phase 3D — Full-Stack FastAPI & Web Dashboard Integration

### 1. Architectural Overview

Phase 3D connects the experimental machine learning models and GenAI service directly into the user-facing application:
- **FastAPI Service (`api/main.py`):**
  - Preserves baseline `/predict` endpoint (Logistic Regression baseline, unchanged).
  - Implements experimental `/explain` endpoint combining:
    1. Experimental Platt-calibrated Random Forest (`model/experiments/calibrated_pipeline_refactored.pkl`).
    2. Local SHAP TreeExplainer customer attributions (`model/explain.py`).
    3. Google Gemini Retention Explanation service (`model/genai_explainer.py`).
- **Interactive Web Dashboard (`dashboard/`):**
  - Adds dual-action workflow:
    - Primary: **Predict Churn (Baseline LR)**
    - Secondary: **AI Retention Brief (Gemini + SHAP)**
  - Seamless Tab Switcher: toggle freely between Baseline model output and the AI Retention Intelligence Dossier.
  - Multi-scale metrics display: Calibrated Churn Likelihood, Uncalibrated Base Score, and Operational Cutoff.
  - Granular SHAP breakdown cards for Elevating Drivers and Mitigating Factors.
  - Tailored, evidence-grounded retention initiatives generated by Gemini.
  - Non-causal statistical disclaimer and scale decoupling notice.

### 2. `/explain` Endpoint Specification

- **Method:** `POST`
- **Route:** `/explain`
- **Request Body (`ExplainRequest`):**
  - Inherits standard 19-attribute `CustomerData`.
  - Optional `threshold`: Float in $[0.0, 1.0]$ (default: `0.27`).
  - Optional `customer_id`: String customer identifier.
  - Optional `allow_fallback`: Boolean (default: `true`). If true, generates transparent rule-based deterministic fallback when Gemini API key is missing or endpoint is unreachable.
- **Response Body (`ExplainResponse`):**
  - `raw_model_score`: Uncalibrated Random Forest internal base score $[0.0, 1.0]$.
  - `calibrated_churn_probability`: Frequentist churn probability from Platt scaling $[0.0, 1.0]$.
  - `operational_threshold`: Active decision cutoff.
  - `risk_classification`: Categorical tier (`"Low"`, `"Medium"`, `"High"`).
  - `shap_risk_drivers`: Top positive SHAP feature attribution items.
  - `shap_protective_factors`: Top negative SHAP feature attribution items.
  - `explanation`: Plain-language probabilistic summary generated by Gemini.
  - `recommended_actions`: Prioritized retention actions with evidence rationales.
  - `disclaimer`: Non-causality interpretability disclaimer.
  - `fallback_used`: Boolean indicating if deterministic fallback was returned.
  - `model_version`: LLM model identifier or `"deterministic-fallback"`.

### 3. Local Setup & Execution Guide (Windows PowerShell)

```powershell
# 1. Clone repository and navigate
git clone https://github.com/Varundeep09/Churn-Prediction-AI-Explanation.git
cd Churn-Prediction-AI-Explanation

# 2. Virtual environment setup
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Securely configure Gemini API Key (optional)
Copy-Item .env.example .env
# Open .env in a local text editor and enter your key:
# GEMINI_API_KEY=your_new_gemini_api_key_here
# GEMINI_MODEL=gemini-2.5-flash
# Note: Never share or commit your API key. If omitted, the service falls back gracefully.

# 4. Run automated test suites
python model/experiments/validate_shap.py
python model/experiments/validate_genai_explainer.py
python model/experiments/validate_api_integration.py

# 5. Start the FastAPI backend
uvicorn api.main:app --reload --port 8000

# 6. In a separate PowerShell terminal, serve the dashboard
python -m http.server 3000 --directory dashboard
# Open http://127.0.0.1:3000 in your browser
```

---

## 11. Root Cause Analysis, SQL Analytics & Retention Playbook

### 1. Statistical Hypothesis Testing (`model/root_cause_analysis.py`)

To identify features with strong statistical dependencies on churn, formal tests were conducted across the IBM Telco population:
- **Categorical Features:** Pearson's Chi-Square ($\chi^2$) Test of Independence with Cramér's V effect size.
- **Continuous Features:** Point-Biserial Correlation ($r$) with two-tailed p-values.

| Feature | Feature Type | Statistical Test | Test Statistic | p-value | Effect Size | Key Finding |
|---|---|---|---|---|---|---|
| `Contract` | Categorical | Pearson's $\chi^2$ | $\chi^2 = 1179.55$ | $< 0.0001$ | Cramér's $V = 0.410$ | Month-to-month contracts have the highest statistical association with customer attrition. |
| `tenure` | Numeric | Point-Biserial | $r = -0.3540$ | $< 0.0001$ | $\|r\| = 0.354$ | Strong protective association: churn risk decreases significantly with each month of tenure. |
| `OnlineSecurity` | Categorical | Pearson's $\chi^2$ | $\chi^2 = 846.68$ | $< 0.0001$ | Cramér's $V = 0.347$ | Absence of security features associates with elevated attrition. |
| `TechSupport` | Categorical | Pearson's $\chi^2$ | $\chi^2 = 824.93$ | $< 0.0001$ | Cramér's $V = 0.343$ | Unresolved customer friction in the absence of tech support associates with departures. |
| `PaymentMethod` | Categorical | Pearson's $\chi^2$ | $\chi^2 = 645.43$ | $< 0.0001$ | Cramér's $V = 0.303$ | Electronic check billing associates with 45.3% churn vs ~15% for automated payments. |

> **Methodological Note:** These statistical tests measure observational dependency and association; they do not establish causal intervention mechanisms.

### 2. SQL Analytics Suite (`sql/`)

A dedicated SQLite data warehouse (`sql/churn.db`) was built to execute analytical SQL queries answering core revenue and retention questions:
1. **`churn_by_tenure_cohort.sql`:** First-year customers exhibit a 47.68% churn rate, dropping to 6.61% for accounts with 5+ years of tenure.
2. **`churn_by_contract_type.sql`:** Month-to-month churn rate is 42.71% compared to 2.85% for two-year contracts.
3. **`churn_by_payment_method.sql`:** Electronic check users churn at 45.29% vs 15.25% for automatic credit card billing.
4. **`churn_by_customer_value.sql`:** High-value customers (>$70/month) total 3,581 accounts representing **$323,148.10/month in total recurring revenue** with a 35.38% churn rate.
5. **`churn_by_service_combo.sql`:** Fiber optic users lacking both Tech Support and Online Security churn at 55.01% (vs 17.05% baseline).
6. **`high_value_at_risk.sql`:** Identifies an actionable high-priority target list of **814 accounts** (month-to-month, tenure < 12 months, charges > $70/month) representing **$67,418.90/month in immediate revenue at risk**.

### 3. Churn Retention Playbook (`model/churn_playbook.json`)

Translates statistical findings and SQL analytics into 6 structured retention interventions integrated into the prediction dashboard:
1. **Critical Risk — High-Value New Customer:** Priority retention outreach within 48 hours; contract upgrade discount or loyalty bill credit ($67,418.90/month immediate revenue at risk).
2. **Critical Risk — Unsupported Fiber Customer:** Complimentary 90-day trial of Tech Support & Online Security bundle; onboarding check-in call.
3. **High Risk — Manual Payment, Month-to-Month:** $5–$10 bill credit incentive to migrate from Electronic Check to automated credit card/ACH billing.
4. **Moderate Risk — Early Tenure, Standard Plan:** Annual renewal discount offered before reaching peak churn window (13–24 months).
5. **Low Risk — Long-Term Contract Holder:** Loyalty rewards, milestone recognition, and premium service upsells.
6. **Low Risk — Fully Protected Customer:** Service satisfaction surveys and testimonial/advocacy outreach.


