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

### Python backend (`requirements.txt` + imports)

| Library | Version pinned? | Used for |
|---|---|---|
| `pandas` | No | DataFrame operations throughout preprocessing, training, and API inference |
| `numpy` | No | Imported transitively; not directly called in source files beyond pandas operations |
| `scikit-learn` | No | `train_test_split`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, and all metric functions |
| `matplotlib` | No | Saving the Random Forest feature importance bar chart to PNG (`matplotlib.use("Agg")` for headless rendering) |
| `seaborn` | No | Used in `notebooks/01_eda.ipynb` for EDA visualizations (not imported in any `.py` file) |
| `fastapi` | No | Web framework for the prediction API; provides routing, request validation, and auto-generated `/docs` |
| `uvicorn` | No | ASGI server used to run the FastAPI app |
| `joblib` | No | Serializing and loading all model artifacts (`.pkl` files) |
| `python-multipart` | No | Required by FastAPI to support form data parsing (listed in `requirements.txt`, not directly imported) |
| `pydantic` | No (FastAPI dependency) | `BaseModel` and `Field` used to define and validate the `CustomerData` request schema in `api/main.py` |
| `pathlib` | stdlib | Path construction in `preprocess.py`, `train.py`, and `api/main.py` |
| `typing` | stdlib | `Optional[float]` used for `TotalCharges` field in `CustomerData` |
| `json` | stdlib | Writing `model_info.json` in `train.py` |
| `datetime` | stdlib | `date.today().isoformat()` for the training date in `model_info.json` |

### Frontend (`dashboard/`)

| Tool | Source | Used for |
|---|---|---|
| HTML5 | Static file | Page structure and form markup |
| CSS3 | Static file | All layout, theming, animations, and responsive breakpoints |
| Vanilla JavaScript (ES2020) | Static file | Form handling, `fetch()` API calls, DOM updates, chart rendering |
| Chart.js `4.4.0` | CDN (`cdn.jsdelivr.net`) | Doughnut chart for prediction result; three bar charts for EDA insights |
| Inter (font) | Google Fonts CDN | Body and heading typography |

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

## 8. Testing Performed

No formal test files (e.g. `pytest` test suite, `test_*.py` files) exist in the project. All testing was manual or ad-hoc via direct API calls and browser interaction.

### API validation — direct HTTP calls

A temporary Python script (`test_low_risk.py`) was written using `urllib.request` to POST the low-risk profile payload directly to `http://127.0.0.1:8001/predict` (note: port 8001, not 8000 — port 8000 was blocked by a Windows socket permission error during the session). The script was deleted after use. Confirmed result:

```json
{"churn_prediction": "No", "churn_probability": 0.0044, "risk_level": "Low"}
```

### Manual browser testing — three risk profiles

| Profile | Key inputs | Returned probability | Risk badge |
|---|---|---|---|
| High risk | tenure=2, Month-to-month, Fiber optic, Electronic check, MultipleLines=Yes, no security/support | ~82.3% | High |
| Medium risk | tenure=24, One year, mixed services | ~61.3% | Medium |
| Low risk | tenure=60, Two year, DSL, OnlineSecurity=Yes, TechSupport=Yes, Mailed check | 0.44% | Low |

The Medium result (61.3%) confirmed the 33–66% band is reachable in practice. The difference between the 61.3% result (MultipleLines=No) and the 82.3% result (MultipleLines=Yes) for otherwise similar high-risk profiles was identified and explained as a real model difference, not a bug.

### Error handling validation

Confirmed by code inspection (`script.js`): when `fetch()` throws a `TypeError` with "fetch" in the message (the browser's connection-refused error), `showError()` is called with the message "Cannot reach the API. Make sure the FastAPI server is running on http://127.0.0.1:8000", which renders visibly in the `#form-error` div. Not a silent console-only failure.

### Chart redraw validation

Confirmed by code inspection (`script.js`): `renderDoughnut()` calls `doughnutChart.destroy()` before creating a new `Chart` instance on every prediction. The module-level `doughnutChart` variable is reassigned each time. No ghosting or overlapping rings possible.

### Known environment issue

Port 8000 was blocked on the development machine by a Windows socket permission error (`[WinError 10013]`). The API was run on port 8001 instead (`uvicorn api.main:app --reload --port 8001`). The dashboard `API_URL` constant in `script.js` still points to `http://127.0.0.1:8000` — this would need updating if port 8001 is used consistently.

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

### 3. Local Setup & Execution (Windows PowerShell)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Configure Gemini API Key (optional - service falls back gracefully if omitted)
# In PowerShell:
$env:GEMINI_API_KEY = "your_actual_gemini_api_key"

# 3. Start the FastAPI backend
.\venv\Scripts\uvicorn.exe api.main:app --reload --host 127.0.0.1 --port 8000

# 4. In a separate terminal, serve the dashboard
python -m http.server 3000 --directory dashboard
# Open http://127.0.0.1:3000 in your browser
```

### 4. Integration Test Suite (`model/experiments/validate_api_integration.py`)

A dedicated 12-point automated integration test suite validates:
1. Root (`/`) and health (`/health`) endpoints.
2. Baseline `/predict` input/output contract preservation.
3. Schema validation and input rejection on `/explain`.
4. Decision threshold defaults, overrides, and invalid boundary rejection.
5. Score decoupling between raw Random Forest score and calibrated probability.
6. SHAP feature attribution integration.
7. Mocked successful Gemini structured generation.
8. Missing API key handling (`allow_fallback=True` returns fallback; `allow_fallback=False` returns HTTP 503).
9. Provider network timeout handling (`allow_fallback=True` returns fallback; `allow_fallback=False` returns HTTP 502).
10. Secret sanitization across all errors and logs.
11. Incompatible pipeline defensive guards.
12. Model artifact immutability.


