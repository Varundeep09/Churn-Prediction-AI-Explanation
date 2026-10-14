// Configurable API base URL with local and deployment fallback support
function resolveApiBase() {
  if (typeof window !== "undefined" && window.API_BASE_URL) {
    return window.API_BASE_URL.replace(/\/+$/, "");
  }
  const saved = typeof localStorage !== "undefined" ? localStorage.getItem("API_BASE_URL") : null;
  if (saved) {
    return saved.replace(/\/+$/, "");
  }
  if (typeof window !== "undefined" && (window.location.protocol === "file:" || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")) {
    return "http://127.0.0.1:8000";
  }
  return typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8000";
}

const API_BASE = resolveApiBase();
const PREDICT_URL = `${API_BASE}/predict`;
const EXPLAIN_URL = `${API_BASE}/explain`;

// Risk thresholds — match api/main.py get_risk_level() exactly:
function getRiskLevel(probability) {
  if (probability >= 0.66) return "High";
  if (probability >= 0.33) return "Medium";
  return "Low";
}

// ── DOM refs ──────────────────────────────────────────────────────
const form                 = document.getElementById("predict-form");
const submitBtn            = document.getElementById("submit-btn");
const btnText              = document.getElementById("btn-text");
const btnSpinner           = document.getElementById("btn-spinner");
const explainBtn           = document.getElementById("explain-btn");
const explainBtnText       = document.getElementById("explain-btn-text");
const explainSpinner       = document.getElementById("explain-spinner");
const formError            = document.getElementById("form-error");

// Tabs & result containers
const resultTabs           = document.getElementById("result-tabs");
const tabBaselineBtn       = document.getElementById("tab-baseline-btn");
const tabAiBtn             = document.getElementById("tab-ai-btn");
const resultPlaceholder    = document.getElementById("result-placeholder");
const resultContent        = document.getElementById("result-content");
const aiResultContent      = document.getElementById("ai-result-content");

// Baseline result refs
const probValue            = document.getElementById("prob-value");
const riskBadge            = document.getElementById("risk-badge");
const predictionVerdict    = document.getElementById("prediction-verdict");
const segmentNameText      = document.getElementById("segment-name-text");
const recommendationText   = document.getElementById("recommendation-text");
const supportingDataText   = document.getElementById("supporting-data-text");

// AI explanation result refs
const aiModelBadge         = document.getElementById("ai-model-badge");
const aiFallbackBadge      = document.getElementById("ai-fallback-badge");
const aiCalProb            = document.getElementById("ai-cal-prob");
const aiRawScore           = document.getElementById("ai-raw-score");
const aiThreshold          = document.getElementById("ai-threshold");
const aiRiskBadge          = document.getElementById("ai-risk-badge");
const aiVerdictSub         = document.getElementById("ai-verdict-sub");
const aiSummaryText        = document.getElementById("ai-summary-text");
const aiRiskDriversList    = document.getElementById("ai-risk-drivers-list");
const aiProtectiveList     = document.getElementById("ai-protective-factors-list");
const aiActionsList        = document.getElementById("ai-actions-list");
const aiDisclaimerText     = document.getElementById("ai-disclaimer-text");

let doughnutChart = null;
let lastBaselineResult = null;
let lastAiResult = null;

// ── Churn Playbook Rules (Derived from SQL & Statistical Findings) ────
const CHURN_PLAYBOOK = [
  {
    id: 1,
    name: "Critical Risk — High-Value New Customer",
    risk_level: "Critical",
    match: (p) => p.Contract === "Month-to-month" && p.tenure < 12 && p.MonthlyCharges > 70.0,
    action: "Priority retention outreach within 48 hours; offer contract upgrade discount or loyalty credit; assign dedicated account representative.",
    supporting_data: "47.68% 1st-year churn rate; 814 target accounts identified in SQL representing $67,418.90/month in revenue at immediate risk (from $323,148.10/month total high-value cohort)."
  },
  {
    id: 2,
    name: "Critical Risk — Unsupported Fiber Customer",
    risk_level: "Critical",
    match: (p) => p.InternetService === "Fiber optic" && p.OnlineSecurity === "No" && p.TechSupport === "No",
    action: "Proactively offer free 90-day trial of Tech Support & Online Security bundle; initiate onboarding check-in call.",
    supporting_data: "55.01% churn rate for this exact service combination vs 17.05% for all other customer profiles."
  },
  {
    id: 3,
    name: "High Risk — Manual Payment, Month-to-Month",
    risk_level: "High",
    match: (p) => p.PaymentMethod === "Electronic check" && p.Contract === "Month-to-month",
    action: "Encourage migration to automated recurring payment (Credit Card / Bank Transfer) via a $5-$10 bill credit incentive.",
    supporting_data: "45.29% churn rate for Electronic check users vs 15.25% for Auto Credit Card and 16.73% for Auto Bank Transfer."
  },
  {
    id: 4,
    name: "Moderate Risk — Early Tenure, Standard Plan",
    risk_level: "Moderate",
    match: (p) => p.tenure >= 13 && p.tenure <= 24 && p.Contract !== "Two year",
    action: "Offer annual contract renewal incentive before customer reaches the 2-year mark; monitor usage and satisfaction score.",
    supporting_data: "28.71% churn rate in 13-24 month cohort vs 20.39% at 25-48 months (statistical correlation r = -0.354 confirms strong protective effect of tenure)."
  },
  {
    id: 5,
    name: "Low Risk — Long-Term Contract Holder",
    risk_level: "Low",
    match: (p) => p.Contract === "Two year" && p.tenure > 48,
    action: "No urgent retention action required; target for loyalty perks, referral incentives, and premium service upsells.",
    supporting_data: "2.85% churn rate for Two-year contracts and 6.61% churn rate for customers with 61-72 months tenure."
  },
  {
    id: 6,
    name: "Low Risk — Fully Protected Customer",
    risk_level: "Low",
    match: (p) => p.OnlineSecurity === "Yes" && p.TechSupport === "Yes" && p.Contract !== "Month-to-month",
    action: "Maintain service quality; eligible for case study/testimonial requests and early access to new features.",
    supporting_data: "OnlineSecurity (Cramér's V = 0.347) and TechSupport (V = 0.343) statistically rank as top-4 drivers preventing churn."
  }
];

function evaluatePlaybookSegment(payload, risk, probability) {
  for (const segment of CHURN_PLAYBOOK) {
    if (segment.match(payload)) {
      return {
        segment_name: segment.name,
        recommended_action: segment.action,
        supporting_data: `Based on: ${segment.supporting_data}`
      };
    }
  }

  const fallbackAction = risk === "High"
    ? "Initiate proactive customer retention review; evaluate contract upgrade and loyalty offers."
    : risk === "Medium"
    ? "Monitor customer engagement across the next billing cycle and schedule a satisfaction check-in."
    : "No immediate action needed — this customer shows low churn risk.";

  return {
    segment_name: `${risk} Risk Profile (Standard)`,
    recommended_action: fallbackAction,
    supporting_data: `Based on: General ${risk.toLowerCase()} risk category threshold (model churn probability: ${(probability * 100).toFixed(1)}%).`
  };
}

// ── Build payload from form ───────────────────────────────────────
function buildPayload() {
  const fd = new FormData(form);
  const payload = {};

  for (const [key, value] of fd.entries()) {
    if (key === "SeniorCitizen" || key === "tenure") {
      payload[key] = parseInt(value, 10);
    } else if (key === "MonthlyCharges") {
      payload[key] = parseFloat(value);
    } else if (key === "TotalCharges") {
      const trimmed = value.trim();
      if (trimmed !== "") payload[key] = parseFloat(trimmed);
    } else if (key === "threshold") {
      const trimmed = value.trim();
      if (trimmed !== "") {
        const tVal = parseFloat(trimmed);
        if (!isNaN(tVal)) payload[key] = tVal;
      }
    } else {
      payload[key] = value;
    }
  }

  return payload;
}

// ── Baseline Submit Handler (/predict) ─────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  setLoadingState(true, false);

  const payload = buildPayload();
  const lastPayload = payload;

  try {
    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const msg = typeof detail.detail === "string"
        ? detail.detail
        : (Array.isArray(detail.detail) ? detail.detail.map(d => d.msg).join(", ") : `Server error: ${response.status}`);
      throw new Error(msg);
    }

    const data = await response.json();
    lastBaselineResult = data;
    renderBaselineResult(data, lastPayload);
    switchTab("baseline");
  } catch (err) {
    if (err instanceof TypeError && err.message.includes("fetch")) {
      showError(`Cannot reach the API. Make sure the FastAPI server is running on ${API_BASE}`);
    } else {
      showError(err.message || "An unexpected error occurred during prediction.");
    }
  } finally {
    setLoadingState(false, false);
  }
});

// ── AI Explanation Handler (/explain) ──────────────────────────────
explainBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  clearError();
  setLoadingState(true, true);

  const payload = buildPayload();
  payload.allow_fallback = true;

  try {
    const response = await fetch(EXPLAIN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const msg = typeof detail.detail === "string"
        ? detail.detail
        : (Array.isArray(detail.detail) ? detail.detail.map(d => d.msg).join(", ") : `Server error: ${response.status}`);
      throw new Error(msg);
    }

    const data = await response.json();
    lastAiResult = data;
    renderAiResult(data);
    switchTab("ai");
  } catch (err) {
    if (err instanceof TypeError && err.message.includes("fetch")) {
      showError(`Cannot reach the API. Make sure the FastAPI server is running on ${API_BASE}`);
    } else {
      showError(err.message || "An unexpected error occurred during AI retention analysis.");
    }
  } finally {
    setLoadingState(false, true);
  }
});

// ── Tab Switching ──────────────────────────────────────────────────
tabBaselineBtn.addEventListener("click", () => switchTab("baseline"));
tabAiBtn.addEventListener("click", () => switchTab("ai"));

function switchTab(mode) {
  resultTabs.classList.remove("hidden");
  resultPlaceholder.classList.add("hidden");

  if (mode === "baseline") {
    tabBaselineBtn.classList.add("active");
    tabAiBtn.classList.remove("active");
    resultContent.classList.remove("hidden");
    aiResultContent.classList.add("hidden");
  } else {
    tabAiBtn.classList.add("active");
    tabBaselineBtn.classList.remove("active");
    aiResultContent.classList.remove("hidden");
    resultContent.classList.add("hidden");
  }
}

// ── Count-Up Animation ─────────────────────────────────────────────
function animateCountUp(element, targetPct, duration, decimals = 1, suffix = "%") {
  const start = performance.now();
  const startVal = parseFloat(element.dataset.current || "0");
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = startVal + (targetPct - startVal) * eased;
    element.textContent = current.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
    else {
      element.dataset.current = targetPct;
      element.textContent = targetPct.toFixed(decimals) + suffix;
    }
  }
  requestAnimationFrame(step);
}

// ── Risk Badge Helpers ─────────────────────────────────────────────
const RISK_ICONS = { Low: "✓", Medium: "▲", High: "▲" };

// ── Render Baseline Prediction ─────────────────────────────────────
function renderBaselineResult(data, payload) {
  const pct = (data.churn_probability * 100).toFixed(1) + "%";
  const risk = data.risk_level;
  const retentionPct = parseFloat((100 - data.churn_probability * 100).toFixed(1));
  const churnPct = parseFloat((data.churn_probability * 100).toFixed(1));

  animateCountUp(probValue, churnPct, 350);

  riskBadge.textContent = RISK_ICONS[risk] + "  " + risk + " Risk";
  riskBadge.className = "risk-badge " + risk.toLowerCase();

  predictionVerdict.textContent =
    data.churn_prediction === "Yes"
      ? `⚠️ This customer is likely to churn (${pct} probability).`
      : `✅ This customer is likely to stay (${pct} churn probability).`;

  const playbook = evaluatePlaybookSegment(payload, risk, data.churn_probability);
  segmentNameText.textContent    = playbook.segment_name;
  recommendationText.textContent = playbook.recommended_action;
  supportingDataText.textContent = playbook.supporting_data;

  renderDoughnut(churnPct, retentionPct);

  resultContent.classList.remove("pulse-in");
  void resultContent.offsetWidth;
  resultContent.classList.add("pulse-in");
}

// ── Render AI Retention Dossier ────────────────────────────────────
function renderAiResult(data) {
  const calPct = data.calibrated_churn_probability * 100;
  const rawScore = data.raw_model_score;
  const threshold = data.operational_threshold;
  const risk = data.risk_classification;

  // 1. Metric values
  animateCountUp(aiCalProb, calPct, 350);
  aiRawScore.textContent = rawScore.toFixed(4);
  aiThreshold.textContent = threshold.toFixed(4);

  aiRiskBadge.textContent = RISK_ICONS[risk] + " " + risk + " Risk";
  aiRiskBadge.className = "risk-badge " + risk.toLowerCase();

  const diffFromThreshold = (data.calibrated_churn_probability - threshold) * 100;
  if (data.calibrated_churn_probability >= threshold) {
    aiVerdictSub.textContent = `+${diffFromThreshold.toFixed(1)}% above cutoff boundary`;
  } else {
    aiVerdictSub.textContent = `${Math.abs(diffFromThreshold).toFixed(1)}% below cutoff boundary`;
  }

  // 2. Model badges
  if (data.fallback_used) {
    aiFallbackBadge.classList.remove("hidden");
    aiModelBadge.textContent = "Deterministic Fallback";
  } else {
    aiFallbackBadge.classList.add("hidden");
    aiModelBadge.textContent = data.model_version || "Gemini 2.5 Flash";
  }

  // 3. Plain language executive summary
  aiSummaryText.textContent = data.explanation;

  // 4. SHAP Risk Drivers List
  aiRiskDriversList.innerHTML = "";
  if (data.shap_risk_drivers && data.shap_risk_drivers.length > 0) {
    data.shap_risk_drivers.forEach(d => {
      const card = document.createElement("div");
      card.className = "shap-item-card";
      const pctStr = d.impact_percentage > 0 ? `+${d.impact_percentage.toFixed(1)}%` : `${d.impact_percentage.toFixed(1)}%`;
      card.innerHTML = `
        <span class="shap-item-name" title="${d.display_name}">${d.display_name}</span>
        <span class="shap-mag-pill">${d.magnitude}</span>
        <span class="shap-impact-badge risk">${pctStr}</span>
      `;
      aiRiskDriversList.appendChild(card);
    });
  } else {
    aiRiskDriversList.innerHTML = `<span class="detail-value citation-text">No significant risk drivers detected.</span>`;
  }

  // 5. SHAP Protective Factors List
  aiProtectiveList.innerHTML = "";
  if (data.shap_protective_factors && data.shap_protective_factors.length > 0) {
    data.shap_protective_factors.forEach(p => {
      const card = document.createElement("div");
      card.className = "shap-item-card";
      const pctStr = p.impact_percentage.toFixed(1) + "%";
      card.innerHTML = `
        <span class="shap-item-name" title="${p.display_name}">${p.display_name}</span>
        <span class="shap-mag-pill">${p.magnitude}</span>
        <span class="shap-impact-badge protective">${pctStr}</span>
      `;
      aiProtectiveList.appendChild(card);
    });
  } else {
    aiProtectiveList.innerHTML = `<span class="detail-value citation-text">No significant protective factors detected.</span>`;
  }

  // 6. Recommended Actions List
  aiActionsList.innerHTML = "";
  if (data.recommended_actions && data.recommended_actions.length > 0) {
    data.recommended_actions.forEach(act => {
      const card = document.createElement("div");
      card.className = "ai-action-card";
      card.innerHTML = `
        <span class="ai-action-title">👉 ${act.action}</span>
        <span class="ai-action-rationale">${act.rationale}</span>
      `;
      aiActionsList.appendChild(card);
    });
  }

  // 7. Disclaimer
  aiDisclaimerText.textContent = data.disclaimer;

  aiResultContent.classList.remove("pulse-in");
  void aiResultContent.offsetWidth;
  aiResultContent.classList.add("pulse-in");
}

// ── Doughnut Chart ────────────────────────────────────────────────
function renderDoughnut(churnPct, retentionPct) {
  const ctx = document.getElementById("doughnut-chart").getContext("2d");

  if (doughnutChart) doughnutChart.destroy();

  doughnutChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Churn", "Retention"],
      datasets: [{
        data: [churnPct, retentionPct],
        backgroundColor: ["#dc2626", "#16a34a"],
        borderWidth: 0,
        hoverOffset: 4,
      }],
    },
    options: {
      cutout: "70%",
      plugins: {
        legend: { position: "bottom", labels: { font: { size: 12 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
          },
        },
      },
    },
  });
}

// ── Loading state ─────────────────────────────────────────────────
function setLoadingState(loading, isAi) {
  submitBtn.disabled = loading;
  explainBtn.disabled = loading;

  if (isAi) {
    explainSpinner.classList.toggle("hidden", !loading);
    explainBtnText.textContent = loading ? "Generating AI Dossier…" : "AI Retention Brief (Gemini + SHAP)";
  } else {
    btnSpinner.classList.toggle("hidden", !loading);
    btnText.textContent = loading ? "Predicting…" : "Predict Churn (Baseline LR)";
  }
}

// ── Error helpers ─────────────────────────────────────────────────
function showError(msg) {
  formError.textContent = msg;
  formError.classList.remove("hidden");
}

function clearError() {
  formError.textContent = "";
  formError.classList.add("hidden");
}

// ── Static EDA Charts ─────────────────────────────────────────────
function initEDACharts() {
  new Chart(document.getElementById("chart-contract").getContext("2d"), {
    type: "bar",
    data: {
      labels: ["Month-to-month", "One year", "Two year"],
      datasets: [{
        label: "Churn Rate (%)",
        data: [42.7, 11.3, 2.8],
        backgroundColor: ["#dc2626", "#d97706", "#16a34a"],
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 55, ticks: { callback: (v) => v + "%" }, grid: { color: "#f0f2f5" } },
        x: { grid: { display: false } },
      },
    },
  });

  new Chart(document.getElementById("chart-tenure").getContext("2d"), {
    type: "bar",
    data: {
      labels: ["0–12 mo", "13–24 mo", "25–48 mo", "49–60 mo", "61–72 mo"],
      datasets: [{
        label: "Churn Rate (%)",
        data: [47.4, 35.2, 22.1, 14.8, 6.6],
        backgroundColor: "#4f46e5",
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 60, ticks: { callback: (v) => v + "%" }, grid: { color: "#f0f2f5" } },
        x: { grid: { display: false } },
      },
    },
  });

  new Chart(document.getElementById("chart-internet").getContext("2d"), {
    type: "bar",
    data: {
      labels: ["Fiber optic", "DSL", "No service"],
      datasets: [{
        label: "Churn Rate (%)",
        data: [41.9, 19.0, 7.4],
        backgroundColor: ["#dc2626", "#d97706", "#16a34a"],
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 55, ticks: { callback: (v) => v + "%" }, grid: { color: "#f0f2f5" } },
        x: { grid: { display: false } },
      },
    },
  });
}

initEDACharts();
