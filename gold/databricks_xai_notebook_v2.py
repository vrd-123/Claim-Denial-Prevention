# Databricks notebook source
# MAGIC %md
# MAGIC # Step 11 — Explainable AI (XAI): SHAP-Based Claim Denial Explanations
# MAGIC
# MAGIC **Lineage:**
# MAGIC ```
# MAGIC workspace.gold.gold_claim_features
# MAGIC         │
# MAGIC         ▼
# MAGIC [SHAP TreeExplainer on XGBoost model loaded from MLflow]
# MAGIC         │
# MAGIC         ▼
# MAGIC workspace.gold.gold_claim_explanations
# MAGIC ```
# MAGIC
# MAGIC ### What this notebook does
# MAGIC | Step | Action |
# MAGIC |------|--------|
# MAGIC | 1 | Loads the best XGBoost model from MLflow via `runs:/<run_id>/model` (CE-compatible) |
# MAGIC | 2 | Runs inference on every row of `gold_claim_features` to get denial probabilities |
# MAGIC | 3 | Computes per-claim SHAP values using `TreeExplainer` (exact, not approximate) |
# MAGIC | 4 | Selects the **Top 3 most impactful features** per claim by absolute SHAP magnitude |
# MAGIC | 5 | Translates technical feature names → human-readable business reasons (sign-aware) |
# MAGIC | 6 | Writes `gold_claim_explanations` to the Gold Delta layer |
# MAGIC | 7 | Produces an aggregate **Top Denial Drivers** summary for ops reporting |
# MAGIC
# MAGIC ### Design Decisions
# MAGIC - **TreeExplainer** is used because it is mathematically exact for tree-based models and runs in
# MAGIC   polynomial time against the tree structure (far faster than KernelSHAP on 1,000+ rows).
# MAGIC - **Sign of SHAP value**: a *negative* value pushes prediction toward `0` (Denied); a *positive*
# MAGIC   value pushes toward `1` (Approved). The reason text is selected accordingly.
# MAGIC - **Top-3 reasons** strike the right balance between completeness and human readability for a
# MAGIC   business-facing denial explanation letter.
# MAGIC - **CE-compatible**: No `mlflow.register_model()` is used. The model is served directly from its
# MAGIC   `runs:/` URI, which works on Databricks Community Edition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Install Dependencies
# MAGIC
# MAGIC Pin versions to ensure SHAP ↔ XGBoost compatibility.
# MAGIC `numpy<2` is required because SHAP's C extensions are not yet compiled for NumPy 2.x.

# COMMAND ----------

# MAGIC %pip install "numpy<2" "scipy<1.14" "xgboost==3.2.0" "shap>=0.45" --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Imports & Configuration
# MAGIC
# MAGIC ⚠️ **ACTION REQUIRED:** Replace `XGB_RUN_ID` with the actual run ID printed at the end of
# MAGIC your model training notebook (Cell 5 — XGBoost run ID line).

# COMMAND ----------

import shap
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── MLflow Config ──────────────────────────────────────────────────────────────
# Must match the experiment path used in the training notebook exactly.
EXPERIMENT_NAME = "/Users/varadnaik03@gmail.com/claim_denial_prevention_mlflow"

# ── ACTION REQUIRED ────────────────────────────────────────────────────────────
# Paste the XGBoost run_id that was printed at the end of your training notebook.
# Example: "7f706dc17f7240deb385f9fb451c428e"
XGB_RUN_ID = "7f706dc17f7240deb385f9fb451c428e"

# ── Feature set — must match gold_claim_features exactly ──────────────────────
# NOTE: diag_category_enc is intentionally excluded here.
# The training notebook (Cell 2) commented it out due to a KeyError, so it is
# NOT present in gold_claim_features. This list must stay in sync with
# FEATURE_COLS in the training notebook.
FEATURE_COLS = [
    "billing_ratio",            # billed_amount / expected_cost
    "cost_diff",                # billed_amount - expected_cost
    "high_cost_flag",           # 1 if billing_ratio > 1.5
    "provider_claim_count",     # total claims submitted by this provider
    "provider_specialty_enc",   # label-encoded provider specialty
    "severity_score",           # Mild=1 / Medium=2 / Severe=3
    "diag_claim_count", 
            # frequency of this diagnosis code in dataset
    "diag_category_enc",
    "is_billed_missing",        # 1 if billed_amount was null in bronze
    "is_proc_missing",          # 1 if procedure_code was null/UNKNOWN
    "is_diag_missing",          # 1 if diagnosis_code was null/UNKNOWN
    "claim_age_days",           # days between claim date and latest claim in dataset
]

TARGET_COL = "denial_flag"   # 0 = Denied, 1 = Approved
ID_COL     = "claim_id"
TOP_N      = 3               # Number of top SHAP drivers to capture per claim

print("=" * 65)
print("  SHAP EXPLAINABILITY PIPELINE — CONFIGURATION")
print("=" * 65)
print(f"  SHAP version         : {shap.__version__}")
print(f"  MLflow experiment    : {EXPERIMENT_NAME}")
print(f"  XGBoost Run ID       : {XGB_RUN_ID}")
print(f"  Feature count        : {len(FEATURE_COLS)}")
print(f"  Top-N reasons/claim  : {TOP_N}")
print(f"  Target encoding      : 0 = Denied  |  1 = Approved")
print("=" * 65)

# ── Guard: catch placeholder before wasting compute ───────────────────────────
if XGB_RUN_ID == "REPLACE_WITH_YOUR_XGB_RUN_ID":
    raise ValueError(
        "XGB_RUN_ID has not been set. "
        "Open Cell 2, paste your actual MLflow XGBoost run ID, and re-run."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Load `gold_claim_features`

# COMMAND ----------

feat_spark = spark.table("workspace.gold.gold_claim_features")
feat_pd    = feat_spark.toPandas()

# Validate that every expected feature column is present
missing_cols = [c for c in FEATURE_COLS + [TARGET_COL, ID_COL] if c not in feat_pd.columns]
if missing_cols:
    raise ValueError(
        f"The following expected columns are missing from gold_claim_features: {missing_cols}\n"
        "Re-run the feature engineering notebook and ensure FEATURE_COLS stays in sync."
    )

X   = feat_pd[FEATURE_COLS].copy()
y   = feat_pd[TARGET_COL].copy()
ids = feat_pd[ID_COL].copy()

# Null guard — every feature must be clean before SHAP
null_counts = X.isnull().sum()
null_features = null_counts[null_counts > 0]
if not null_features.empty:
    raise ValueError(
        f"Null values detected in feature table — SHAP requires a clean input.\n"
        f"Affected columns:\n{null_features}"
    )

print(f"  Loaded gold_claim_features : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"  Denied   (0)               : {int((y == 0).sum()):,}")
print(f"  Approved (1)               : {int((y == 1).sum()):,}")
print("  ✅ No nulls detected in feature columns.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Load XGBoost Model from MLflow & Run Inference
# MAGIC
# MAGIC We load the sklearn-wrapper flavour of the XGBoost model so that `predict_proba` is
# MAGIC available. The raw booster (`get_booster()`) does **not** expose `predict_proba`.

# COMMAND ----------

model_uri = f"runs:/{XGB_RUN_ID}/model"
print(f"  Loading model from : {model_uri}")

# Load as XGBoost sklearn wrapper — this gives us predict / predict_proba
xgb_model = mlflow.xgboost.load_model(model_uri)
print("  ✅ Model loaded successfully.")

# Run inference across the full feature table
# y_prob_all[:, 1] = P(Approved);  1 - P(Approved) = P(Denied)
y_prob_all = xgb_model.predict_proba(X)[:, 1]   # shape: (n_claims,)
y_pred_all = xgb_model.predict(X)               # shape: (n_claims,)  — 0 or 1

print(f"\n  Predictions generated for {len(y_pred_all):,} claims.")
print(f"    Predicted Denied   (0) : {int((y_pred_all == 0).sum()):,}")
print(f"    Predicted Approved (1) : {int((y_pred_all == 1).sum()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Compute SHAP Values (TreeExplainer)
# MAGIC
# MAGIC ### XGBoost 3.x + SHAP Compatibility Fix
# MAGIC XGBoost ≥ 3.0 serialises `base_score` as an array string (e.g., `"[0.5]"`) inside the
# MAGIC booster config JSON. Older SHAP versions expect a plain float string (`"0.5"`), which causes
# MAGIC a `ValueError` when initialising `TreeExplainer`. The fix below patches the booster config
# MAGIC in-memory before passing it to SHAP — no model retraining required.
# MAGIC
# MAGIC ### Output shape note
# MAGIC For binary XGBoost classifiers, `shap_values()` returns shape `(n_claims, n_features)`.
# MAGIC Some SHAP versions return a list of two arrays `[shap_class_0, shap_class_1]`; the code
# MAGIC below handles both cases safely.

# COMMAND ----------

print("  Initialising SHAP TreeExplainer...")

# ── XGBoost 3.x base_score compatibility patch ────────────────────────────────
booster = xgb_model.get_booster()
config  = json.loads(booster.save_config())

raw_base_score = config["learner"]["learner_model_param"].get("base_score", "0.5")
# Handles both plain "0.5" and array "[0.5]" formats produced by XGBoost 3.x
if isinstance(raw_base_score, str) and raw_base_score.startswith("["):
    try:
        base_score_float = float(json.loads(raw_base_score)[0])
    except (json.JSONDecodeError, IndexError, TypeError):
        base_score_float = 0.5
    config["learner"]["learner_model_param"]["base_score"] = str(base_score_float)
    booster.load_config(json.dumps(config))
    print(f"  ⚙  base_score patched: '{raw_base_score}' → '{base_score_float}'")
else:
    print(f"  ⚙  base_score format OK ('{raw_base_score}') — no patch needed.")

# ── Build TreeExplainer on the patched booster ────────────────────────────────
explainer = shap.TreeExplainer(booster)

print(f"  Computing SHAP values for {X.shape[0]:,} claims × {X.shape[1]} features...")
raw_shap = explainer.shap_values(X)

# ── Shape normalisation ───────────────────────────────────────────────────────
# SHAP returns either:
#   (a) ndarray of shape (n_claims, n_features)          — newer API
#   (b) list of two arrays [(n_claims, n_features), ...]  — older API, one per class
# We always want the class-1 (Approved) perspective; class-0 is its exact negative.
if isinstance(raw_shap, list):
    # Binary classifier list: index 1 = class 1 (Approved direction)
    shap_values = raw_shap[1]
    print("  ℹ  SHAP returned list format — using class-1 slice.")
elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
    # Some versions stack as (n_claims, n_features, n_classes)
    shap_values = raw_shap[:, :, 1]
    print("  ℹ  SHAP returned 3-D array — slicing class-1.")
else:
    # Standard 2-D array (n_claims, n_features)
    shap_values = raw_shap
    print("  ℹ  SHAP returned standard 2-D array.")

assert shap_values.shape == (len(X), len(FEATURE_COLS)), (
    f"Unexpected SHAP matrix shape: {shap_values.shape}. "
    f"Expected ({len(X)}, {len(FEATURE_COLS)})."
)

print(f"  ✅ SHAP values computed. Matrix shape: {shap_values.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Business Reason Mapping
# MAGIC
# MAGIC Each feature maps to a **(denial_message, approval_message)** pair.
# MAGIC The correct message is selected based on the **sign** of the SHAP value:
# MAGIC - **Negative SHAP** → feature is pushing the claim toward `0` (Denied) → use denial_message
# MAGIC - **Positive SHAP** → feature is pushing the claim toward `1` (Approved) → use approval_message

# COMMAND ----------

# Keys must exactly match the strings in FEATURE_COLS above.
REASON_MAP = {
    "billing_ratio": (
        "Claim amount is significantly higher than the benchmark expected cost.",
        "Claim amount is within the accepted benchmark cost range.",
    ),
    "cost_diff": (
        "The absolute cost gap between billed and expected amounts is excessively large.",
        "The cost gap between billed and expected amounts is within acceptable limits.",
    ),
    "high_cost_flag": (
        "Claim has been flagged as an extreme high-cost outlier by the cost model.",
        "No high-cost outlier flag detected; billing appears standard.",
    ),
    "provider_claim_count": (
        "The provider's low historical claim volume indicates a higher operational risk profile.",
        "The provider has a high historical claim volume, suggesting a reliable submission pattern.",
    ),
    "provider_specialty_enc": (
        "The billing pattern is inconsistent with the provider's recorded medical specialty.",
        "The billing pattern is consistent with the provider's medical specialty.",
    ),
    "severity_score": (
        "The clinical severity level is inconsistent with the standard billing profile for this claim.",
        "The clinical severity level aligns with the expected billing profile.",
    ),
    "diag_claim_count": (
        "This diagnosis code has an unusually low historical claim frequency, indicating potential miscoding.",
        "This diagnosis code has a strong historical claim frequency, indicating a reliable submission.",
    ),
    "is_billed_missing": (
        "The claim billed amount was missing from the original source submission.",
        "The claim billed amount is present and valid in the original submission.",
    ),
    "is_proc_missing": (
        "The medical procedure code is missing or invalid in the claim submission.",
        "The medical procedure code is present and valid.",
    ),
    "is_diag_missing": (
        "Diagnosis is missing — the medical diagnosis code is absent or invalid in the claim submission.",
        "The medical diagnosis code is present and valid.",
    ),
    "claim_age_days": (
        "The claim was submitted significantly late relative to the service date.",
        "The claim was submitted promptly relative to the service date.",
    ),
}

# Validate coverage — every feature in FEATURE_COLS must have a mapping
unmapped = [f for f in FEATURE_COLS if f not in REASON_MAP]
if unmapped:
    raise ValueError(
        f"The following features are missing from REASON_MAP: {unmapped}\n"
        "Add a (denial_msg, approval_msg) tuple for each before proceeding."
    )

def get_reason_text(feature_name: str, shap_value: float) -> str:
    """
    Returns the correct business reason message for a given feature,
    based on whether its SHAP value is driving toward Denial (negative)
    or Approval (positive).
    Falls back to a generic message for any feature not in REASON_MAP.
    """
    denial_msg, approval_msg = REASON_MAP.get(
        feature_name,
        (
            f"Feature '{feature_name}' contributed to the denial decision.",
            f"Feature '{feature_name}' supported the approval decision.",
        ),
    )
    return denial_msg if shap_value < 0 else approval_msg

print(f"  ✅ REASON_MAP validated. {len(REASON_MAP)}/{len(FEATURE_COLS)} features fully mapped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Build Explanation Rows (Top-3 SHAP Drivers per Claim)
# MAGIC
# MAGIC For each claim we:
# MAGIC 1. Determine the predicted outcome (DENIED / APPROVED) and the denial probability score.
# MAGIC 2. Rank all feature SHAP values by **absolute magnitude** (largest impact first).
# MAGIC 3. Take the Top 3 ranked features.
# MAGIC 4. Look up the sign-aware business reason text for each.
# MAGIC 5. Store the feature name, reason text, and raw SHAP impact for auditability.

# COMMAND ----------

PROCESSED_AT = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
rows = []

for i in range(len(X)):
    claim_id          = ids.iloc[i]
    denial_probability = round(float(1.0 - y_prob_all[i]), 4)   # P(Denied)
    predicted_status  = "DENIED" if y_pred_all[i] == 0 else "APPROVED"

    # SHAP values for this single claim — shape: (n_features,)
    claim_shap = shap_values[i]

    # Sort feature indices by descending absolute SHAP magnitude
    sorted_indices = np.argsort(np.abs(claim_shap))[::-1]
    top_indices    = sorted_indices[:TOP_N]

    row = {
        "claim_id"           : claim_id,
        "denial_probability" : denial_probability,
        "predicted_status"   : predicted_status,
        "processed_at"       : PROCESSED_AT,
    }

    for rank, feat_idx in enumerate(top_indices, start=1):
        feat_name  = FEATURE_COLS[feat_idx]
        shap_val   = round(float(claim_shap[feat_idx]), 6)
        reason_txt = get_reason_text(feat_name, shap_val)

        row[f"reason_{rank}_feature"] = feat_name
        row[f"reason_{rank}_text"]    = reason_txt
        row[f"reason_{rank}_impact"]  = shap_val   # negative = denial driver

    rows.append(row)

explanations_pd = pd.DataFrame(rows)

# Enforce a consistent column order for the Delta schema
col_order = [
    "claim_id",
    "denial_probability",
    "predicted_status",
    "reason_1_feature", "reason_1_text", "reason_1_impact",
    "reason_2_feature", "reason_2_text", "reason_2_impact",
    "reason_3_feature", "reason_3_text", "reason_3_impact",
    "processed_at",
]
explanations_pd = explanations_pd[col_order]

n_denied   = int((explanations_pd["predicted_status"] == "DENIED").sum())
n_approved = int((explanations_pd["predicted_status"] == "APPROVED").sum())

print("=" * 65)
print("  EXPLANATION TABLE BUILD — COMPLETE")
print("=" * 65)
print(f"  Total rows         : {len(explanations_pd):,}")
print(f"  Predicted DENIED   : {n_denied:,}")
print(f"  Predicted APPROVED : {n_approved:,}")
print(f"  Columns            : {list(explanations_pd.columns)}")
print()
print("  Sample (first 3 rows):")
display(
    explanations_pd[[
        "claim_id", "predicted_status", "denial_probability",
        "reason_1_feature", "reason_1_text", "reason_1_impact"
    ]].head(3)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Write `gold_claim_explanations` to Gold Layer

# COMMAND ----------

# Convert Pandas → Spark for Delta write
explanations_spark = spark.createDataFrame(explanations_pd)

print("  Schema of gold_claim_explanations:")
explanations_spark.printSchema()

(
    explanations_spark
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.gold.gold_claim_explanations")
)

written_count = spark.table("workspace.gold.gold_claim_explanations").count()
print(f"\n  ✅ workspace.gold.gold_claim_explanations written — {written_count:,} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Verify: Read Back & Spot-Check

# COMMAND ----------

verify_df = spark.table("workspace.gold.gold_claim_explanations")

print("=" * 65)
print("  VERIFICATION : workspace.gold.gold_claim_explanations")
print("=" * 65)
print(f"  Row count : {verify_df.count():,}")
print()

print("  --- Top 5 DENIED Claims (highest denial probability) ---")
(
    verify_df
    .filter("predicted_status = 'DENIED'")
    .select(
        "claim_id", "denial_probability",
        "reason_1_feature", "reason_1_text", "reason_1_impact",
        "reason_2_feature", "reason_2_text", "reason_2_impact",
        "reason_3_feature", "reason_3_text", "reason_3_impact",
    )
    .orderBy("denial_probability", ascending=False)
    .limit(5)
    .show(truncate=65)
)

print("  --- Top 5 APPROVED Claims (lowest denial probability) ---")
(
    verify_df
    .filter("predicted_status = 'APPROVED'")
    .select(
        "claim_id", "denial_probability",
        "reason_1_feature", "reason_1_text", "reason_1_impact",
        "reason_2_feature", "reason_2_text",
    )
    .orderBy("denial_probability")
    .limit(5)
    .show(truncate=65)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Aggregate: Top Primary Denial Drivers
# MAGIC
# MAGIC Which features are most often the **#1 driver** of a denied claim?
# MAGIC This gives operations teams a systemic view of where submissions are failing.

# COMMAND ----------

denied_pd = explanations_pd[explanations_pd["predicted_status"] == "DENIED"].copy()

top_drivers = (
    denied_pd["reason_1_feature"]
    .value_counts()
    .reset_index()
)
top_drivers.columns = ["primary_denial_feature", "count"]
top_drivers["pct_of_denied_claims"] = (
    top_drivers["count"] / len(denied_pd) * 100
).round(1)

# Add the human-readable denial reason text for the summary
top_drivers["business_reason"] = top_drivers["primary_denial_feature"].apply(
    lambda f: REASON_MAP.get(f, (f"Feature: {f}", ""))[0]
)

print("=" * 65)
print("  TOP PRIMARY DENIAL DRIVERS ACROSS ALL DENIED CLAIMS")
print("=" * 65)
print(top_drivers[["primary_denial_feature", "count", "pct_of_denied_claims", "business_reason"]]
      .to_string(index=False))
print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11 — Global Feature Importance (Mean |SHAP|)
# MAGIC
# MAGIC The mean absolute SHAP value per feature across all claims gives the global
# MAGIC importance ranking — which features matter most to the model overall, not
# MAGIC just for a single claim.

# COMMAND ----------

mean_abs_shap = pd.DataFrame({
    "feature"          : FEATURE_COLS,
    "mean_abs_shap"    : np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

mean_abs_shap["rank"] = mean_abs_shap.index + 1
mean_abs_shap["mean_abs_shap"] = mean_abs_shap["mean_abs_shap"].round(6)

print("=" * 55)
print("  GLOBAL FEATURE IMPORTANCE (Mean |SHAP| across all claims)")
print("=" * 55)
print(mean_abs_shap[["rank", "feature", "mean_abs_shap"]].to_string(index=False))
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12 — Pipeline Summary

# COMMAND ----------

print("=" * 65)
print("  WEEK 5 — EXPLAINABLE AI PIPELINE — COMPLETE")
print("=" * 65)
print(f"  Input table        : workspace.gold.gold_claim_features")
print(f"  Output table       : workspace.gold.gold_claim_explanations")
print(f"  Total claims       : {len(explanations_pd):,}")
print(f"  Predicted DENIED   : {n_denied:,}")
print(f"  Predicted APPROVED : {n_approved:,}")
print(f"  XAI method         : SHAP TreeExplainer (exact, not approximate)")
print(f"  Model              : XGBoost (Tuned) — run ID: {XGB_RUN_ID}")
print(f"  Top-N reasons      : {TOP_N} per claim")
print(f"  Processed at       : {PROCESSED_AT} UTC")
print()
print("  Output schema:")
print("    claim_id            — Primary key (from gold_claim_features)")
print("    denial_probability  — P(Denied) = 1 - P(Approved)")
print("    predicted_status    — DENIED | APPROVED")
print("    reason_1_feature    — Top SHAP driver (technical column name)")
print("    reason_1_text       — Top SHAP driver (business-readable explanation)")
print("    reason_1_impact     — SHAP value (negative = denial driver)")
print("    reason_2_* / reason_3_*  — Same structure for 2nd and 3rd drivers")
print("    processed_at        — Pipeline run timestamp (UTC)")
print("=" * 65)
