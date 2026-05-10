# Databricks notebook source
# MAGIC %md
# MAGIC # Week 2 — Steps 2 & 3 of 5: SQL Analysis + Basic Joins
# MAGIC
# MAGIC ## Full Week 2 Pipeline
# MAGIC | Step | Notebook | What it does |
# MAGIC |------|----------|--------------|
# MAGIC | 1 | `03_eda_bronze` | Top rows, unique values, distributions |
# MAGIC | **2** | **`04_sql_analysis_bronze`** ← *you are here* | **SQL aggregations: total claims, per provider, per diagnosis** |
# MAGIC | **3** | **`04_sql_analysis_bronze`** ← *you are here* | **Basic joins: claims+provider, claims+diagnosis** |
# MAGIC | 4 | `05_views_bronze` | Persistent SQL Views: by specialty, region, high-cost |
# MAGIC | 5 | `06_dashboard_bronze` | Databricks native dashboard |
# MAGIC
# MAGIC > **Input:** Bronze managed Delta tables · **Output:** Aggregated summary tables + decision on `billed_amount` null strategy for Week 3

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1 — Setup: Load Bronze Tables

# COMMAND ----------

df_claims    = spark.table("default.bronze_claims_raw")
df_providers = spark.table("default.bronze_provider_raw")
df_diagnosis = spark.table("default.bronze_diagnosis_raw")
df_cost      = spark.table("default.bronze_cost_raw")

# Register as temp views for direct SQL usage in this session
df_claims.createOrReplaceTempView("claims")
df_providers.createOrReplaceTempView("providers")
df_diagnosis.createOrReplaceTempView("diagnosis")
df_cost.createOrReplaceTempView("cost")

print("Bronze tables loaded and registered as temp views.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2 — Total Volume Summary
# MAGIC *How big is this dataset overall?*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- High-level dataset overview: volumes, unique entities, date coverage
# MAGIC SELECT
# MAGIC   COUNT(*)                     AS total_claims,
# MAGIC   COUNT(DISTINCT patient_id)   AS unique_patients,
# MAGIC   COUNT(DISTINCT provider_id)  AS unique_providers,
# MAGIC   MIN(date)                    AS date_from,
# MAGIC   MAX(date)                    AS date_to
# MAGIC FROM claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3 — ⚠️ CRITICAL: billed_amount Null Analysis & Silver Strategy Decision
# MAGIC
# MAGIC Before any business analysis, we first determine what % of claims have a missing `billed_amount`.
# MAGIC The result of this query **drives the Silver layer cleaning strategy**.
# MAGIC
# MAGIC | Null % | Decision | Rationale |
# MAGIC |--------|----------|-----------|
# MAGIC | < 5% | **DROP the null rows** | Loss is negligible; Silver stays 100% auditable |
# MAGIC | 5% – 15% | **Impute using per-procedure median** + flag `is_imputed=True` | Acceptable trade-off to preserve data volume |
# MAGIC | > 15% | **Escalate to business team** | Too much data missing; imputation would distort results |

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Step 1: Count and percentage of null billed_amount rows
# MAGIC SELECT
# MAGIC   COUNT(*)                                                          AS total_claims,
# MAGIC   COUNT(CASE WHEN billed_amount IS NULL THEN 1 END)                AS null_billed_count,
# MAGIC   ROUND(
# MAGIC     100.0 * COUNT(CASE WHEN billed_amount IS NULL THEN 1 END)
# MAGIC     / COUNT(*), 2
# MAGIC   )                                                                 AS null_billed_pct
# MAGIC FROM claims;

# COMMAND ----------

# Apply the decision rule in PySpark based on the actual result above
from pyspark.sql.functions import col, count, when, round as spark_round, lit, expr, percentile_approx

total_rows       = df_claims.count()
null_rows        = df_claims.filter(col("billed_amount").isNull()).count()
null_pct         = round((null_rows / total_rows) * 100, 2)

print(f"Total claims       : {total_rows:,}")
print(f"Null billed_amount : {null_rows:,}  ({null_pct}%)")
print()

# ── DECISION LOGIC ───────────────────────────────────────────────────────────
if null_pct < 5:
    decision = "DROP"
    rationale = (
        f"Only {null_pct}% of rows are missing billed_amount. "
        "Dropping is safe — data loss is negligible and Silver stays fully auditable. "
        "No imputation (i.e., no data alteration) is needed."
    )
elif null_pct <= 15:
    decision = "IMPUTE"
    rationale = (
        f"{null_pct}% of rows are missing billed_amount. "
        "Dropping would cause significant data loss. "
        "We will impute using the MEDIAN billed_amount per procedure_code "
        "and add an `is_billed_imputed = True` flag to maintain full transparency."
    )
else:
    decision = "ESCALATE"
    rationale = (
        f"{null_pct}% of rows are missing billed_amount — this is too high to fix automatically. "
        "This must be escalated to the data owner / business team for review "
        "before proceeding to the Silver layer."
    )

print(f"Silver Layer Decision → {decision}")
print(f"Rationale            : {rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4 — Claims Per Provider
# MAGIC *Which providers submit the most claims and what do they bill on average?*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Claims per provider: volume + financial metrics
# MAGIC SELECT
# MAGIC   c.provider_id,
# MAGIC   p.doctor_name,
# MAGIC   p.specialty,
# MAGIC   p.location,
# MAGIC   COUNT(c.claim_id)                              AS total_claims,
# MAGIC   ROUND(AVG(CAST(c.billed_amount AS DOUBLE)), 2) AS avg_billed,
# MAGIC   ROUND(SUM(CAST(c.billed_amount AS DOUBLE)), 2) AS total_billed
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC LEFT JOIN workspace.default.bronze_provider_raw p ON c.provider_id = p.provider_id
# MAGIC GROUP BY c.provider_id, p.doctor_name, p.specialty, p.location
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# Save as a Delta summary table so the Dashboard notebook can query it directly
spark.sql("""
  SELECT
    c.provider_id,
    p.doctor_name,
    p.specialty,
    p.location,
    COUNT(c.claim_id)              AS total_claims,
    ROUND(AVG(c.billed_amount), 2) AS avg_billed,
    ROUND(SUM(c.billed_amount), 2) AS total_billed
  FROM workspace.default.bronze_claims_raw c
  LEFT JOIN workspace.default.bronze_provider_raw p ON c.provider_id = p.provider_id
  GROUP BY c.provider_id, p.doctor_name, p.specialty, p.location
""").write.format("delta").mode("overwrite").saveAsTable("default.summary_claims_per_provider")

print("Saved → default.summary_claims_per_provider")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 5 — Average Billed Amount & Distribution
# MAGIC *What is the statistical profile of billing amounts?*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Full statistical breakdown of billed_amount
# MAGIC SELECT
# MAGIC   COUNT(billed_amount)                        AS claims_with_amount,
# MAGIC   COUNT(CASE WHEN billed_amount IS NULL THEN 1 END) AS missing_amount,
# MAGIC   ROUND(MIN(billed_amount), 2)                AS min_billed,
# MAGIC   ROUND(PERCENTILE_APPROX(billed_amount, 0.25), 2) AS q1,
# MAGIC   ROUND(PERCENTILE_APPROX(billed_amount, 0.50), 2) AS median,
# MAGIC   ROUND(AVG(billed_amount), 2)                AS mean,
# MAGIC   ROUND(PERCENTILE_APPROX(billed_amount, 0.75), 2) AS q3,
# MAGIC   ROUND(MAX(billed_amount), 2)                AS max_billed,
# MAGIC   COUNT(CASE WHEN billed_amount < 0 THEN 1 END)    AS negative_amounts,
# MAGIC   COUNT(CASE WHEN billed_amount > 100000 THEN 1 END) AS extreme_outliers
# MAGIC FROM workspace.default.bronze_claims_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 6 — Claims Per Diagnosis
# MAGIC *Which diagnoses are most common and what is their average cost?*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Claims by diagnosis with clinical metadata
# MAGIC SELECT
# MAGIC   c.diagnosis_code,
# MAGIC   d.category,
# MAGIC   d.severity,
# MAGIC   COUNT(c.claim_id)              AS total_claims,
# MAGIC   ROUND(AVG(c.billed_amount), 2) AS avg_billed
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC LEFT JOIN workspace.default.bronze_diagnosis_raw d ON c.diagnosis_code = d.diagnosis_code
# MAGIC GROUP BY c.diagnosis_code, d.category, d.severity
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# Save for dashboard
spark.sql("""
  SELECT
    c.diagnosis_code,
    d.category,
    d.severity,
    COUNT(c.claim_id)              AS total_claims,
    ROUND(AVG(c.billed_amount), 2) AS avg_billed
  FROM workspace.default.bronze_claims_raw c
  LEFT JOIN workspace.default.bronze_diagnosis_raw d ON c.diagnosis_code = d.diagnosis_code
  GROUP BY c.diagnosis_code, d.category, d.severity
""").write.format("delta").mode("overwrite").saveAsTable("default.summary_claims_per_diagnosis")

print("Saved → default.summary_claims_per_diagnosis")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 7 — Basic Joins (Step 3 of Week 2)
# MAGIC
# MAGIC ### 7a. Claims + Provider (Row-level enrichment)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Every claim enriched with the provider's name, specialty, and location
# MAGIC SELECT
# MAGIC   c.claim_id,
# MAGIC   c.patient_id,
# MAGIC   p.doctor_name,
# MAGIC   p.specialty,
# MAGIC   p.location,
# MAGIC   c.billed_amount,
# MAGIC   c.date
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC LEFT JOIN workspace.default.bronze_provider_raw p ON c.provider_id = p.provider_id
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Claims + Diagnosis (Clinical context enrichment)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Every claim enriched with diagnosis category and severity
# MAGIC SELECT
# MAGIC   c.procedure_code,
# MAGIC   d.category,
# MAGIC   d.severity,
# MAGIC   c.expected_cost,
# MAGIC   c.average_cost,
# MAGIC   c.region
# MAGIC FROM workspace.default.bronze_cost_raw c
# MAGIC LEFT JOIN workspace.default.bronze_diagnosis_raw d ON c.procedure_code = d.diagnosis_code
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 8 — Financial Gap Analysis
# MAGIC *Are claims billed more than the benchmark expected cost? This is the core signal for denial detection.*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare each claim's billed_amount against the procedure's expected_cost
# MAGIC -- billing_ratio > 1.5 means the claim is "overpriced" — a strong denial signal
# MAGIC SELECT
# MAGIC   c.claim_id,
# MAGIC   c.procedure_code,
# MAGIC   c.billed_amount,
# MAGIC   co.expected_cost,
# MAGIC   ROUND(c.billed_amount / co.expected_cost, 2)    AS billing_ratio,
# MAGIC   CASE
# MAGIC     WHEN c.billed_amount > 1.5 * co.expected_cost THEN 'OVERPRICED'
# MAGIC     ELSE 'NORMAL'
# MAGIC   END                                             AS pricing_flag
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC JOIN workspace.default.bronze_cost_raw co ON c.procedure_code = co.procedure_code
# MAGIC WHERE co.expected_cost IS NOT NULL AND co.expected_cost > 0
# MAGIC ORDER BY billing_ratio DESC
# MAGIC LIMIT 25;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 9 — Analysis Summary & Handoff to Views Notebook
# MAGIC
# MAGIC ### Summary Tables Created (reusable by Dashboard)
# MAGIC | Delta Table | Contents |
# MAGIC |-------------|----------|
# MAGIC | `default.summary_claims_per_provider` | Claim volume + billing totals by provider |
# MAGIC | `default.summary_claims_per_diagnosis` | Claim volume + avg cost by diagnosis |
# MAGIC
# MAGIC ### Silver Layer Decision (set by Section 3)
# MAGIC The `billed_amount` null strategy was decided programmatically above.
# MAGIC Check the printed output of **Section 3** to confirm whether Silver will:
# MAGIC - **DROP** null rows (if null % < 5) — Zero data alteration, fully auditable
# MAGIC - **IMPUTE** by procedure median (if null % 5–15) — Flagged transparently with `is_billed_imputed`
# MAGIC - **ESCALATE** (if null % > 15) — Business decision required
# MAGIC
# MAGIC > **Next step:** Open `05_views_bronze` to create **persistent SQL Views** for specialty, region, and high-cost claim segments.
