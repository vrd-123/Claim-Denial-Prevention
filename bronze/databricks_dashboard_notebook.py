# Databricks notebook source
# MAGIC %md
# MAGIC # Week 2 — Step 5 of 5: Bronze Layer Analytics Dashboard
# MAGIC
# MAGIC ## Full Week 2 Pipeline
# MAGIC | Step | Notebook | What it does |
# MAGIC |------|----------|--------------|
# MAGIC | 1 | `03_eda_bronze` | Distributions, outliers, cross-dataset exploration |
# MAGIC | 2+3 | `04_sql_analysis_bronze` | SQL aggregations + basic joins |
# MAGIC | 4 | `05_views_bronze` | Persistent SQL Views |
# MAGIC | **5** | **`06_dashboard_bronze`** ← *you are here* | **Analytics Dashboard** |
# MAGIC
# MAGIC ## How to use this notebook as a Dashboard
# MAGIC Since we are on **Databricks Community Edition**, we use the built-in **Notebook Dashboard** feature:
# MAGIC 1. Run **all cells** in this notebook (Cell → Run All)
# MAGIC 2. Click **View** in the top menu → **+ New Dashboard**
# MAGIC 3. Databricks will open a visual editor showing all the `display()` outputs as draggable tiles
# MAGIC 4. Arrange tiles, hide code cells, and save the layout
# MAGIC
# MAGIC > **Note:** This gives you a live dashboard that refreshes every time you re-run the notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Setup: Load Views & Bronze Tables

# COMMAND ----------

from pyspark.sql.functions import (
    col, count, avg, sum as spark_sum, round as spark_round,
    when, isnan, date_format, to_date, percentile_approx
)

# Load from Persistent Views (created in Step 4)
df_specialty = spark.table("default.vw_claims_by_specialty")
df_region    = spark.table("default.vw_claims_by_region")
df_high_cost = spark.table("default.vw_high_cost_claims")
df_missing   = spark.table("default.vw_missing_billed_claims")

# Load Bronze tables directly for KPIs and charts not covered by views
df_claims    = spark.table("default.bronze_claims_raw")
df_providers = spark.table("default.bronze_provider_raw")
df_diagnosis = spark.table("default.bronze_diagnosis_raw")
df_cost      = spark.table("default.bronze_cost_raw")

# Register for SQL cells
df_claims.createOrReplaceTempView("claims")
df_providers.createOrReplaceTempView("providers")
df_diagnosis.createOrReplaceTempView("diagnosis")
df_cost.createOrReplaceTempView("cost")

print("All views and tables loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 1 — KPI Summary Cards
# MAGIC *High-level numbers for total claims, providers, billing and data quality*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                                                           AS total_claims,
# MAGIC   COUNT(DISTINCT patient_id)                                         AS unique_patients,
# MAGIC   COUNT(DISTINCT provider_id)                                        AS unique_providers,
# MAGIC   ROUND(SUM(billed_amount), 0)                                       AS total_billed,
# MAGIC   ROUND(AVG(billed_amount), 0)                                       AS avg_billed_per_claim,
# MAGIC   COUNT(CASE WHEN billed_amount IS NULL THEN 1 END)                  AS missing_billed_count,
# MAGIC   ROUND(100.0 * COUNT(CASE WHEN billed_amount IS NULL THEN 1 END)
# MAGIC         / COUNT(*), 1)                                               AS missing_billed_pct,
# MAGIC   ROUND(100.0 * COUNT(CASE WHEN billed_amount IS NOT NULL THEN 1 END)
# MAGIC         / COUNT(*), 1)                                               AS data_completeness_pct
# MAGIC FROM claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 2 — Claims by Provider Specialty
# MAGIC *Which specialties drive the highest claim volume?*
# MAGIC *Switch the output to a **Bar Chart** for a visual representation.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT specialty, total_claims, avg_billed, total_billed, missing_amount_count
# MAGIC FROM default.vw_claims_by_specialty
# MAGIC WHERE specialty IS NOT NULL
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 3 — Claims by Region + Billing Gap
# MAGIC *Which regions overbill the most relative to the expected benchmark?*
# MAGIC *Switch to a **Bar Chart** and add `gap_pct` as a secondary metric.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT region, total_claims, avg_billed, avg_expected, total_billed, gap_pct
# MAGIC FROM default.vw_claims_by_region
# MAGIC WHERE region IS NOT NULL
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 4 — Claims by Diagnosis Category
# MAGIC *Which clinical categories have the highest claim frequency and average cost?*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.category,
# MAGIC   d.severity,
# MAGIC   COUNT(c.claim_id)              AS total_claims,
# MAGIC   ROUND(AVG(c.billed_amount), 2) AS avg_billed
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC LEFT JOIN workspace.default.bronze_diagnosis_raw d ON c.diagnosis_code = d.diagnosis_code
# MAGIC WHERE d.category IS NOT NULL
# MAGIC GROUP BY d.category, d.severity
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 5 — Monthly Claims Volume & Cost Trend
# MAGIC *Are claim volumes and billing amounts growing over time?*
# MAGIC *Switch to a **Line Chart** with two Y-axes for claims and billed amount.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   DATE_FORMAT(TO_DATE(date), 'yyyy-MM')  AS month,
# MAGIC   COUNT(claim_id)                         AS total_claims,
# MAGIC   ROUND(SUM(billed_amount), 0)            AS total_billed
# MAGIC FROM workspace.default.bronze_claims_raw
# MAGIC WHERE date IS NOT NULL
# MAGIC GROUP BY DATE_FORMAT(TO_DATE(date), 'yyyy-MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 6 — Billing Ratio: Billed vs Expected Cost
# MAGIC *Claims overpriced by more than 1.5x the benchmark — the PRIMARY denial signal.*
# MAGIC *Switch to a **Bar Chart**: X = provider, Y = avg_billing_ratio.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   p.specialty,
# MAGIC   COUNT(c.claim_id)                                             AS total_claims,
# MAGIC   ROUND(AVG(c.billed_amount / NULLIF(co.expected_cost, 0)), 2) AS avg_billing_ratio,
# MAGIC   COUNT(CASE WHEN c.billed_amount > 1.5 * co.expected_cost THEN 1 END) AS overpriced_count,
# MAGIC   ROUND(100.0 * COUNT(CASE WHEN c.billed_amount > 1.5 * co.expected_cost THEN 1 END)
# MAGIC         / COUNT(c.claim_id), 1)                                AS overpriced_pct
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC LEFT JOIN workspace.default.bronze_cost_raw co   ON c.procedure_code      = co.procedure_code
# MAGIC LEFT JOIN workspace.default.bronze_provider_raw p  ON c.provider_id  = p.provider_id
# MAGIC WHERE co.expected_cost IS NOT NULL AND co.expected_cost > 0
# MAGIC GROUP BY p.specialty
# MAGIC ORDER BY overpriced_pct DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 7 — Denial Risk Distribution
# MAGIC *Pre-computed preliminary denial risk signal (before full Silver risk scoring).*
# MAGIC *Each claim is given a basic risk category based on available Bronze data.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CASE
# MAGIC     WHEN billed_amount IS NULL                             THEN 'High Risk  — Missing Bill'
# MAGIC     WHEN billed_amount > 1.5 * co.expected_cost          THEN 'High Risk  — Overpriced'
# MAGIC     WHEN provider_id IS NULL OR patient_id IS NULL        THEN 'Medium Risk — Missing Identity'
# MAGIC     ELSE                                                       'Low Risk   — Normal'
# MAGIC   END                       AS risk_category,
# MAGIC   COUNT(c.claim_id)         AS total_claims
# MAGIC FROM workspace.default.bronze_claims_raw c
# MAGIC LEFT JOIN workspace.default.bronze_cost_raw co ON c.procedure_code = co.procedure_code
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 8 — Data Quality Summary
# MAGIC *Null counts and completeness % for every key column in the Bronze claims table.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   'claim_id'       AS column_name, COUNT(CASE WHEN claim_id IS NULL THEN 1 END)       AS null_count, ROUND(100.0*COUNT(CASE WHEN claim_id IS NULL THEN 1 END)/COUNT(*),1) AS null_pct FROM claims
# MAGIC UNION ALL SELECT 'patient_id',    COUNT(CASE WHEN patient_id IS NULL THEN 1 END),    ROUND(100.0*COUNT(CASE WHEN patient_id IS NULL THEN 1 END)/COUNT(*),1)    FROM claims
# MAGIC UNION ALL SELECT 'provider_id',   COUNT(CASE WHEN provider_id IS NULL THEN 1 END),   ROUND(100.0*COUNT(CASE WHEN provider_id IS NULL THEN 1 END)/COUNT(*),1)   FROM claims
# MAGIC UNION ALL SELECT 'proc_id',       COUNT(CASE WHEN proc_id IS NULL THEN 1 END),       ROUND(100.0*COUNT(CASE WHEN proc_id IS NULL THEN 1 END)/COUNT(*),1)       FROM claims
# MAGIC UNION ALL SELECT 'billed_amount', COUNT(CASE WHEN billed_amount IS NULL THEN 1 END), ROUND(100.0*COUNT(CASE WHEN billed_amount IS NULL THEN 1 END)/COUNT(*),1) FROM claims
# MAGIC ORDER BY null_pct DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 9 — High-Cost Claims Table (Top 20 · ≥ 90th Percentile)
# MAGIC *The most expensive claims in the dataset — highest risk for denial review.*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   claim_id,
# MAGIC   patient_id,
# MAGIC   doctor_name,
# MAGIC   specialty,
# MAGIC   diagnosis_category,
# MAGIC   diagnosis_severity,
# MAGIC   billed_amount,
# MAGIC   expected_cost,
# MAGIC   billing_ratio,
# MAGIC   date
# MAGIC FROM default.vw_high_cost_claims
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tile 10 — Missing Billed Amount: Imputation Readiness
# MAGIC *Of the 34.3% claims missing a billed amount — how many can be imputed from the cost table?*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                                                    AS total_missing_billed,
# MAGIC   COUNT(CASE WHEN expected_cost IS NOT NULL THEN 1 END)      AS can_impute_count,
# MAGIC   COUNT(CASE WHEN expected_cost IS NULL     THEN 1 END)      AS cannot_impute_count,
# MAGIC   ROUND(100.0 * COUNT(CASE WHEN expected_cost IS NOT NULL THEN 1 END) / COUNT(*), 1) AS imputable_pct
# MAGIC FROM default.vw_missing_billed_claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Dashboard Summary
# MAGIC
# MAGIC | Tile | Chart Type | Key Insight |
# MAGIC |------|------------|-------------|
# MAGIC | 1 | KPI Cards | Total claims, providers, billed amount, data completeness |
# MAGIC | 2 | Bar Chart | Claims volume by specialty |
# MAGIC | 3 | Bar Chart | Billing gap (actual vs expected) by region |
# MAGIC | 4 | Bar Chart | Claims by diagnosis category + severity |
# MAGIC | 5 | Line Chart | Monthly claim volume & cost trend |
# MAGIC | 6 | Bar Chart | Overpriced claims % by specialty (denial signal) |
# MAGIC | 7 | Bar/Pie Chart | Pre-computed denial risk distribution |
# MAGIC | 8 | Table | Data quality: null % per column |
# MAGIC | 9 | Table | Top 20 high-cost claims (≥ 90th percentile) |
# MAGIC | 10 | KPI | Missing billed amount imputation readiness |
# MAGIC
# MAGIC > **Week 2 COMPLETE ✅**
# MAGIC >
# MAGIC > **Next:** Week 3 — Silver Layer Cleaning
# MAGIC > - Handle null `billed_amount` (impute with per-procedure median + `is_billed_missing` flag)
# MAGIC > - Remove duplicate claims
# MAGIC > - Fix data types (dates, amounts)
# MAGIC > - Standardize ICD/CPT codes and text casing
# MAGIC > - Write `silver_claims`, `silver_provider`, `silver_diagnosis`, `silver_cost` Delta tables
