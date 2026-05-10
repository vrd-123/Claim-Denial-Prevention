# Databricks notebook source
# MAGIC %md
# MAGIC # Week 2 — Step 1 of 5: Explore Data (EDA)
# MAGIC
# MAGIC ## Full Week 2 Pipeline
# MAGIC | Step | Notebook | What it does |
# MAGIC |------|----------|--------------|
# MAGIC | **1** | **`03_eda_bronze`** ← *you are here* | Top rows, unique values, distributions, cross-dataset relationships |
# MAGIC | 2 | `04_sql_analysis_bronze` | Total claims, claims per provider, avg billed amount, claims per diagnosis |
# MAGIC | 3 | `04_sql_analysis_bronze` | Basic joins: claims+provider, claims+diagnosis |
# MAGIC | 4 | `05_views_bronze` | Create persistent Databricks SQL Views: by specialty, by region, high-cost |
# MAGIC | 5 | `06_dashboard_bronze` | Databricks native dashboard: total claims, cost trends, provider activity |
# MAGIC
# MAGIC > **Input:** Bronze managed Delta tables · **Output:** Insights (no cleaning yet — that is Week 3)
# MAGIC
# MAGIC **Goal of this notebook:** Understand the raw data deeply enough to:
# MAGIC 1. Define the cleaning rules for the **Silver layer** (Week 3)
# MAGIC 2. Identify which columns / relationships are worth surfacing in the **Dashboard** (Step 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1 — Load Bronze Tables

# COMMAND ----------

df_claims    = spark.table("default.bronze_claims_raw")
df_providers = spark.table("default.bronze_provider_raw")
df_diagnosis = spark.table("default.bronze_diagnosis_raw")
df_cost      = spark.table("default.bronze_cost_raw")

# Register as temp views so we can query them directly with %sql cells
df_claims.createOrReplaceTempView("claims")
df_providers.createOrReplaceTempView("providers")
df_diagnosis.createOrReplaceTempView("diagnosis")
df_cost.createOrReplaceTempView("cost")

print("All Bronze tables loaded and registered as temp views.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2 — Dataset Shape & Column Overview

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Claims

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Shape and a preview of the claims table
# MAGIC SELECT * FROM claims LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Total row count and distinct key count in one query
# MAGIC SELECT
# MAGIC   COUNT(*)                    AS total_rows,
# MAGIC   COUNT(DISTINCT claim_id)    AS unique_claims,
# MAGIC   COUNT(DISTINCT provider_id) AS unique_providers,
# MAGIC   COUNT(DISTINCT patient_id)  AS unique_patients
# MAGIC FROM claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Providers

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM providers LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total_rows, COUNT(DISTINCT provider_id) AS unique_providers FROM providers;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. Diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diagnosis LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. Cost

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cost LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3 — Missing Values Analysis
# MAGIC *Identifying which columns have nulls and how severe the problem is.*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Null summary for the claims table
# MAGIC SELECT
# MAGIC   ROUND(100.0 * SUM(CASE WHEN claim_id    IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS claim_id_null_pct,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN patient_id  IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS patient_id_null_pct,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN provider_id IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS provider_id_null_pct,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN billed_amount IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS billed_amount_null_pct
# MAGIC FROM claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4 — Numeric Distributions
# MAGIC *Are billing amounts normally distributed? Are there outliers?*
# MAGIC
# MAGIC **How to visualize:** After running the cell below, click the **Chart** icon in the output and select **Histogram** to see the distribution.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of billed_amount (use Chart > Histogram in the output)
# MAGIC SELECT billed_amount FROM claims WHERE billed_amount IS NOT NULL;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quartile breakdown of billed_amount to spot skew
# MAGIC SELECT
# MAGIC   MIN(billed_amount)                                            AS min_billed,
# MAGIC   PERCENTILE_APPROX(billed_amount, 0.25)                       AS q1,
# MAGIC   PERCENTILE_APPROX(billed_amount, 0.50)                       AS median,
# MAGIC   ROUND(AVG(billed_amount), 2)                                  AS mean,
# MAGIC   PERCENTILE_APPROX(billed_amount, 0.75)                       AS q3,
# MAGIC   MAX(billed_amount)                                            AS max_billed,
# MAGIC   COUNT(CASE WHEN billed_amount < 0       THEN 1 END)          AS negative_amounts,
# MAGIC   COUNT(CASE WHEN billed_amount > 100000  THEN 1 END)          AS extreme_outliers
# MAGIC FROM claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 5 — Categorical Distributions
# MAGIC *Value distribution for key categorical columns.*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How many claims per provider? Helps spot dominant/unusual providers.
# MAGIC -- Switch chart to BAR chart in output for easier reading.
# MAGIC SELECT provider_id, COUNT(*) AS claim_count
# MAGIC FROM claims
# MAGIC WHERE provider_id IS NOT NULL
# MAGIC GROUP BY provider_id
# MAGIC ORDER BY claim_count DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of diagnosis severity in the diagnosis table
# MAGIC SELECT severity, COUNT(*) AS count
# MAGIC FROM diagnosis
# MAGIC GROUP BY severity
# MAGIC ORDER BY count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 6 — Relationship Analysis (Cross-Dataset)
# MAGIC *Does billed amount vary by provider or diagnosis severity? These are the key signals for our claim denial model.*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Average billed amount by provider — who bills the most on average?
# MAGIC SELECT
# MAGIC   c.provider_id,
# MAGIC   COUNT(*) AS total_claims,
# MAGIC   ROUND(AVG(CAST(c.billed_amount AS DOUBLE)), 2) AS avg_billed,
# MAGIC   ROUND(MIN(CAST(c.billed_amount AS DOUBLE)), 2) AS min_billed,
# MAGIC   ROUND(MAX(CAST(c.billed_amount AS DOUBLE)), 2) AS max_billed
# MAGIC FROM claims c
# MAGIC WHERE c.provider_id IS NOT NULL
# MAGIC GROUP BY c.provider_id
# MAGIC ORDER BY avg_billed DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Join claims with diagnosis to see if severity affects billing amounts
# MAGIC SELECT
# MAGIC   d.severity,
# MAGIC   COUNT(c.claim_id)              AS total_claims,
# MAGIC   ROUND(AVG(c.billed_amount), 2) AS avg_billed
# MAGIC FROM claims c
# MAGIC JOIN diagnosis d ON c.diagnosis_code = d.diagnosis_code
# MAGIC GROUP BY d.severity
# MAGIC ORDER BY avg_billed DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compare billed_amount vs expected_cost from the cost table
# MAGIC -- Overpriced claims (billed > 1.5x expected) are a key denial signal
# MAGIC SELECT
# MAGIC   c.claim_id,
# MAGIC   c.billed_amount,
# MAGIC   co.expected_cost,
# MAGIC   ROUND(c.billed_amount / co.expected_cost, 2) AS billing_ratio,
# MAGIC   CASE WHEN c.billed_amount > 1.5 * co.expected_cost THEN 'OVERPRICED' ELSE 'NORMAL' END AS pricing_flag
# MAGIC FROM claims c
# MAGIC JOIN cost co ON c.procedure_code = co.procedure_code
# MAGIC WHERE co.expected_cost IS NOT NULL AND co.expected_cost > 0
# MAGIC LIMIT 50;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 7 — Duplicate Key Check

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Any duplicate claim_ids? These would indicate a data integrity issue.
# MAGIC SELECT claim_id, COUNT(*) AS duplicate_count
# MAGIC FROM claims
# MAGIC GROUP BY claim_id
# MAGIC HAVING COUNT(*) > 1
# MAGIC ORDER BY duplicate_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 8 — EDA Summary & Silver Layer Requirements
# MAGIC
# MAGIC Based on the findings above, document the Silver layer cleaning rules here:
# MAGIC
# MAGIC | Issue Found | Silver Action |
# MAGIC |---|---|
# MAGIC | Null `patient_id` / `provider_id` / `proc_id` | Keep rows; flag as `null_identity = True` for risk scoring |
# MAGIC | Negative `billed_amount` | Drop or flag as invalid |
# MAGIC | `billed_amount` > 100,000 | Retain; these are legitimate extreme cases to investigate |
# MAGIC | Duplicate `claim_id` | Deduplicate by keeping the latest `ingestion_time` |
# MAGIC
# MAGIC > **Next step:** Open `04_sql_analysis_bronze` to run formal aggregation queries that will feed the Databricks Dashboard.
