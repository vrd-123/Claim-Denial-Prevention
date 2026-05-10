# Databricks notebook source
# MAGIC %md
# MAGIC # Week 2 — Step 4 of 5: Create Persistent SQL Views
# MAGIC
# MAGIC ## Full Week 2 Pipeline
# MAGIC | Step | Notebook | What it does |
# MAGIC |------|----------|--------------|
# MAGIC | 1 | `03_eda_bronze` | Distributions, outliers, cross-dataset exploration |
# MAGIC | 2+3 | `04_sql_analysis_bronze` | SQL aggregations + basic joins |
# MAGIC | **4** | **`05_views_bronze`** ← *you are here* | **Persistent SQL Views in Hive Metastore** |
# MAGIC | 5 | `06_dashboard_bronze` | Databricks native dashboard |
# MAGIC
# MAGIC ## What makes a "View" different from a temp view?
# MAGIC | | Temp View (`createOrReplaceTempView`) | Persistent SQL View (`CREATE OR REPLACE VIEW`) |
# MAGIC |---|---|---|
# MAGIC | **Lifetime** | Deleted when the Spark session ends | Lives permanently in the Hive Metastore |
# MAGIC | **Access** | Only in this notebook/session | Available to ALL notebooks and the Dashboard builder |
# MAGIC | **Storage** | No actual data stored — just the query definition | No actual data stored — just the query definition |
# MAGIC | **Use case** | Temporary analysis inside one notebook | Shared, reusable views for dashboards & reports |
# MAGIC
# MAGIC > **Input:** Bronze Delta tables · **Output:** 3 persistent SQL views registered in `default` database

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1 — Load Bronze Tables as Temp Views (for SQL use in this notebook)

# COMMAND ----------

df_claims    = spark.table("default.bronze_claims_raw")
df_providers = spark.table("default.bronze_provider_raw")
df_diagnosis = spark.table("default.bronze_diagnosis_raw")
df_cost      = spark.table("default.bronze_cost_raw")

df_claims.createOrReplaceTempView("claims")
df_providers.createOrReplaceTempView("providers")
df_diagnosis.createOrReplaceTempView("diagnosis")
df_cost.createOrReplaceTempView("cost")

print("Bronze tables loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2 — View 1: Claims by Specialty
# MAGIC *How many claims does each medical specialty generate, and what do they cost on average?*
# MAGIC This view is a key tile on the Dashboard — it shows which specialties drive the most volume and cost.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Drop and recreate to ensure this is always fresh
# MAGIC CREATE OR REPLACE VIEW default.vw_claims_by_specialty AS
# MAGIC SELECT
# MAGIC   p.specialty,
# MAGIC   COUNT(c.claim_id)                                            AS total_claims,
# MAGIC   ROUND(AVG(c.billed_amount), 2)                              AS avg_billed,
# MAGIC   ROUND(SUM(c.billed_amount), 2)                              AS total_billed,
# MAGIC   COUNT(CASE WHEN c.billed_amount IS NULL THEN 1 END)         AS missing_amount_count
# MAGIC FROM claims c
# MAGIC LEFT JOIN providers p ON c.provider_id = p.provider_id
# MAGIC GROUP BY p.specialty
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify View 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.vw_claims_by_specialty;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3 — View 2: Claims by Region
# MAGIC *Which geographic regions generate the most claims, and where is the biggest gap between billed vs expected costs?*

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW default.vw_claims_by_region AS
# MAGIC SELECT
# MAGIC   co.region,
# MAGIC   COUNT(c.claim_id)                                      AS total_claims,
# MAGIC   ROUND(AVG(c.billed_amount), 2)                        AS avg_billed,
# MAGIC   ROUND(AVG(co.expected_cost), 2)                       AS avg_expected,
# MAGIC   ROUND(SUM(c.billed_amount), 2)                        AS total_billed,
# MAGIC   -- gap_pct: how much more (%) is being billed vs the benchmark
# MAGIC   ROUND(
# MAGIC     100.0 * (AVG(c.billed_amount) - AVG(co.expected_cost))
# MAGIC     / NULLIF(AVG(co.expected_cost), 0),
# MAGIC   1)                                                     AS gap_pct
# MAGIC FROM claims c
# MAGIC LEFT JOIN cost co ON c.proc_id = co.procedure_code
# MAGIC GROUP BY co.region
# MAGIC ORDER BY total_claims DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify View 2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.vw_claims_by_region;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4 — View 3: High-Cost Claims
# MAGIC *Claims where `billed_amount` is above the 90th percentile — these are the highest-risk candidates for denial.*

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW default.vw_high_cost_claims AS
# MAGIC SELECT
# MAGIC   c.claim_id,
# MAGIC   c.patient_id,
# MAGIC   p.doctor_name,
# MAGIC   p.specialty,
# MAGIC   c.diagnosis_code,
# MAGIC   d.category            AS diagnosis_category,
# MAGIC   d.severity            AS diagnosis_severity,
# MAGIC   c.proc_id,
# MAGIC   c.billed_amount,
# MAGIC   co.expected_cost,
# MAGIC   ROUND(c.billed_amount / NULLIF(co.expected_cost, 0), 2) AS billing_ratio,
# MAGIC   c.date
# MAGIC FROM claims c
# MAGIC LEFT JOIN providers p  ON c.provider_id      = p.provider_id
# MAGIC LEFT JOIN diagnosis d  ON c.diag_code         = d.diagnosis_code
# MAGIC LEFT JOIN cost co      ON c.proc_id            = co.procedure_code
# MAGIC WHERE c.billed_amount >= (
# MAGIC   SELECT PERCENTILE_APPROX(billed_amount, 0.90) FROM claims WHERE billed_amount IS NOT NULL
# MAGIC )
# MAGIC ORDER BY c.billed_amount DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify View 3

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.vw_high_cost_claims LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 5 — View 4: Missing Billed Amount (Imputation Transparency View)
# MAGIC *Since 34.3% of billed_amount values are NULL, we create a dedicated view to track these claims.*
# MAGIC *This view will be used by the Silver layer to identify which rows need imputation.*
# MAGIC
# MAGIC **Confirmed Silver Strategy:** Keep rows + impute with per-procedure median + flag `is_billed_missing = True`

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW default.vw_missing_billed_claims AS
# MAGIC SELECT
# MAGIC   c.claim_id,
# MAGIC   c.patient_id,
# MAGIC   c.provider_id,
# MAGIC   c.proc_id,
# MAGIC   c.diagnosis_code,
# MAGIC   c.billed_amount,                         -- NULL for these rows
# MAGIC   co.expected_cost,                         -- Will be used as the imputation benchmark
# MAGIC   TRUE                 AS is_billed_missing  -- Transparency flag for downstream use
# MAGIC FROM claims c
# MAGIC LEFT JOIN cost co ON c.proc_id = co.procedure_code
# MAGIC WHERE c.billed_amount IS NULL;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How many missing claims have a matching expected_cost we can impute from?
# MAGIC SELECT
# MAGIC   COUNT(*)                                                        AS total_missing_billed,
# MAGIC   COUNT(CASE WHEN expected_cost IS NOT NULL THEN 1 END)           AS can_impute_from_cost_table,
# MAGIC   COUNT(CASE WHEN expected_cost IS NULL     THEN 1 END)           AS no_imputation_possible
# MAGIC FROM default.vw_missing_billed_claims;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 6 — All Views Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Confirm all 4 views exist in the default database
# MAGIC SHOW VIEWS IN default;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Views Created
# MAGIC | View Name | Purpose | Used by |
# MAGIC |-----------|---------|---------|
# MAGIC | `default.vw_claims_by_specialty` | Claim volume + cost by specialty | Dashboard tile 1 |
# MAGIC | `default.vw_claims_by_region` | Claim volume + billing gap by region | Dashboard tile 2 |
# MAGIC | `default.vw_high_cost_claims` | 90th percentile high-risk claims | Dashboard tile 3 |
# MAGIC | `default.vw_missing_billed_claims` | 34.3% null `billed_amount` rows + imputation readiness | Silver layer (Week 3) |
# MAGIC
# MAGIC > **Next step:** Open `06_dashboard_bronze` to build the Databricks native dashboard using these views.
# MAGIC >
# MAGIC > **Week 3 Silver Imputation Plan:**
# MAGIC > - Rows where `billed_amount IS NULL` will be kept
# MAGIC > - Imputed using **median `billed_amount` per `proc_id`** from non-null claims
# MAGIC > - A new column `is_billed_missing = True` will be added to flag every imputed row
# MAGIC > - The original Bronze table remains **untouched** (full audit trail preserved)

