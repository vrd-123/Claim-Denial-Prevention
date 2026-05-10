# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Data Profiling (Databricks)
# MAGIC In Databricks, we don't need to generate `.txt` reports because we can use native interactive profiling tools.
# MAGIC
# MAGIC This notebook demonstrates how to profile the tables we just ingested into the Bronze layer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Interactive Built-in Profiling
# MAGIC Databricks has a powerful built-in command `dbutils.data.summarize()` that automatically calculates min, max, mean, missing values, and shows data distributions for every column.

# COMMAND ----------

# Load the bronze tables from the Hive Metastore
df_claims = spark.table("default.bronze_claims_raw")
df_providers = spark.table("default.bronze_provider_raw")
df_diagnosis = spark.table("default.bronze_diagnosis_raw")
df_cost = spark.table("default.bronze_cost_raw")

# COMMAND ----------

# Generate an interactive profiling report for Claims
df = spark.read.csv("/Volumes/workspace/default/myvol/raw/claims/", header=True, inferSchema=True)
dbutils.data.summarize(df)

# COMMAND ----------

# Generate an interactive profiling report for Providers
dbutils.data.summarize(df_providers)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Domain-Specific SQL Checks
# MAGIC In your local script, you checked for negative billed amounts or unusually high amounts. 
# MAGIC In Databricks, the easiest way to do these specific checks is using SQL!

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check for negative billed amounts
# MAGIC SELECT count(*) as negative_billed_amounts
# MAGIC FROM default.bronze_claims_raw
# MAGIC WHERE billed_amount < 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check for abnormally high billed amounts (> 100,000)
# MAGIC SELECT count(*) as high_billed_amounts
# MAGIC FROM default.bronze_claims_raw
# MAGIC WHERE billed_amount > 100000;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check for duplicate claim IDs (Key uniqueness check)
# MAGIC SELECT claim_id, count(*) as duplicate_count
# MAGIC FROM default.bronze_claims_raw
# MAGIC GROUP BY claim_id
# MAGIC HAVING count(*) > 1;
