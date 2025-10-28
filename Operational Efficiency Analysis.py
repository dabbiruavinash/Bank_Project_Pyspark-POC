def analyze_operational_efficiency(employees_df, branches_df, transactions_df, accounts_df):
    """Analyze operational efficiency across banking processes"""
    from pyspark.sql import functions as F
    
    # Branch efficiency metrics
    branch_efficiency = branches_df \
        .join(accounts_df.groupBy("branch_id")
              .agg(F.count("*").alias("accounts_managed"),
                   F.sum("balance").alias("deposits_managed")), "branch_id", "left") \
        .join(transactions_df.join(accounts_df, "account_id")
              .groupBy("branch_id")
              .agg(F.count("*").alias("transactions_processed")), "branch_id", "left") \
        .join(employees_df.groupBy("branch_id")
              .agg(F.count("*").alias("staff_count"),
                   F.avg("salary").alias("avg_salary")), "branch_id", "left") \
        .fillna(0) \
        .withColumn("accounts_per_staff", 
                   F.col("accounts_managed") / F.col("staff_count")) \
        .withColumn("transactions_per_staff",
                   F.col("transactions_processed") / F.col("staff_count")) \
        .withColumn("deposits_per_staff",
                   F.col("deposits_managed") / F.col("staff_count")) \
        .withColumn("cost_per_transaction",
                   F.col("avg_salary") / F.col("transactions_per_staff")) \
        .withColumn("efficiency_score",
                   (F.col("accounts_per_staff") * 0.3 + 
                    F.col("transactions_per_staff") * 0.4 + 
                    F.col("deposits_per_staff") * 0.3)) \
        .withColumn("efficiency_quartile",
                   F.ntile(4).over(Window.orderBy("efficiency_score"))) \
        .withColumn("improvement_recommendation",
                   F.when(F.col("efficiency_quartile") == 1, "MAJOR_IMPROVEMENT_NEEDED")
                    .when(F.col("efficiency_quartile") == 2, "MODERATE_IMPROVEMENT_NEEDED")
                    .when(F.col("efficiency_quartile") == 3, "MINOR_IMPROVEMENT_NEEDED")
                    .otherwise("MAINTAIN_PERFORMANCE"))
    
    return branch_efficiency