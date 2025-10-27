def assess_operational_risk(transactions_df, employees_df, branches_df):
    """Identify operational risk hotspots"""
    from pyspark.sql import functions as F
    
    operational_risk = transactions_df.join(accounts_df, "account_id") \
        .join(branches_df, "branch_id") \
        .withColumn("is_high_value", F.col("amount") > 5000) \
        .withColumn("is_weekend", 
                   F.when(F.dayofweek("transaction_date").isin(1, 7), 1).otherwise(0)) \
        .groupBy("branch_id") \
        .agg(
            F.count("*").alias("total_transactions"),
            F.sum(F.col("is_high_value")).alias("high_value_transactions"),
            F.sum(F.col("is_weekend")).alias("weekend_transactions"),
            F.avg("amount").alias("avg_transaction_size")
        ) \
        .withColumn("operational_risk_score",
                   (F.col("high_value_transactions") * 0.4 + 
                    F.col("weekend_transactions") * 0.3 + 
                    F.col("avg_transaction_size") * 0.3)) \
        .withColumn("risk_category",
                   F.when(F.col("operational_risk_score") > F.percentile_approx("operational_risk_score", 0.8).over(Window.partitionBy()), "HIGH")
                    .when(F.col("operational_risk_score") > F.percentile_approx("operational_risk_score", 0.5).over(Window.partitionBy()), "MEDIUM")
                    .otherwise("LOW"))
    
    return operational_risk