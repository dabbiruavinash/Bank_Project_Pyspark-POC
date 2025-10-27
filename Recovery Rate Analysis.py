def analyze_recovery_rates(loans_df, loan_payments_df, accounts_df):
    """Analyze loan recovery rates and patterns for defaulted loans"""
    from pyspark.sql import functions as F
    
    # Defaulted loans analysis
    defaulted_loans = loans_df.filter(F.col("status") == "DEFAULTED") \
        .join(loan_payments_df, "loan_id") \
        .groupBy("loan_id") \
        .agg(
            F.sum("amount_due").alias("total_amount_due"),
            F.sum("amount_paid").alias("total_amount_paid"),
            F.max("payment_date").alias("last_payment_date"),
            F.count("*").alias("total_payments")
        ) \
        .withColumn("recovery_rate", 
                   F.col("total_amount_paid") / F.col("total_amount_due")) \
        .withColumn("recovery_time_months",
                   F.months_between(F.col("last_payment_date"), 
                                   F.first("start_date").over(Window.partitionBy("loan_id"))))
    
    recovery_analysis = defaulted_loans.join(loans_df, "loan_id") \
        .groupBy("loan_type") \
        .agg(
            F.count("*").alias("defaulted_loans_count"),
            F.avg("recovery_rate").alias("avg_recovery_rate"),
            F.percentile_approx("recovery_rate", 0.5).alias("median_recovery_rate"),
            F.avg("recovery_time_months").alias("avg_recovery_time"),
            F.sum("total_amount_due").alias("total_defaulted_amount"),
            F.sum("total_amount_paid").alias("total_recovered_amount")
        ) \
        .withColumn("recovery_efficiency",
                   F.col("avg_recovery_rate") / F.col("avg_recovery_time")) \
        .withColumn("recovery_performance",
                   F.when(F.col("avg_recovery_rate") > 0.7, "EXCELLENT")
                    .when(F.col("avg_recovery_rate") > 0.5, "GOOD")
                    .when(F.col("avg_recovery_rate") > 0.3, "FAIR")
                    .otherwise("POOR"))
    
    return recovery_analysis