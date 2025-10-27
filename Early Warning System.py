def create_early_warning_system(loans_df, loan_payments_df, accounts_df, transactions_df):
    """Create indicators for potential problems before they occur"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    
    # Payment behavior trends
    payment_trends = loan_payments_df \
        .withColumn("payment_delay", 
                   F.datediff(F.coalesce("payment_date", F.current_date()), "due_date")) \
        .withColumn("is_delayed", F.col("payment_delay") > 0) \
        .groupBy("loan_id") \
        .agg(
            F.avg("payment_delay").alias("avg_payment_delay"),
            F.stddev("payment_delay").alias("payment_delay_volatility"),
            F.sum("is_delayed").alias("delayed_payment_count"),
            F.count("*").alias("total_payments")
        ) \
        .withColumn("delinquency_trend",
                   F.col("delayed_payment_count") / F.col("total_payments"))
    
    # Account behavior changes
    account_behavior = transactions_df.join(accounts_df, "account_id") \
        .withColumn("transaction_month", F.date_trunc("month", "transaction_date")) \
        .groupBy("account_id", "transaction_month") \
        .agg(
            F.count("*").alias("monthly_transaction_count"),
            F.sum("amount").alias("monthly_transaction_volume"),
            F.avg("amount").alias("avg_transaction_size")
        ) \
        .withColumn("prev_month_volume", 
                   F.lag("monthly_transaction_volume").over(Window.partitionBy("account_id").orderBy("transaction_month"))) \
        .withColumn("volume_change", 
                   (F.col("monthly_transaction_volume") - F.col("prev_month_volume")) / F.col("prev_month_volume")) \
        .filter(F.col("prev_month_volume").isNotNull()) \
        .groupBy("account_id") \
        .agg(
            F.avg("volume_change").alias("avg_volume_change"),
            F.stddev("volume_change").alias("volume_change_volatility"),
            F.min("volume_change").alias("worst_volume_decline")
        )
    
    # Early warning indicators
    early_warnings = loans_df.join(payment_trends, "loan_id", "left") \
        .join(accounts_df, "account_id") \
        .join(account_behavior, "account_id", "left") \
        .fillna(0) \
        .withColumn("warning_score",
                   (F.when(F.col("avg_payment_delay") > 15, 0.3).otherwise(0) + 
                    F.when(F.col("delinquency_trend") > 0.5, 0.3).otherwise(0) + 
                    F.when(F.col("worst_volume_decline") < -0.5, 0.2).otherwise(0) + 
                    F.when(F.col("volume_change_volatility") > 0.8, 0.2).otherwise(0))) \
        .withColumn("warning_level",
                   F.when(F.col("warning_score") > 0.7, "RED_ALERT")
                    .when(F.col("warning_score") > 0.5, "ORANGE_ALERT")
                    .when(F.col("warning_score") > 0.3, "YELLOW_ALERT")
                    .otherwise("GREEN")) \
        .withColumn("recommended_action",
                   F.when(F.col("warning_level") == "RED_ALERT", "IMMEDIATE_INTERVENTION")
                    .when(F.col("warning_level") == "ORANGE_ALERT", "ENHANCED_MONITORING")
                    .when(F.col("warning_level") == "YELLOW_ALERT", "STANDARD_MONITORING")
                    .otherwise("NORMAL_MONITORING"))
    
    return early_warnings