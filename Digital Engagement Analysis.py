def analyze_digital_engagement(transactions_df, accounts_df, customers_df):
    """Analyze digital banking engagement and channel preferences"""
    from pyspark.sql import functions as F
    
    # Transaction channel analysis (simplified - would normally have channel data)
    digital_engagement = transactions_df.join(accounts_df, "account_id") \
        .join(customers_df, "customer_id") \
        .withColumn("transaction_hour", F.hour("transaction_date")) \
        .withColumn("is_off_hours", 
                   F.when((F.col("transaction_hour") < 9) | (F.col("transaction_hour") > 17), 1)
                    .otherwise(0)) \
        .withColumn("is_weekend", 
                   F.when(F.dayofweek("transaction_date").isin(1, 7), 1)
                    .otherwise(0)) \
        .withColumn("likely_digital_channel",
                   F.when((F.col("is_off_hours") == 1) | (F.col("is_weekend") == 1), "DIGITAL")
                    .otherwise("BRANCH")) \
        .groupBy("customer_id") \
        .agg(
            F.count("*").alias("total_transactions"),
            F.sum("is_off_hours").alias("off_hours_transactions"),
            F.sum("is_weekend").alias("weekend_transactions"),
            F.avg("amount").alias("avg_transaction_size")
        ) \
        .withColumn("digital_engagement_ratio",
                   (F.col("off_hours_transactions") + F.col("weekend_transactions")) / F.col("total_transactions")) \
        .withColumn("digital_adoption_level",
                   F.when(F.col("digital_engagement_ratio") > 0.8, "DIGITAL_FIRST")
                    .when(F.col("digital_engagement_ratio") > 0.5, "DIGITAL_PREFERRED")
                    .when(F.col("digital_engagement_ratio") > 0.2, "MIXED_USAGE")
                    .otherwise("BRANCH_PREFERRED")) \
        .withColumn("migration_opportunity",
                   F.when(F.col("digital_adoption_level") == "BRANCH_PREFERRED", "DIGITAL_ONBOARDING")
                    .when(F.col("digital_adoption_level") == "MIXED_USAGE", "DIGITAL_ENHANCEMENT")
                    .when(F.col("digital_adoption_level") == "DIGITAL_PREFERRED", "ADVANCED_FEATURES")
                    .otherwise("MAINTAIN_SUPPORT"))
    
    return digital_engagement