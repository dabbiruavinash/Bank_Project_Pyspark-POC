def assess_counterparty_risk(customers_df, accounts_df, loans_df, account_beneficiaries_df):
    """Analyze risk from business relationships and interconnected entities"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    
    # Business relationship analysis
    business_relationships = account_beneficiaries_df \
        .filter(F.col("relationship") == "BUSINESS_PARTNER") \
        .groupBy("customer_id") \
        .agg(
            F.count("*").alias("business_partner_count"),
            F.sum("allocation_percentage").alias("total_business_allocation")
        )
    
    # Large exposure analysis
    large_exposures = loans_df.join(accounts_df, "account_id") \
        .groupBy("customer_id") \
        .agg(F.sum("loan_amount").alias("total_borrowing")) \
        .withColumn("exposure_percentile", 
                   F.percent_rank().over(Window.orderBy("total_borrowing"))) \
        .withColumn("is_large_exposure", F.col("exposure_percentile") > 0.95)
    
    # Interconnected entity risk
    interconnected_risk = customers_df \
        .join(business_relationships, "customer_id", "left") \
        .join(large_exposures, "customer_id", "left") \
        .fillna(0) \
        .withColumn("counterparty_risk_score",
                   (F.when(F.col("business_partner_count") > 5, 0.3).otherwise(0) + 
                    F.when(F.col("total_business_allocation") > 50, 0.3).otherwise(0) + 
                    F.when(F.col("is_large_exposure") == True, 0.4).otherwise(0))) \
        .withColumn("risk_category",
                   F.when(F.col("counterparty_risk_score") > 0.6, "HIGH_RISK")
                    .when(F.col("counterparty_risk_score") > 0.3, "MEDIUM_RISK")
                    .otherwise("LOW_RISK")) \
        .filter(F.col("counterparty_risk_score") > 0)
    
    return interconnected_risk