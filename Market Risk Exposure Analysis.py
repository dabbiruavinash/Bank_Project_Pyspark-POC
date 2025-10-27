def analyze_market_risk_exposure(loans_df, accounts_df, transactions_df):
    """Analyze sensitivity to market conditions and interest rate changes"""
    from pyspark.sql import functions as F
    
    # Interest rate gap analysis
    rate_sensitivity = loans_df.filter(F.col("status") == "ACTIVE") \
        .withColumn("rate_change_1pct_impact", 
                   F.col("loan_amount") * F.col("interest_rate") * 0.01 * F.col("term_months") / 12) \
        .withColumn("duration_category",
                   F.when(F.col("term_months") <= 12, "SHORT_TERM")
                    .when(F.col("term_months") <= 60, "MEDIUM_TERM")
                    .otherwise("LONG_TERM")) \
        .groupBy("duration_category", "loan_type") \
        .agg(
            F.count("*").alias("loan_count"),
            F.sum("loan_amount").alias("total_exposure"),
            F.sum("rate_change_1pct_impact").alias("total_impact_1pct"),
            F.avg("interest_rate").alias("avg_interest_rate")
        ) \
        .withColumn("sensitivity_ratio", 
                   F.col("total_impact_1pct") / F.col("total_exposure")) \
        .withColumn("market_risk_level",
                   F.when(F.col("sensitivity_ratio") > 0.15, "HIGH_EXPOSURE")
                    .when(F.col("sensitivity_ratio") > 0.10, "MEDIUM_EXPOSURE")
                    .when(F.col("sensitivity_ratio") > 0.05, "LOW_EXPOSURE")
                    .otherwise("MINIMAL_EXPOSURE"))
    
    # Deposit beta analysis (sensitivity of deposits to rate changes)
    deposit_sensitivity = accounts_df.filter(F.col("status") == "ACTIVE") \
        .withColumn("deposit_category",
                   F.when(F.col("account_type") == "SAVINGS", "RATE_SENSITIVE")
                    .when(F.col("account_type") == "CHECKING", "STABLE")
                    .otherwise("OTHER")) \
        .groupBy("deposit_category") \
        .agg(
            F.sum("balance").alias("total_deposits"),
            F.count("*").alias("account_count"),
            F.avg("balance").alias("avg_balance")
        ) \
        .withColumn("deposit_stability_score",
                   F.when(F.col("deposit_category") == "STABLE", 0.9)
                    .when(F.col("deposit_category") == "RATE_SENSITIVE", 0.6)
                    .otherwise(0.7))
    
    market_risk_analysis = rate_sensitivity.crossJoin(
        deposit_sensitivity.agg(F.avg("deposit_stability_score").alias("avg_deposit_stability"))
    )
    
    return market_risk_analysis