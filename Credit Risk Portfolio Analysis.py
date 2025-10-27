def analyze_credit_risk_portfolio(loans_df, credit_cards_df, customers_df):
    """Comprehensive credit risk assessment"""
    from pyspark.sql import functions as F
    
    credit_risk = loans_df.filter(F.col("status") == "ACTIVE") \
        .unionByName(credit_cards_df.filter(F.col("status") == "ACTIVE")
                      .select(F.col("card_id").alias("loan_id"),
                             F.col("customer_id"),
                             F.col("current_balance").alias("loan_amount"),
                             F.lit("CREDIT_CARD").alias("loan_type"),
                             F.lit(0.18).alias("interest_rate"))) \
        .join(customers_df, "customer_id") \
        .groupBy("customer_type", "loan_type") \
        .agg(
            F.count("*").alias("credit_facilities"),
            F.sum("loan_amount").alias("total_exposure"),
            F.avg("interest_rate").alias("avg_interest_rate"),
            F.percentile_approx("loan_amount", 0.95).alias("p95_exposure")
        ) \
        .withColumn("risk_adjusted_return",
                   F.col("avg_interest_rate") * F.col("total_exposure")) \
        .withColumn("portfolio_quality",
                   F.when(F.col("avg_interest_rate") > 0.15, "HIGH_RISK")
                    .when(F.col("avg_interest_rate") > 0.10, "MEDIUM_RISK")
                    .otherwise("LOW_RISK"))
    
    return credit_risk