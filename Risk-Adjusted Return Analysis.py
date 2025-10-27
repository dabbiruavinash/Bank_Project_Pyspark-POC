def analyze_risk_adjusted_returns(loans_df, accounts_df, credit_cards_df, loan_payments_df):
    """Calculate returns adjusted for risk exposure across products"""
    from pyspark.sql import functions as F
    
    # Revenue calculation
    loan_revenue = loans_df.filter(F.col("status") == "ACTIVE") \
        .withColumn("estimated_annual_revenue", 
                   F.col("loan_amount") * F.col("interest_rate")) \
        .groupBy("loan_type") \
        .agg(F.sum("estimated_annual_revenue").alias("total_annual_revenue"))
    
    # Risk calculation (simplified)
    loan_risk = loans_df.join(loan_payments_df, "loan_id") \
        .groupBy("loan_id", "loan_type") \
        .agg(
            F.sum(F.when(F.col("status") == "OVERDUE", 1).otherwise(0)).alias("overdue_count"),
            F.avg(F.col("interest_rate")).alias("interest_rate")
        ) \
        .groupBy("loan_type") \
        .agg(
            F.avg("overdue_count").alias("avg_overdue_frequency"),
            F.avg("interest_rate").alias("avg_interest_rate")
        )
    
    # Risk-adjusted return calculation
    risk_adjusted_returns = loan_revenue.join(loan_risk, "loan_type") \
        .withColumn("risk_adjustment_factor",
                   F.when(F.col("avg_overdue_frequency") > 3, 0.5)
                    .when(F.col("avg_overdue_frequency") > 2, 0.7)
                    .when(F.col("avg_overdue_frequency") > 1, 0.8)
                    .otherwise(1.0)) \
        .withColumn("risk_adjusted_revenue",
                   F.col("total_annual_revenue") * F.col("risk_adjustment_factor")) \
        .withColumn("return_per_risk_unit",
                   F.col("risk_adjusted_revenue") / F.col("avg_overdue_frequency")) \
        .withColumn("performance_ranking",
                   F.when(F.col("return_per_risk_unit") > 100000, "TOP_PERFORMER")
                    .when(F.col("return_per_risk_unit") > 50000, "STRONG_PERFORMER")
                    .when(F.col("return_per_risk_unit") > 25000, "AVERAGE_PERFORMER")
                    .otherwise("UNDER_PERFORMER"))
    
    return risk_adjusted_returns