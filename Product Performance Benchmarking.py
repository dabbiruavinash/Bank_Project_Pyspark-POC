def benchmark_product_performance(accounts_df, loans_df, credit_cards_df, transactions_df):
    """Benchmark product performance against targets and historical data"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    
    # Account product benchmarking
    account_benchmarks = accounts_df.groupBy("account_type") \
        .agg(
            F.count("*").alias("current_accounts"),
            F.sum("balance").alias("current_balance"),
            F.avg("balance").alias("current_avg_balance")
        ) \
        .withColumn("target_accounts",  # Example targets
                   F.when(F.col("account_type") == "SAVINGS", 10000)
                    .when(F.col("account_type") == "CHECKING", 15000)
                    .when(F.col("account_type") == "BUSINESS", 5000)
                    .otherwise(2000)) \
        .withColumn("target_balance",
                   F.when(F.col("account_type") == "SAVINGS", 50000000)
                    .when(F.col("account_type") == "CHECKING", 75000000)
                    .when(F.col("account_type") == "BUSINESS", 100000000)
                    .otherwise(10000000)) \
        .withColumn("accounts_target_achievement",
                   F.col("current_accounts") / F.col("target_accounts")) \
        .withColumn("balance_target_achievement",
                   F.col("current_balance") / F.col("target_balance")) \
        .withColumn("overall_performance_score",
                   (F.col("accounts_target_achievement") * 0.4 + 
                    F.col("balance_target_achievement") * 0.6)) \
        .withColumn("performance_status",
                   F.when(F.col("overall_performance_score") > 1.1, "EXCEEDS_TARGET")
                    .when(F.col("overall_performance_score") > 0.9, "MEETS_TARGET")
                    .when(F.col("overall_performance_score") > 0.7, "APPROACHING_TARGET")
                    .otherwise("BELOW_TARGET"))
    
    return account_benchmarks