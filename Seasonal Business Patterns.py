def analyze_seasonal_patterns(transactions_df, accounts_df, loans_df):
    """Identify and analyze seasonal business patterns"""
    from pyspark.sql import functions as F
    
    seasonal_analysis = transactions_df.join(accounts_df, "account_id") \
        .withColumn("transaction_year", F.year("transaction_date")) \
        .withColumn("transaction_month", F.month("transaction_date")) \
        .withColumn("transaction_quarter", F.quarter("transaction_date")) \
        .groupBy("transaction_year", "transaction_quarter", "transaction_month", "account_type") \
        .agg(
            F.count("*").alias("transaction_count"),
            F.sum("amount").alias("transaction_volume"),
            F.avg("amount").alias("avg_transaction_size")
        ) \
        .withColumn("seasonal_index",
                   F.col("transaction_volume") / F.avg("transaction_volume").over(
                       Window.partitionBy("account_type", "transaction_month"))) \
        .withColumn("seasonal_pattern",
                   F.when(F.col("transaction_month").isin(11, 12), "HOLIDAY_PEAK")
                    .when(F.col("transaction_month").isin(1, 2), "POST_HOLIDAY_DIP")
                    .when(F.col("transaction_month").isin(4, 5), "SPRING_INCREASE")
                    .when(F.col("transaction_month").isin(7, 8), "SUMMER_SLOWDOWN")
                    .otherwise("NORMAL")) \
        .withColumn("year_over_year_growth",
                   (F.col("transaction_volume") - F.lag("transaction_volume").over(
                       Window.partitionBy("account_type", "transaction_month").orderBy("transaction_year"))) /
                   F.lag("transaction_volume").over(Window.partitionBy("account_type", "transaction_month").orderBy("transaction_year")))
    
    return seasonal_analysis