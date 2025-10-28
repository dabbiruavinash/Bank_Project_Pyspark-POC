def segment_customer_behavior(customers_df, accounts_df, transactions_df, credit_cards_df):
    """Advanced customer segmentation based on comprehensive behavior analysis"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    
    # Financial behavior metrics
    financial_behavior = transactions_df.join(accounts_df, "account_id") \
        .groupBy("customer_id") \
        .agg(
            F.count("*").alias("transaction_frequency"),
            F.avg("amount").alias("avg_transaction_size"),
            F.stddev("amount").alias("transaction_size_volatility"),
            F.sum("amount").alias("total_transaction_volume"),
            F.countDistinct("transaction_type").alias("product_usage_variety")
        )
    
    # Balance patterns
    balance_patterns = accounts_df.groupBy("customer_id") \
        .agg(
            F.sum("balance").alias("total_balance"),
            F.avg("balance").alias("avg_balance"),
            F.stddev("balance").alias("balance_volatility"),
            F.count("*").alias("number_of_accounts")
        )
    
    # Credit usage
    credit_usage = credit_cards_df.filter(F.col("status") == "ACTIVE") \
        .groupBy("customer_id") \
        .agg(
            F.sum("credit_limit").alias("total_credit_limit"),
            F.sum("current_balance").alias("total_credit_usage"),
            F.avg(F.col("current_balance") / F.col("credit_limit")).alias("avg_utilization_rate")
        )
    
    # Customer segments
    customer_segments = customers_df \
        .join(financial_behavior, "customer_id", "left") \
        .join(balance_patterns, "customer_id", "left") \
        .join(credit_usage, "customer_id", "left") \
        .fillna(0) \
        .withColumn("segment_category",
                   F.when((F.col("total_balance") > 100000) & (F.col("transaction_frequency") > 100), "WEALTHY_ACTIVE")
                    .when((F.col("total_balance") > 50000) & (F.col("avg_utilization_rate") < 0.3), "SAVERS")
                    .when((F.col("avg_utilization_rate") > 0.7) & (F.col("transaction_frequency") > 50), "CREDIT_HEAVY")
                    .when((F.col("transaction_frequency") < 10) & (F.col("total_balance") < 1000), "DORMANT")
                    .when(F.col("product_usage_variety") > 3, "DIVERSIFIED")
                    .otherwise("MAINSTREAM")) \
        .withColumn("segment_characteristics",
                   F.when(F.col("segment_category") == "WEALTHY_ACTIVE", "High balance, frequent transactions")
                    .when(F.col("segment_category") == "SAVERS", "Good savings, low credit usage")
                    .when(F.col("segment_category") == "CREDIT_HEAVY", "High credit utilization, active spending")
                    .when(F.col("segment_category") == "DORMANT", "Low activity, minimal balances")
                    .when(F.col("segment_category") == "DIVERSIFIED", "Uses multiple products actively")
                    .otherwise("Standard banking behavior"))
    
    return customer_segments