def analyze_geographic_expansion(branches_df, customers_df, accounts_df, transactions_df):
    """Analyze potential for geographic expansion based on market gaps"""
    from pyspark.sql import functions as F
    
    # Current geographic coverage
    current_coverage = customers_df.join(accounts_df, "customer_id") \
        .groupBy("state", "city") \
        .agg(
            F.countDistinct("customer_id").alias("current_customers"),
            F.sum("balance").alias("total_deposits"),
            F.avg("balance").alias("avg_balance_per_customer")
        ) \
        .join(branches_df.groupBy("state", "city")
              .agg(F.count("*").alias("branch_count")), ["state", "city"], "left") \
        .fillna(0)
    
    # Market potential analysis
    market_potential = current_coverage \
        .withColumn("customers_per_branch",
                   F.when(F.col("branch_count") > 0, 
                         F.col("current_customers") / F.col("branch_count"))
                    .otherwise(F.col("current_customers"))) \
        .withColumn("deposits_per_branch",
                   F.when(F.col("branch_count") > 0, 
                         F.col("total_deposits") / F.col("branch_count"))
                    .otherwise(F.col("total_deposits"))) \
        .withColumn("market_saturation",
                   F.when(F.col("customers_per_branch") > 5000, "OVERSATURATED")
                    .when(F.col("customers_per_branch") > 3000, "MATURE")
                    .when(F.col("customers_per_branch") > 1500, "GROWING")
                    .when(F.col("customers_per_branch") > 500, "EMERGING")
                    .otherwise("UNDERSERVED")) \
        .withColumn("expansion_priority",
                   F.when(F.col("market_saturation") == "UNDERSERVED", "HIGH_PRIORITY")
                    .when(F.col("market_saturation") == "EMERGING", "MEDIUM_PRIORITY")
                    .when(F.col("market_saturation") == "GROWING", "LOW_PRIORITY")
                    .otherwise("NO_EXPANSION")) \
        .withColumn("potential_new_customers",
                   F.when(F.col("market_saturation") == "UNDERSERVED", 5000)
                    .when(F.col("market_saturation") == "EMERGING", 3000)
                    .when(F.col("market_saturation") == "GROWING", 1000)
                    .otherwise(0))
    
    return market_potential