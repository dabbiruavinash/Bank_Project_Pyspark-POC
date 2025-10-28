def analyze_demographic_targeting(customers_df, accounts_df, transactions_df, loans_df):
    """Analyze customer demographics for targeted marketing and product development"""
    from pyspark.sql import functions as F
    
    # Age group analysis
    demographic_analysis = customers_df.join(accounts_df, "customer_id") \
        .withColumn("age", 
                   F.floor(F.datediff(F.current_date(), "date_of_birth") / 365.25)) \
        .withColumn("age_group",
                   F.when(F.col("age") < 25, "18-24")
                    .when(F.col("age") < 35, "25-34")
                    .when(F.col("age") < 45, "35-44")
                    .when(F.col("age") < 55, "45-54")
                    .when(F.col("age") < 65, "55-64")
                    .otherwise("65+")) \
        .groupBy("age_group", "customer_type") \
        .agg(
            F.countDistinct("customer_id").alias("customer_count"),
            F.avg("balance").alias("avg_balance"),
            F.sum("balance").alias("total_balance"),
            F.percentile_approx("balance", 0.5).alias("median_balance")
        ) \
        .join(loans_df.join(customers_df, "customer_id")
              .groupBy("age_group", "customer_type")
              .agg(F.count("*").alias("loan_count")), ["age_group", "customer_type"], "left") \
        .fillna(0) \
        .withColumn("loan_penetration_rate",
                   F.col("loan_count") / F.col("customer_count")) \
        .withColumn("targeting_opportunity",
                   F.when((F.col("avg_balance") > 50000) & (F.col("loan_penetration_rate") < 0.3), "WEALTH_MANAGEMENT")
                    .when((F.col("age_group").isin("25-34", "35-44")) & (F.col("loan_penetration_rate") < 0.4), "HOME_LOAN_TARGET")
                    .when((F.col("age_group") == "18-24") & (F.col("customer_count") > 1000), "STUDENT_BANKING")
                    .when((F.col("age_group") == "65+") & (F.col("avg_balance") > 75000), "RETIREMENT_PLANNING")
                    .otherwise("GENERAL_BANKING"))
    
    return demographic_analysis