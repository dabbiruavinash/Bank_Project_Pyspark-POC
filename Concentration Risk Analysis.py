def analyze_concentration_risk(accounts_df, loans_df, customers_df):
    """Analyze exposure to specific customer segments"""
    from pyspark.sql import functions as F
    
    # Customer segment concentration
    segment_concentration = customers_df.join(accounts_df, "customer_id") \
        .groupBy("customer_type", "state") \
        .agg(
            F.countDistinct("customer_id").alias("customer_count"),
            F.sum("balance").alias("total_exposure"),
            F.avg("balance").alias("avg_exposure")
        ) \
        .withColumn("exposure_percentage", 
                   F.col("total_exposure") / F.sum("total_exposure").over(Window.partitionBy())) \
        .withColumn("concentration_risk",
                   F.when(F.col("exposure_percentage") > 0.1, "HIGH")
                    .when(F.col("exposure_percentage") > 0.05, "MEDIUM")
                    .otherwise("LOW"))
    
    return segment_concentration