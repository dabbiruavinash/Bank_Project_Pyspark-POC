def provide_strategic_planning_support(customers_df, accounts_df, loans_df, transactions_df, branches_df):
    """Provide comprehensive analytics for strategic planning"""
    from pyspark.sql import functions as F
    
    # Market position analysis
    market_position = customers_df.join(accounts_df, "customer_id") \
        .groupBy("state") \
        .agg(
            F.countDistinct("customer_id").alias("customer_base"),
            F.sum("balance").alias("total_deposits"),
            F.avg("balance").alias("avg_balance")
        ) \
        .withColumn("market_share_ranking", 
                   F.percent_rank().over(Window.orderBy(F.desc("total_deposits"))))
    
    # Product portfolio analysis
    product_portfolio = accounts_df.groupBy("account_type") \
        .agg(
            F.count("*").alias("account_count"),
            F.sum("balance").alias("total_balance"),
            F.avg(F.datediff(F.current_date(), "open_date")).alias("avg_account_age")
        ) \
        .withColumn("portfolio_concentration",
                   F.col("total_balance") / F.sum("total_balance").over(Window.partitionBy()))
    
    # Growth opportunities
    growth_opportunities = customers_df.join(accounts_df, "customer_id") \
        .withColumn("products_per_customer", 
                   F.count("*").over(Window.partitionBy("customer_id"))) \
        .filter(F.col("products_per_customer") == 1) \
        .groupBy("state", "customer_type") \
        .agg(F.count("*").alias("cross_sell_opportunities"))
    
    strategic_insights = market_position \
        .join(product_portfolio, market_position.state == product_portfolio.account_type, "left") \
        .join(growth_opportunities, market_position.state == growth_opportunities.state, "left") \
        .withColumn("strategic_recommendation",
                   F.when((F.col("market_share_ranking") < 0.3) & (F.col("cross_sell_opportunities") > 1000), "AGGRESSIVE_EXPANSION")
                    .when((F.col("market_share_ranking") < 0.6) & (F.col("portfolio_concentration") > 0.3), "DIVERSIFICATION_FOCUS")
                    .when(F.col("cross_sell_opportunities") > 5000, "CROSS_SELL_INTENSIVE")
                    .otherwise("MAINTAIN_STRATEGY"))
    
    return strategic_insights