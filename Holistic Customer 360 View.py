def create_customer_360_view(customers_df, accounts_df, transactions_df, loans_df, 
                           credit_cards_df, account_beneficiaries_df):
    """Create comprehensive 360-degree view of each customer"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    
    # Core customer information
    customer_core = customers_df
    
    # Financial summary
    financial_summary = accounts_df.filter(F.col("status") == "ACTIVE") \
        .groupBy("customer_id") \
        .agg(
            F.count("*").alias("active_accounts"),
            F.sum("balance").alias("total_deposits"),
            F.avg("balance").alias("avg_account_balance"),
            F.countDistinct("account_type").alias("account_types")
        )
    
    # Transaction behavior
    transaction_behavior = transactions_df.join(accounts_df, "account_id") \
        .groupBy("customer_id") \
        .agg(
            F.count("*").alias("lifetime_transactions"),
            F.avg("amount").alias("avg_transaction_amount"),
            F.sum("amount").alias("lifetime_transaction_volume"),
            F.max("transaction_date").alias("last_transaction_date")
        ) \
        .withColumn("days_since_last_transaction",
                   F.datediff(F.current_date(), "last_transaction_date"))
    
    # Credit relationships
    credit_relationships = loans_df.filter(F.col("status") == "ACTIVE") \
        .unionByName(credit_cards_df.filter(F.col("status") == "ACTIVE")
                      .select(F.col("customer_id"), 
                             F.col("current_balance").alias("loan_amount"),
                             F.lit("CREDIT_CARD").alias("loan_type"))) \
        .groupBy("customer_id") \
        .agg(
            F.count("*").alias("active_credit_products"),
            F.sum("loan_amount").alias("total_credit_exposure"),
            F.countDistinct("loan_type").alias("credit_product_types")
        )
    
    # Relationship network
    relationship_network = account_beneficiaries_df.groupBy("customer_id") \
        .agg(
            F.count("*").alias("beneficiary_relationships"),
            F.countDistinct("relationship").alias("relationship_types")
        )
    
    # Comprehensive 360 view
    customer_360 = customer_core \
        .join(financial_summary, "customer_id", "left") \
        .join(transaction_behavior, "customer_id", "left") \
        .join(credit_relationships, "customer_id", "left") \
        .join(relationship_network, "customer_id", "left") \
        .fillna(0) \
        .withColumn("customer_value_tier",
                   F.when(F.col("total_deposits") > 100000, "PREMIUM")
                    .when(F.col("total_deposits") > 50000, "GOLD")
                    .when(F.col("total_deposits") > 10000, "SILVER")
                    .otherwise("STANDARD")) \
        .withColumn("relationship_strength",
                   (F.col("active_accounts") * 0.2 + 
                    F.col("lifetime_transactions") * 0.15 + 
                    F.col("active_credit_products") * 0.15 + 
                    F.col("beneficiary_relationships") * 0.1 + 
                    F.when(F.col("days_since_last_transaction") < 30, 0.4).otherwise(0.1))) \
        .withColumn("next_best_action",
                   F.when((F.col("active_credit_products") == 0) & (F.col("total_deposits") > 10000), "CREDIT_CARD_OFFER")
                    .when((F.col("account_types") == 1) & (F.col("lifetime_transactions") > 50), "ADDITIONAL_ACCOUNT")
                    .when((F.col("beneficiary_relationships") == 0) & (F.col("total_deposits") > 50000), "ESTATE_PLANNING")
                    .otherwise("RELATIONSHIP_REVIEW"))
    
    return customer_360