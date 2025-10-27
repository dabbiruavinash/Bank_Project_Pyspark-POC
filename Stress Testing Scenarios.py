def perform_stress_testing(loans_df, accounts_df, transactions_df, credit_cards_df):
    """Simulate various stress scenarios on the portfolio"""
    from pyspark.sql import functions as F
    
    # Base portfolio metrics
    base_portfolio = loans_df.filter(F.col("status") == "ACTIVE") \
        .unionByName(credit_cards_df.filter(F.col("status") == "ACTIVE")
                      .select(F.col("card_id").alias("loan_id"),
                             F.col("customer_id"),
                             F.col("current_balance").alias("loan_amount"),
                             F.lit("CREDIT_CARD").alias("loan_type"),
                             F.lit(0.18).alias("interest_rate"),
                             F.lit(36).alias("term_months"))) \
        .withColumn("base_pd",  # Probability of Default - simplified
                   F.when(F.col("loan_type") == "CREDIT_CARD", 0.03)
                    .when(F.col("loan_type") == "PERSONAL", 0.02)
                    .when(F.col("loan_type") == "AUTO", 0.015)
                    .when(F.col("loan_type") == "MORTGAGE", 0.01)
                    .otherwise(0.025))
    
    # Stress scenarios
    stress_scenarios = base_portfolio \
        .withColumn("stress_scenario_1",  # Mild recession
                   F.col("base_pd") * 1.5) \
        .withColumn("stress_scenario_2",  # Severe recession
                   F.col("base_pd") * 2.5) \
        .withColumn("stress_scenario_3",  # Financial crisis
                   F.col("base_pd") * 4.0) \
        .withColumn("expected_loss_base",
                   F.col("loan_amount") * F.col("base_pd")) \
        .withColumn("expected_loss_stress_1",
                   F.col("loan_amount") * F.col("stress_scenario_1")) \
        .withColumn("expected_loss_stress_2",
                   F.col("loan_amount") * F.col("stress_scenario_2")) \
        .withColumn("expected_loss_stress_3",
                   F.col("loan_amount") * F.col("stress_scenario_3"))
    
    # Aggregate stress test results
    stress_results = stress_scenarios.groupBy("loan_type") \
        .agg(
            F.count("*").alias("portfolio_count"),
            F.sum("loan_amount").alias("total_exposure"),
            F.sum("expected_loss_base").alias("base_expected_loss"),
            F.sum("expected_loss_stress_1").alias("stress_1_expected_loss"),
            F.sum("expected_loss_stress_2").alias("stress_2_expected_loss"),
            F.sum("expected_loss_stress_3").alias("stress_3_expected_loss")
        ) \
        .withColumn("loss_increase_scenario_1", 
                   (F.col("stress_1_expected_loss") - F.col("base_expected_loss")) / F.col("base_expected_loss")) \
        .withColumn("loss_increase_scenario_2", 
                   (F.col("stress_2_expected_loss") - F.col("base_expected_loss")) / F.col("base_expected_loss")) \
        .withColumn("loss_increase_scenario_3", 
                   (F.col("stress_3_expected_loss") - F.col("base_expected_loss")) / F.col("base_expected_loss")) \
        .withColumn("capital_adequacy_stress_1",
                   F.when(F.col("loss_increase_scenario_1") > 1.0, "INADEQUATE")
                    .when(F.col("loss_increase_scenario_1") > 0.5, "MARGINAL")
                    .otherwise("ADEQUATE")) \
        .withColumn("capital_adequacy_stress_2",
                   F.when(F.col("loss_increase_scenario_2") > 2.0, "INADEQUATE")
                    .when(F.col("loss_increase_scenario_2") > 1.0, "MARGINAL")
                    .otherwise("ADEQUATE"))
    
    return stress_results