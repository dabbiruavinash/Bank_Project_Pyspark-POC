def create_comprehensive_reporting_framework(customers_df, accounts_df, transactions_df, loans_df, 
                                           credit_cards_df, branches_df, employees_df):
    """Create integrated reporting framework with key performance indicators"""
    from pyspark.sql import functions as F
    
    # Executive Summary Metrics
    executive_summary = customers_df.agg(
        F.countDistinct("customer_id").alias("total_customers"),
        F.countDistinct(F.when(F.col("customer_type") == "BUSINESS", "customer_id")).alias("business_customers"),
        F.countDistinct(F.when(F.col("customer_type") == "INDIVIDUAL", "customer_id")).alias("individual_customers")
    ).crossJoin(
        accounts_df.agg(
            F.count("*").alias("total_accounts"),
            F.sum("balance").alias("total_deposits"),
            F.avg("balance").alias("avg_account_balance")
        )
    ).crossJoin(
        loans_df.filter(F.col("status") == "ACTIVE").agg(
            F.count("*").alias("active_loans"),
            F.sum("loan_amount").alias("total_loan_portfolio")
        )
    ).crossJoin(
        transactions_df.agg(
            F.count("*").alias("total_transactions"),
            F.sum("amount").alias("total_transaction_volume")
        )
    )
    
    # Product Performance Dashboard
    product_dashboard = accounts_df.groupBy("account_type") \
        .agg(
            F.count("*").alias("accounts"),
            F.sum("balance").alias("deposits"),
            F.avg("balance").alias("avg_balance")
        ).unionByName(
            loans_df.filter(F.col("status") == "ACTIVE")
            .groupBy(F.col("loan_type").alias("account_type"))
            .agg(
                F.count("*").alias("accounts"),
                F.sum("loan_amount").alias("deposits"),
                F.avg("loan_amount").alias("avg_balance")
            )
        ).unionByName(
            credit_cards_df.filter(F.col("status") == "ACTIVE")
            .groupBy(F.lit("CREDIT_CARD").alias("account_type"))
            .agg(
                F.count("*").alias("accounts"),
                F.sum("current_balance").alias("deposits"),
                F.avg("current_balance").alias("avg_balance")
            )
        )
    
    # Regional Performance
    regional_performance = branches_df.join(
        accounts_df.groupBy("branch_id")
        .agg(F.count("*").alias("accounts"), F.sum("balance").alias("deposits")), 
        "branch_id", "left"
    ).join(
        employees_df.groupBy("branch_id")
        .agg(F.count("*").alias("employees"), F.avg("salary").alias("avg_salary")),
        "branch_id", "left"
    ).fillna(0) \
     .withColumn("deposits_per_employee", F.col("deposits") / F.col("employees")) \
     .withColumn("efficiency_ratio", F.col("accounts") / F.col("employees"))
    
    # Risk Overview
    risk_overview = loans_df.filter(F.col("status") == "DEFAULTED") \
        .agg(
            F.count("*").alias("defaulted_loans"),
            F.sum("loan_amount").alias("defaulted_amount")
        ).crossJoin(
            loans_df.agg(
                F.count("*").alias("total_loans"),
                F.sum("loan_amount").alias("total_loan_amount")
            )
        ).withColumn("default_rate", F.col("defaulted_loans") / F.col("total_loans")) \
         .withColumn("default_ratio", F.col("defaulted_amount") / F.col("total_loan_amount"))
    
    # Comprehensive Report
    comprehensive_report = {
        "executive_summary": executive_summary,
        "product_dashboard": product_dashboard,
        "regional_performance": regional_performance,
        "risk_overview": risk_overview
    }
    
    return comprehensive_report