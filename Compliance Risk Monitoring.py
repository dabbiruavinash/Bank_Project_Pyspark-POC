def monitor_compliance_risk(transactions_df, accounts_df, customers_df, credit_cards_df):
    """Monitor regulatory compliance risks across operations"""
    from pyspark.sql import functions as F
    
    # AML/CTF compliance monitoring
    aml_monitoring = transactions_df.join(accounts_df, "account_id") \
        .join(customers_df, "customer_id") \
        .withColumn("suspicious_indicator",
                   F.when(F.col("amount") > 10000, "LARGE_TRANSACTION")
                    .when((F.col("amount") > 5000) & 
                          (F.col("transaction_type") == "CASH_DEPOSIT"), "SUSPICIOUS_CASH")
                    .when((F.col("amount") > 8000) & 
                          (F.col("description").like("%international%")), "CROSS_BORDER")
                    .otherwise("NORMAL")) \
        .filter(F.col("suspicious_indicator") != "NORMAL") \
        .groupBy("customer_id", "suspicious_indicator") \
        .agg(
            F.count("*").alias("alert_count"),
            F.sum("amount").alias("total_suspicious_amount"),
            F.collect_set("transaction_date").alias("suspicious_dates")
        )
    
    # KYC compliance monitoring
    kyc_compliance = customers_df \
        .withColumn("kyc_risk_indicator",
                   F.when(F.col("date_of_birth").isNull(), "MISSING_DOB")
                    .when(F.col("address").isNull(), "MISSING_ADDRESS")
                    .when(F.col("phone").isNull(), "MISSING_PHONE")
                    .when(F.col("email").isNull(), "MISSING_EMAIL")
                    .otherwise("COMPLETE")) \
        .filter(F.col("kyc_risk_indicator") != "COMPLETE") \
        .groupBy("kyc_risk_indicator") \
        .agg(F.count("*").alias("non_compliant_customers"))
    
    # Credit compliance monitoring
    credit_compliance = credit_cards_df.join(customers_df, "customer_id") \
        .withColumn("compliance_issue",
                   F.when(F.col("current_balance") > F.col("credit_limit") * 0.9, "HIGH_UTILIZATION")
                    .when(F.col("credit_limit") > 50000, "HIGH_LIMIT")
                    .otherwise("COMPLIANT")) \
        .filter(F.col("compliance_issue") != "COMPLIANT") \
        .groupBy("compliance_issue") \
        .agg(
            F.count("*").alias("non_compliant_cards"),
            F.avg("current_balance").alias("avg_balance"),
            F.avg("credit_limit").alias("avg_limit")
        )
    
    compliance_summary = aml_monitoring.agg(F.countDistinct("customer_id").alias("aml_alert_customers")) \
        .crossJoin(kyc_compliance.agg(F.sum("non_compliant_customers").alias("kyc_non_compliant"))) \
        .crossJoin(credit_compliance.agg(F.sum("non_compliant_cards").alias("credit_non_compliant"))) \
        .withColumn("overall_compliance_risk",
                   F.when((F.col("aml_alert_customers") > 100) | 
                          (F.col("kyc_non_compliant") > 50) |
                          (F.col("credit_non_compliant") > 200), "HIGH_RISK")
                    .when((F.col("aml_alert_customers") > 50) | 
                          (F.col("kyc_non_compliant") > 20) |
                          (F.col("credit_non_compliant") > 100), "MEDIUM_RISK")
                    .otherwise("LOW_RISK"))
    
    return compliance_summary