# Database Schema Documentation

## Overview
The database design follows academic best practices with complete audit trail and GDPR compliance.

## Schema Structure

### 1. **academic_audit** Schema
Tracks all academic compliance and data access.

| Table | Purpose | Key Fields |
|-------|---------|------------|
| compliance_metadata | Ethics compliance tracking | ethics_ref, student_id, gdpr_compliance |
| access_log | Data access audit trail | action, table_accessed, timestamp |
| data_provenance | Data lineage tracking | original_source, anonymization_method |
| model_registry | ML model versioning | model_type, performance_metrics |

### 2. **casino_data** Schema  
Contains anonymized casino operational data.

| Table | Purpose | Key Fields |
|-------|---------|------------|
| customer_demographics | Anonymized customer profiles | customer_id (CUST_XXXXXX), age_range |
| player_sessions | Gaming activity with temporal noise | session_start (±30min), total_bet |
| daily_aggregates | Pre-computed daily statistics | date, total_sessions |
| customer_features | ML feature store | loss_chasing_score, bet_volatility |
| promotion_history | Campaign response tracking | promotion_type, response |
| casino_machines | Machine location mapping | machine_zone, floor_location |

### 3. **Views and Functions**

| Object | Type | Purpose |
|--------|------|---------|
| v_customer_summary | VIEW | Aggregated customer statistics for analysis |
| calculate_risk_score() | FUNCTION | Simplified risk scoring for academic demonstration |
| log_data_access() | FUNCTION | Manual audit trail logging |
| auto_log_trigger() | TRIGGER FUNCTION | Automatic access logging (optional) |
| log_customer_access | TRIGGER | Auto-log access on customer_demographics (optional) |

### 4. **Indexes**
Performance optimized with indexes on:
- All foreign keys
- Date columns for time-based queries
- Customer ID for fast lookups

## Privacy Measures
- Customer IDs pre-anonymized (CUST_XXXXXX format)
- Age generalized to ranges (18-24, 25-34, etc.)
- Location generalized to regions
- Temporal noise (±30 minutes) on all timestamps
- No PII stored in any table

## Performance Optimizations
- Indexes on all foreign keys
- Daily aggregation tables for reporting
- Partitioning ready (by date) for large datasets

## Academic Notes
- All access is logged for reproducibility
- Risk calculations are simplified for demonstration
- Full compliance with University of Bath Ethics (Ref: 10351-12382)