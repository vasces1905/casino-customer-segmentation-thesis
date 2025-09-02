***** IMPORTANT NOTE:As the full project archive is approximately 1.2 GB and includes compiled database files, only a reduced submission package is provided here; the complete 1.2 GB version can be accessed via the accompanying GitHub repository link:https://github.com/vasces1905/casino-customer-segmentation-thesis

**** All files were uploaded as a zip in sequential as shown below:
root folder: casino-customer-segmentation-thesis
1) 1.zip: Please open under root directly (included: 
2) src.zip: Please open secondly
3) venv folder is 400 MB was not able to compressed. /Lib folder is empty and other files were compressed in it.
4) /src/models/models/rf_generic folder is included PKL files but was not able to compressed. 



=============================================================
Casino-2 Final (Real Data + Docker + PostgreSQL) - README
=============================================================

This environment represents the production-aligned pipeline (Casino-2),
built on anonymized real casino data, PostgreSQL, and Docker.

----------------------------------------
1. Core Pipeline Scripts (src/)
----------------------------------------
- main_pipeline.py
  Orchestrator for the complete pipeline.
  Supports multiple periods (e.g., 2022-H1, 2022-H2).
  CLI Example:
    python main_pipeline.py --periods 2022-H1 2022-H2 --do-segmentation

- features/complete_feature_engineering_v3.py
  Generates engineered behavioural and temporal features.
  Writes results into casino_data.customer_features.

- models/segmentation_v2.py
  Runs K-Means clustering and saves segmentation tables.

- models/rf_clean_harmonized_training_2024.py
  Trains Random Forest classifier on harmonized features.
  Saves models with timestamps and updates LATEST.pkl.

- models/rf_direct_working_predict_2.py
  Runs batch predictions using trained RF model.
  Exports: working_predictions_{PERIOD}_*.csv

- models/rf_eval_collapse3.py
  Evaluates predictions in a 3-class format.
  Outputs:
    {PERIOD}_3cls_summary.txt
    {PERIOD}_3cls_report.csv

- models/rf_make_baseline_simple_models.py
  Trains baseline classifiers (Logistic Regression, SVM, DT, KNN, NB).
  Outputs comparative CV scores.

- models/make_final_comparison_tables.py
  Produces consolidated tables:
    rf_by_period.csv
    comparison_by_period.csv

----------------------------------------
2. Database & Config
----------------------------------------
- PostgreSQL is embedded inside the Docker environment.
- When you run `docker-compose up --build`, the database container is automatically created and initialized with:
    * casino_data schema (sessions, demographics, features)
    * academic_audit schema (access log, compliance metadata)
- No manual schema creation is required; all SQL files under /schema are executed during container startup.

Configuration:
- Database connection settings are provided via environment variables (.env file).
- An example configuration is included in `.env.example`:
PostgreSQL Configuration (Docker Container)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=casino_research
DB_USER=researcher
DB_PASSWORD=academic_password_2024

Note:
- For security reasons, the actual password is not stored in the repository. Just in case it is shared above.
- Users must copy `.env.example` to `.env` and set their own password before running the pipeline.

- schema/03_casino_data_tables.sql
  Defines casino_data schema (sessions, demographics, features).

- schema/02_academic_audit_tables.sql
  Defines audit tables for ethical compliance.

- src/data/db_connector.py
  PostgreSQL connector with audit logging.

- src/data/anonymizer.py
  GDPR-compliant anonymization: CUST_xxxx IDs, age ranges, temporal noise.

- src/data/compatibility_layer.py
  Maps synthetic (Casino-1) features to real database equivalents.

----------------------------------------
3. API Layer
----------------------------------------
- src/api/main.py
  FastAPI endpoints:
    /segment   -> Predicts customer segment
    /promotion -> Predicts promotion eligibility
    /compliance -> Returns GDPR/ethics metadata

----------------------------------------
4. Execution Flow
----------------------------------------
casino-customer-segmentation-thesis.zip
├── install-docker.ps1            ← Automated Docker installation script
├── start-casino-environment.bat  ← One-click environment startup
├── setup-docker.bat              ← Manual Docker setup script
├── EVALUATOR-SETUP.md            ← Guide for thesis evaluators
├── docker-compose.academic.yml   ← Academic Docker environment configuration
├── src/                          ← Source code (features, models, API)
├── schema/                       ← Database schema files (PostgreSQL)
└── ... (other project files)

Step 1: Initialize Docker
   -> docker-compose up --build
   (starts PostgreSQL + Python containers)

Step 2: Run main pipeline
   -> main_pipeline.py
   (feature engineering → segmentation → RF training → prediction)

Step 3: Evaluate
   -> rf_eval_collapse3.py
   (generates period-based evaluation reports)

Step 4: Compare
   -> make_final_comparison_tables.py
   (consolidated results for thesis tables)

### Usage

- **Initial setup**: Double-click `install-docker.ps1`
- **Manual setup**: Double-click `setup-docker.bat`
- **Quick start**: Double-click `start-casino-environment.bat`

----------------------------------------
5. Outputs
----------------------------------------
- working_predictions_{PERIOD}_*.csv
- {PERIOD}_3cls_summary.txt
- {PERIOD}_3cls_report.csv
- rf_by_period.csv
- comparison_by_period.csv
- LaTeX-ready tables in thesis_outputs/
