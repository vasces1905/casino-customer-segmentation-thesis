# Casino Customer Segmentation: An AI-Powered Approach to Promotional Decision Making

> **University of Bath - MSc Computer Science (Software Engineering)**  
> **Academic Year**: 2024-2025  
> **Student**: Muhammed Yavuzhan CANLI  
> **Supervisor**: Dr. Moody Alam  
> **Ethics Approval**: 10351-12382
> **Compliance: GDPR Article 26 (Data fully anonymized and non-commercial)

## Citation
If you use this work in your research, please cite it as:
Canlı, M.Y. (2025). *Casino Customer Segmentation: An AI-Powered Approach to Promotional Decision Making*. MSc Thesis, University of Bath.

## Abstract

This project is part of an MSc Dissertation at the University of Bath focused on customer behavior segmentation and AI-based promotional strategies in physical casino environments. The system integrates real-world slot machine data, anonymized CRM data, and machine learning pipelines to generate segmentation labels and personalized promotional decisions.

Using anonymized historical data from Imperial Palace Hotel Casino (Bulgaria), this framework bridges synthetic proof-of-concept (Casino-1) with a production-ready pipeline (Casino-2).  
It applies feature engineering, unsupervised clustering (K-Means), supervised learning (Random Forest), and comparative baselines to evaluate promotional targeting strategies.  
The pipeline is reproducible through Docker and PostgreSQL, ensuring both academic rigor and practical applicability.

## Academic Compliance

- **Ethics Approval**: University of Bath Ethics Committee (Ref: 10351-12382)
- **Data Protection**: Full GDPR compliance per Article 4(5) - complete anonymization
- **Academic Integrity**: All code is original work with proper attribution
- **Reproducibility**: Documented methodology with version-controlled dependencies

## Research Contributions
- **Synthetic-to-Real Data Mapping**: Compatibility layer bridging CSV/XML prototypes with real MSSQL/PostgreSQL casino data
- **Domain-Specific Feature Engineering**: 30+ engineered features (volatility, loss chasing, bet trend ratio, session-based metrics)
- **Hybrid ML Pipeline**: K-Means clustering \citep{MacQueen1967} + Random Forest classification \citep{Breiman2001}
- **Segment Migration & Temporal Drift**: 5-period segmentation views with migration matrices
- **Baseline Comparisons**: Logistic Regression, SVM, Decision Tree, KNN, Naïve Bayes
- **GDPR-Compliant Promotion Targeting**: Ethical design (CUST_xxxx IDs, age ranges, temporal noise)
- **Dockerized Academic Architecture**: PostgreSQL + Python services reproducible with `docker-compose up`

## Key Modules
- **scripts/** – One-time data import and generation tools (e.g., demographics via Faker)
- **schema/** – PostgreSQL schema (casino_data + academic_audit)
- **src/features/** – Feature engineering (behavioural + temporal)
- **src/models/** – Segmentation (KMeans), RF training, evaluation, baselines
- **src/api/** – FastAPI service for prediction endpoints
- **src/config/** – Database + academic configuration files
- **docker/** – Dockerfiles, container setup
- **thesis_outputs/** – LaTeX-ready figures, tables, evaluation reports

## Project Structure
casino-customer-segmentation-thesis/
├── academic/ # Ethics docs, compliance materials
├── data/ # Raw and processed datasets
├── schema/ # SQL scripts for database + views
│ ├── 01_create_database.sql
│ ├── 02_academic_audit_tables.sql
│ ├── 03_casino_data_tables.sql
│ └── 04_views_and_functions.sql
├── scripts/ # Data generators & batch importers
│ ├── import_casino_batch_data.py
│ ├── generate_demographics_from_valid_ids.py
│ └── populate_missing_customers.py
├── src/ # Core ML pipeline
│ ├── api/ # REST endpoints (FastAPI)
│ ├── config/ # db_config, academic_config
│ ├── data/ # anonymizer, db_connector, compatibility
│ ├── features/ # feature_engineering, temporal_features
│ ├── models/ # segmentation, promotion_rf, rf training/eval
│ └── main_pipeline.py # Orchestrator for full pipeline
├── notebooks/ # Jupyter notebooks for exploration
├── docker/ # Docker + compose files
├── thesis_outputs/ # Figures, tables, model comparison, reports
├── README.md # Project overview (this file)
└── file_structure.txt # Auto-generated file listing

## Version History ##

### v0.1.0 (16 June 2025)
- **Initial academic project structure
- **Database schema design
- **Basic ML pipeline with synthetic data
- **API endpoints implementation

### v0.1.1 (19 Jun 2025)
- **Batch import script for Casino IT data, multi-mode pipeline support, feature bug fixes

### v0.2.0 (20 June 2025):
- **Added batch data import script (`import_casino_batch_data.py`)
- **Multi-mode pipeline support (synthetic, batch, comparison)
- **Fixed feature engineering bugs
- **Updated main_pipeline.py with real data support

### v0.2.1 (2025-06-25): 
- **PostgreSQL migration
- **schema integration, ethics config

### v0.3.0–0.3.2 (7–19 July 2025)
- Cleaned and validated 2.7M+ records
- Extracted 38,319 valid IDs
- Created demographics with Faker (GDPR compliant)
- Built 5-period segmentation views (2022–2024)
- Unified export view for temporal drift analysis

### v0.3.1 (8 July 2025):
- **Confirmed one-to-one relationship across player_id ↔ customer_id
- **Verified data year/month coverage: Jul 2021 to Mar 2024
- **Ready for behavioral feature engineering (avg_bet, volatility, etc.)
- **Real data validation complete
- **pipeline ready

### v0.3.2 (19 July 2025):
- **Finalized 5-period segmentation views: 2022-H1 to 2024-H1
- **Created unified view: kmeans_export_all_periods for temporal drift analysis
- **View validation complete (35,974 customers, 100% coverage)
- **Added get_period_export_view() SQL helper for Python dynamic integration
- **Migration & promo impact analysis tools now ready

### v0.4.0 (22-24 July 2025):
- **K-Means segmentation finalized for 2022-H1 to 2023-H2
- **Implemented DBSCAN, GMM, Hierarchical clustering for algorithm comparison
- **Temporal silhouette analysis, outlier tracking, and agreement scoring added
- **Segment evolution visualization and risk overlay integration ready
- **Prepared hybrid KMeans+DBSCAN strategy for Random Forest integration phase
- **Multi-algorithm clustering finalized (KMeans, DBSCAN, GMM, Hierarchical)
- **Agreement matrices and temporal drift analysis completed
- **Segment collapse (Casual → Regular) confirmed by KMeans logic

###  v0.4.1 (26 July 2025) – RF-Ready Release (Multi Segementation)
- **Capped outliers (€1.5M max bet), CV < 3.0 achieved
- **Clean re-segmentation for 4 periods using KMeans + DBSCAN
- **Feature normalization (log1p, winsorization, RobustScaler)
- **Training base: 27,879 customers, weighted per period
- **DBSCAN outliers included as auxiliary risk label

### v0.5.0 (5 August 2025)
- Completed Chapter 5: Results & Evaluation
- Integrated segment-based promotion distributions
- Added temporal promotion trends, confidence scores, baseline comparisons
- Extracted rule-based decision matrix from RF predictions
- Added demographic-based promotional analyses
- Bibliography and Appendix fully updated for submission

### v0.6.0 (10 August 2025) – **Final Thesis Update**
- Final pipeline freeze for dissertation submission
- Dockerized full system: `docker-compose up --build` launches PostgreSQL + Python
- Automated pipeline execution via `main_pipeline.py`
- Results exported to `results/` and thesis-ready tables
- Chapter 5 diagrams aligned with `thesis_outputs/`
- Complete LaTeX integration tested with Bath template
- Repository archived for MSc examination

### Recent Updates: v0.5.0 (5 August 2025): Full Evaluation & Thesis Documentation Finalized
- Completed Chapter 5: Results and Evaluation (Sections 5.1–5.6)
- Segment-based promotion distributions integrated (visual + LaTeX format)
- Temporal promotion trends and model confidence visualized
- Model comparison implemented (RF vs LR/SVM/DT) with engineered vs non-engineered features
- Rule-based decision matrix extracted from RF model predictions
- Country-based and age-gender promotional response analysis included
- Feature-based customer risk profiling completed using volatility, loss chasing, and behavioral signals
- Added Appendix references and improved bibliography compliance
- Finalized Casino-2 AI pipeline (PostgreSQL + SMOTE + RF) with LaTeX-compliant diagrams
- Bibliography fully matched with Literature Review and model chapters


## Docker Setup (Windows – Automated)
For convenience, three helper scripts and a quick start guide are included in the project root:

- **install-docker.ps1** – Full installation script for Docker and dependencies (PowerShell).
- **setup-docker.bat** – Manual environment setup script (Batch).
- **start-casino-environment.bat** – Quick start launcher for Casino AI containers (Batch).
- **QUICK-START.md** – Step-by-step guide.

### Usage
- **Initial setup**: Double-click `install-docker.ps1`
- **Manual setup**: Double-click `setup-docker.bat`
- **Quick start**: Double-click `start-casino-environment.bat`


All files are located in the root directory of the repository.


### Auto Version History
- 7f87912 (2025-06-22): [DOCS] Auto-update version history in README
- 9d5f528 (2025-06-22): [DOCS] Auto-update version history in README
- dfc1e55 (2025-06-22): Merge branch 'main' of https://github.com/vasces1905/casino-customer-segmentation-thesis
- 34bf68a (2025-06-22): [UPDATE] Add multi-mode pipeline support and update documentation
- e470ef7 (2025-06-16): [FEAT] Implement main ML pipeline orchestrator
- fab2bd5 (2025-06-16): [FEAT] Implement FastAPI endpoints for ML inference
- f409285 (2025-06-15): Update README.md
- 356d5ec (2025-06-15): [FEAT] Implement ML models with academic standards
- 58fd4d8 (2025-06-14): [DOCS] Add reference to compatibility demo in README
- 8bbf326 (2025-06-14): [TEST] Add compatibility layer demo script
