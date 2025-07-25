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

This research implements machine learning techniques for customer segmentation and promotional response prediction in physical casino environments. Using anonymized historical data from Imperial Palace Hotel Casino (Bulgaria), we develop a novel framework that bridges synthetic proof-of-concept models with production-ready systems, ensuring both academic rigor and practical applicability.

## Academic Compliance

- **Ethics Approval**: University of Bath Ethics Committee (Ref: 10351-12382)
- **Data Protection**: Full GDPR compliance per Article 4(5) - complete anonymization
- **Academic Integrity**: All code is original work with proper attribution
- **Reproducibility**: Documented methodology with version-controlled dependencies

## Research Contributions

- Novel Synthetic-to-Real Data Mapping Framework: Bridging the gap between academic prototypes and production systems
- Domain-Specific Feature Engineering: Casino-specific temporal and behavioral features
- Hybrid ML Pipeline: Combining unsupervised (K-means) and supervised (Random Forest) learning
- Academic-Compliant Architecture: Demonstrating enterprise patterns in academic context
- Segment Migration & Temporal Drift Analysis: Period-based segment change tracking and simulation-ready A/B evaluation - framework
- GDPR-Compliant Promotion Targeting: Simulated and real-segment driven promotion assignment with ethical AI logic

## Key Modules
- scripts/ – One-time data generators and transformation tools
- schema/ – PostgreSQL schema and indexing setup
- src/features/ – Feature engineering (behavioral and temporal)
- src/models/ – Machine learning training (K-Means, Random Forest)
- thesis_outputs/ – Visual outputs, tables, and validation sets

## Project Structure
casino-customer-segmentation-thesis/
├── academic/                    # Ethics docs, compliance materials
├── data/                        # Raw and processed datasets
├── schema/                      # SQL scripts for database setup
├── scripts/                     # Data generation scripts
│   ├── import_casino_batch_data.py
│   ├── generate_demographics_from_valid_ids.py
├── src/                         # ML pipeline code
│   ├── features/
│   ├── models/
│   └── main_pipeline.py
├── notebooks/                   # Jupyter analysis notebooks
├── thesis_outputs/              # Figures, charts, exports
├── docker/                      # Docker and container configs
├── README.md                    # Project overview (this file)
└── file_structure.txt           # Auto-generated file list

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

### v0.3.0 (7 July 2025):
- **Cleaned and validated game_events with 2.7M+ records
- **Extracted 38,319 valid player_id values using regex and NULL filters
- **Created customer_demographics using Faker (GDPR compliant)
- **Removed 275 malformed customer_id records from demographics

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

### v0.4.0 (22 July 2025):
- **K-Means segmentation finalized for 2022-H1 to 2023-H2
- **Implemented DBSCAN, GMM, Hierarchical clustering for algorithm comparison
- **Temporal silhouette analysis, outlier tracking, and agreement scoring added
- **Segment evolution visualization and risk overlay integration ready
- **Prepared hybrid KMeans+DBSCAN strategy for Random Forest integration phase
- Multi-algorithm clustering finalized (KMeans, DBSCAN, GMM, Hierarchical)
- Agreement matrices and temporal drift analysis completed
- Segment collapse (Casual → Regular) confirmed by KMeans logic

### Recent Updates: v0.4.1 (23 July 2025) – RF-Ready Release (Multi Segementation)
- Capped outliers (€1.5M max bet), CV < 3.0 achieved
- Clean re-segmentation for 4 periods using KMeans + DBSCAN
- Feature normalization (log1p, winsorization, RobustScaler)
- Training base: 27,879 customers, weighted per period
- DBSCAN outliers included as auxiliary risk label


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
