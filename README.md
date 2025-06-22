# Casino Customer Segmentation: An AI-Powered Approach to Promotional Decision Making

> **University of Bath - MSc Computer Science (Software Engineering)**  
> **Academic Year**: 2024-2025  
> **Student**: Muhammed Yavuzhan CANLI  
> **Supervisor**: Dr. Moody Alam  
> **Ethics Approval**: 10351-12382

## Citation
If you use this work in your research, please cite it as:
Canlı, M.Y. (2025). *Casino Customer Segmentation: An AI-Powered Approach to Promotional Decision Making*. MSc Thesis, University of Bath.

## Version History

### v0.2.0 (19 June 2025) - Current
- Added batch data import script (`import_casino_batch_data.py`)
- Multi-mode pipeline support (synthetic, batch, comparison)
- Fixed feature engineering bugs
- Updated main_pipeline.py with real data support

### v0.1.0 (16 June 2025)
- Initial academic project structure
- Database schema design
- Basic ML pipeline with synthetic data
- API endpoints implementation

## Abstract

This research implements machine learning techniques for customer segmentation and promotional response prediction in physical casino environments. Using anonymized historical data from Imperial Palace Hotel Casino (Bulgaria), we develop a novel framework that bridges synthetic proof-of-concept models with production-ready systems, ensuring both academic rigor and practical applicability.

## Academic Compliance

- **Ethics Approval**: University of Bath Ethics Committee (Ref: 10351-12382)
- **Data Protection**: Full GDPR compliance per Article 4(5) - complete anonymization
- **Academic Integrity**: All code is original work with proper attribution
- **Reproducibility**: Documented methodology with version-controlled dependencies

## Research Contributions

1. **Novel Synthetic-to-Real Data Mapping Framework**: Bridging the gap between academic prototypes and production systems
2. **Domain-Specific Feature Engineering**: Casino-specific temporal and behavioral features
3. **Hybrid ML Pipeline**: Combining unsupervised (K-means) and supervised (Random Forest) learning
4. **Academic-Compliant Architecture**: Demonstrating enterprise patterns in academic context

## Recent Updates (19 June 2025)

- **Batch Import Script**: Ready for Casino IT data (`scripts/import_casino_batch_data.py`)
- **Pipeline Modes**: `--mode batch` for real data, `--mode comparison` for analysis
- **Bug Fixes**: Feature engineering multi_game_player issue resolved


## Project Structure



## Version History
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
