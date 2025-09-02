# Academic Submission Guide
## Casino Customer Segmentation - University of Bath

> **Student**: Muhammed Yavuzhan CANLI  
> **Supervisor**: Dr. Moody Alam  
> **Ethics Approval**: 10351-12382  
> **Submission Date**: 2024-2025 Academic Year

---

## 🎓 For Academic Evaluators

### **Quick Start (2 minutes)**

```bash
# 1. Extract the submission package
unzip casino-customer-segmentation-thesis.zip
cd casino-customer-segmentation-thesis/

# 2. Run the academic demo
docker compose -f docker-compose.academic.yml up

# 3. Wait for completion message
# "🎉 Academic demo ready for evaluation!"
```

### **What Happens Automatically**

✅ **PostgreSQL Database** - Starts with academic schema  
✅ **Synthetic Data Generation** - 1000 GDPR-compliant records  
✅ **Feature Engineering** - Behavioral and temporal features  
✅ **Customer Segmentation** - K-means clustering demo  
✅ **Results Export** - CSV files for review  
✅ **Academic Compliance** - Ethics and GDPR validation  

### **Expected Output**

```
[SUCCESS] Academic demo complete!

📊 DEMO RESULTS SUMMARY:
   Total customers: 1000
   Segments created: 4
   Segment 0: 245 customers (avg spend: $523.45)
   Segment 1: 267 customers (avg spend: $445.12)
   Segment 2: 251 customers (avg spend: $612.78)
   Segment 3: 237 customers (avg spend: $389.23)

📋 EVALUATION OPTIONS:
   1. Review generated data: data/demo_results.csv
   2. Check database: docker exec -it casino_postgres_academic psql -U researcher -d casino_research
   3. Run full pipeline: python main_pipeline.py --mode synthetic
   4. View source code: src/ directory
   5. Check academic compliance: README-MINIMAL.md
```

---

## 📦 Package Contents

### **Core Files (< 50MB)**
- `src/` - Complete source code with academic documentation
- `schema/` - PostgreSQL database schemas and views
- `scripts/` - Data processing and analysis scripts
- `academic-demo.py` - **Main evaluation script**
- `docker-compose.academic.yml` - **One-click setup**
- `README-MINIMAL.md` - Complete documentation

### **Generated During Demo**
- `data/synthetic_customers_demo.csv` - Sample dataset
- `data/demo_results.csv` - Segmentation results
- Database tables with academic metadata

---

## 🔬 Academic Evaluation Points

### **1. Code Quality & Architecture**
- **Modular Design**: Clear separation of concerns
- **Academic Standards**: Proper documentation and citations
- **Error Handling**: Robust exception management
- **Testing**: Unit tests in `tests/` directory

### **2. Machine Learning Implementation**
- **Feature Engineering**: Domain-specific casino features
- **Segmentation**: K-means clustering with validation
- **Model Registry**: Version control for ML models
- **Evaluation Metrics**: Academic-standard validation

### **3. Data Ethics & Compliance**
- **GDPR Article 4(5)**: Complete data anonymization
- **Ethics Approval**: University of Bath reference 10351-12382
- **Synthetic Data**: Privacy-preserving research approach
- **Academic Integrity**: Original work with proper attribution

### **4. Technical Innovation**
- **Synthetic-to-Real Mapping**: Novel academic contribution
- **Temporal Analysis**: Period-based segment migration
- **Production-Ready**: Enterprise patterns in academic context
- **Reproducibility**: Version-controlled dependencies

---

## 🛠️ Advanced Evaluation (Optional)

### **Full Pipeline with Real Data**
```bash
# If you have the complete dataset (not included due to size)
python main_pipeline.py --mode batch

# Comparative analysis
python main_pipeline.py --mode comparison
```

### **Database Exploration**
```bash
# Connect to academic database
docker exec -it casino_postgres_academic psql -U researcher -d casino_research

# View academic metadata
\dt academic_*

# Check segmentation results
SELECT segment, COUNT(*), AVG(total_spend) FROM customers GROUP BY segment;
```

### **Source Code Review**
- `src/models/segmentation.py` - Core ML algorithms
- `src/features/feature_engineering.py` - Domain expertise
- `src/data/anonymizer.py` - Privacy protection
- `src/config/academic_config.py` - Compliance settings

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| **Setup Time** | < 5 minutes | ~2-3 minutes |
| **Package Size** | < 50MB | ~20MB |
| **Demo Completion** | < 3 minutes | ~1-2 minutes |
| **Database Ready** | < 30 seconds | ~15 seconds |
| **Synthetic Data** | 1000 records | ✅ Generated |
| **Segments** | 4 clusters | ✅ Created |

---

## 🎯 Thesis Contributions

1. **Novel Framework**: Synthetic-to-real data mapping for sensitive domains
2. **Academic Compliance**: First-class ethics and privacy integration
3. **Production Readiness**: Enterprise patterns in academic research
4. **Reproducibility**: Complete Docker-based research environment
5. **Domain Innovation**: Casino-specific ML feature engineering

---

## 📞 Support & Contact

**Student**: Muhammed Yavuzhan CANLI  
**Email**: mycc21@bath.ac.uk  
**Institution**: University of Bath  
**Department**: Computer Science (Software Engineering)  

**Supervisor**: Dr. Moody Alam  
**Ethics Committee**: Reference 10351-12382  

---

## ⚖️ Academic Declaration

This work represents original research conducted under the supervision of the University of Bath. All code, methodologies, and innovations are the result of independent academic work. External libraries and frameworks are properly attributed. The research complies with all ethical guidelines and data protection regulations.

**Academic Year**: 2024-2025  
**Submission Status**: Ready for Evaluation  
**Package Version**: 1.0-Academic
