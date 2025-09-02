# 🎓 Academic Evaluator Setup Guide

**Casino Customer Segmentation Thesis - University of Bath**  
**Student:** Muhammed Yavuzhan CANLI  
**Ethics Approval:** 10351-12382

---

## 📦 **For Academic Evaluators**

### **Option 1: Complete Auto-Setup (Recommended)**

1. **Extract ZIP file** to any directory
2. **Right-click** `install-docker.ps1` → **"Run with PowerShell"**
3. **Wait** for automatic Docker installation (if needed)
4. **Restart computer** if prompted
5. **Double-click** `start-casino-environment.bat`

### **Option 2: Manual Docker Setup**

1. **Extract ZIP file** to any directory
2. **Install Docker Desktop** from: https://www.docker.com/products/docker-desktop/
3. **Restart computer** after Docker installation
4. **Double-click** `start-casino-environment.bat`

### **Option 3: Command Line (Advanced)**

```bash
# Extract and navigate
unzip casino-customer-segmentation-thesis.zip
cd casino-customer-segmentation-thesis/

# Start academic environment
docker-compose -f docker-compose.academic.yml up --build
```

---

## ✅ **What Happens Automatically**

```
🐳 Docker Desktop starts automatically
🐘 PostgreSQL database launches (port 5432)
🎰 Casino analysis environment ready
📊 Academic demo data generated
🔬 All models and tools available
```

### **Database Access**
```
Host: localhost:5432
Database: casino_research
Username: researcher  
Password: academic_password_2024
```

---

## 🎯 **Academic Evaluation Commands**

### **Quick Demo (2 minutes)**
```bash
# Run academic demonstration
python academic-demo.py
```

### **Full Pipeline Analysis**
```bash
# Synthetic data analysis
python main_pipeline.py --mode synthetic

# Batch processing demo  
python main_pipeline.py --mode batch

# Model comparison
python main_pipeline.py --mode comparison
```

### **Database Exploration**
```bash
# Connect to database
docker exec -it casino_postgres_academic psql -U researcher -d casino_research

# View tables
\dt

# Check segmentation results
SELECT segment, COUNT(*), AVG(total_spend) FROM customers GROUP BY segment;
```

---

## 📊 **Expected Results**

### **Segmentation Output**
```
Total customers: 1000 (synthetic)
Segments created: 4
Segment 0: ~250 customers (High Value)
Segment 1: ~250 customers (Medium Value)  
Segment 2: ~250 customers (Low Value)
Segment 3: ~250 customers (New/Inactive)
```

### **Generated Files**
- `data/synthetic_customers_demo.csv` - Sample dataset
- `data/demo_results.csv` - Segmentation results
- `models/Output/` - Trained ML models
- Database tables with academic metadata

---

## 🔧 **Troubleshooting**

### **Docker Issues**
- Ensure Docker Desktop is running
- Restart computer after Docker installation
- Run PowerShell as Administrator if needed

### **Port Conflicts**
- PostgreSQL uses port 5432
- Stop other PostgreSQL instances if running

### **Permission Issues**
- Right-click PowerShell → "Run as Administrator"
- Check Windows execution policy: `Set-ExecutionPolicy RemoteSigned`

---

## 📋 **Evaluation Checklist**

### **Code Quality** ✅
- [ ] Modular architecture in `src/` directory
- [ ] Academic documentation standards
- [ ] Error handling and logging
- [ ] Unit tests available

### **Machine Learning** ✅  
- [ ] Feature engineering (`src/features/`)
- [ ] K-means segmentation (`src/models/`)
- [ ] Model validation and metrics
- [ ] Reproducible results

### **Academic Compliance** ✅
- [ ] Ethics approval metadata
- [ ] GDPR-compliant synthetic data
- [ ] University of Bath standards
- [ ] Proper citations and attribution

### **Technical Innovation** ✅
- [ ] Synthetic-to-real data mapping
- [ ] Temporal customer analysis
- [ ] Production-ready architecture
- [ ] Docker containerization

---

## 📞 **Support**

**Student:** Muhammed Yavuzhan CANLI  
**Email:** mycc21@bath.ac.uk  
**Supervisor:** Dr. Moody Alam  
**Institution:** University of Bath

**Quick Help:**
- All setup files are in the root directory
- Double-click `.bat` files for Windows
- Use `docker-compose.academic.yml` for evaluation
- Check `QUICK-START.md` for detailed instructions

---

**Ready for academic evaluation in under 5 minutes!**
