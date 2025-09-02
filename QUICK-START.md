# 🎰 Casino Customer Segmentation - Quick Start Guide

**University of Bath - Academic Thesis Project**  
**Student:** Muhammed Yavuzhan CANLI  
**Ethics Approval:** 10351-12382

## 🚀 One-Click Setup

### Option 1: Complete Auto-Installation (Recommended)
**Double-click:** `install-docker.ps1`
- Automatically installs Docker Desktop if not present
- Sets up complete environment
- Restarts required after Docker installation

### Option 2: Quick Start (Docker Already Installed)
**Double-click:** `start-casino-environment.bat`
- Starts Docker Desktop if needed
- Launches PostgreSQL database
- Starts casino analysis environment

### Option 3: Manual Setup
**Double-click:** `setup-docker.bat`
- Manual Docker setup process
- Full control over installation steps

## 📋 What Gets Installed

### 🐳 Docker Environment
- **PostgreSQL 16** database on port 5432
- **Python 3.9** analysis environment
- **Academic demo** application
- **Persistent data** storage

### 🎯 Database Configuration
```
Host: localhost:5432
Database: casino_research
Username: researcher
Password: academic_password_2024
```

### 📊 Available Services
- Customer segmentation models
- Promotion response prediction
- Temporal analysis tools
- Academic compliance features

## 🛠️ Management Commands

### Start Environment
```bash
docker-compose -f docker-compose.academic.yml up -d
```

### View Logs
```bash
docker-compose -f docker-compose.academic.yml logs -f
```

### Stop Environment
```bash
docker-compose -f docker-compose.academic.yml down
```

### Restart Services
```bash
docker-compose -f docker-compose.academic.yml restart
```

## 📁 Project Structure
```
casino-customer-segmentation-thesis/
├── 🚀 install-docker.ps1           # Auto-installer
├── 🎯 start-casino-environment.bat # Quick launcher  
├── ⚙️ setup-docker.bat             # Manual setup
├── 🐳 docker-compose.academic.yml  # Academic environment
├── 📊 src/                         # Analysis code
├── 🗄️ schema/                      # Database schemas
└── 📈 models/                      # ML models
```

## ✅ Success Indicators

When setup is complete, you'll see:
- ✅ PostgreSQL database running
- ✅ Casino demo application active
- ✅ Academic analysis tools available
- ✅ All containers healthy

## 🔧 Troubleshooting

### Docker Not Starting
1. Ensure Docker Desktop is installed
2. Restart computer after Docker installation
3. Run as Administrator if needed

### Port Conflicts
- PostgreSQL uses port 5432
- Ensure no other PostgreSQL instances running

### Permission Issues
- Run PowerShell as Administrator
- Check Windows execution policy

## 🎓 Academic Compliance

This environment includes:
- Ethics approval metadata
- Academic logging standards
- University of Bath compliance
- Thesis evaluation tools

---

**Ready for academic evaluation and thesis demonstration!**
