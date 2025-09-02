# Casino Customer Segmentation - Minimal Package

> **University of Bath - MSc Computer Science**  
> **Student**: Muhammed Yavuzhan CANLI  
> **Ethics Approval**: 10351-12382

## 📦 Minimal Package (< 50MB)

Bu paket, ana projenin çalışabilir minimum versiyonudur. Büyük veri dosyaları ve modeller ayrı olarak indirilmelidir.

### 🚀 Hızlı Başlangıç

```bash
# 1. Setup'ı çalıştır
python setup-minimal.py

# 2. Docker Compose ile başlat
docker compose -f docker-compose.minimal.yml up

# 3. Uygulamaya erişim
# PostgreSQL: localhost:5432
# App: Container içinde çalışır
```

### 📁 Paket İçeriği

**Dahil Edilenler (~15-20MB):**
- ✅ Kaynak kod (`src/`)
- ✅ Veritabanı şemaları (`schema/`)
- ✅ Temel scriptler (`scripts/`)
- ✅ Docker konfigürasyonu
- ✅ Minimal requirements

**Ayrı İndirilecekler:**
- 📥 ML Modelleri (`.pkl` dosyaları - ~800MB)
- 📥 Büyük veri setleri (`.csv` dosyaları - ~50MB)
- 📥 Backup veritabanı (`backup.sql` - 2.7GB)

### 🔧 Kurulum Detayları

#### Önkoşullar
- Docker Desktop
- Python 3.9+
- 4GB RAM (minimum)

#### Manuel Kurulum
```bash
# 1. Bağımlılıkları yükle
pip install -r requirements-minimal.txt

# 2. Çevre değişkenlerini ayarla
cp .env.example .env

# 3. PostgreSQL'i başlat
docker compose -f docker-compose.minimal.yml up postgres -d

# 4. Şemaları yükle
psql -h localhost -U researcher -d casino_research -f schema/01_create_database.sql
```

### 📊 Boyut Karşılaştırması

| Versiyon | Boyut | İçerik |
|----------|-------|--------|
| **Tam Proje** | 4.18GB | Tüm veriler + modeller + backup |
| **Minimal** | ~20MB | Sadece kod + şema + Docker |
| **Modeller** | ~800MB | Ayrı indirme |
| **Veriler** | ~50MB | Ayrı indirme |

### 🔗 Büyük Dosyaları İndirme

```bash
# Modelleri indir (Google Drive/OneDrive linklerinden)
# [LİNKLER EKLENECEk]

# Verileri indir
# [LİNKLER EKLENECEk]

# models/ klasörüne yerleştir
mkdir models
# İndirilen .pkl dosyalarını models/ içine kopyala
```

### 🎯 Kullanım Senaryoları

1. **Demo/Test**: Minimal paket + synthetic data
2. **Geliştirme**: Minimal paket + model subset
3. **Production**: Minimal paket + tüm modeller
4. **Akademik**: Minimal paket + documentation

### ⚠️ Önemli Notlar

- Bu minimal paket sadece temel çalışma için yeterlidir
- Tam fonksiyonalite için büyük dosyalar gereklidir
- Akademik kullanım için ethics approval gereklidir
- GDPR uyumlu - tamamen anonimleştirilmiş veriler

### 🆘 Sorun Giderme

**Docker build hatası:**
```bash
docker system prune -a
docker compose -f docker-compose.minimal.yml build --no-cache
```

**PostgreSQL bağlantı hatası:**
```bash
docker compose -f docker-compose.minimal.yml logs postgres
```

**Model yükleme hatası:**
- Modellerin `models/` klasöründe olduğunu kontrol edin
- Dosya izinlerini kontrol edin

---

**İletişim**: Muhammed Yavuzhan CANLI - University of Bath  
**Proje**: MSc Computer Science Thesis  
**Tarih**: 2024-2025
