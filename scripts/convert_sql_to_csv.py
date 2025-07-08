import pandas as pd
import re

# 📄 Girdi ve çıktı dosyaları
input_file = "src/data/game_events_archive.sql"
output_file = "src/data/game_events_cleaned.csv"

# CSV başlıkları (PostgreSQL tablo ile birebir aynı sırada)
columns = [
    "id", "ts", "player_id", "event_id", "locno", "win", "bet", "credit",
    "point", "denom", "promo_bet", "gameing_day", "asset", "curr_type", "avg_bet"
]

# 📥 SQL dosyasını oku
with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    sql_content = f.read()

# 🔍 INSERT satırlarını regex ile yakala
pattern = r"INSERT INTO `game_events_archive` .*?VALUES\s*(.*?);"
matches = re.findall(pattern, sql_content, re.DOTALL)

if not matches:
    print("❌ INSERT verisi bulunamadı.")
    exit()

# 📌 Satırları ayıkla: her ( ... ) bloğu bir kayıt
raw_values = matches[0]
rows_raw = re.findall(r"\((.*?)\)", raw_values)

data = []
for row in rows_raw:
    # Satırı virgüle göre böl, tırnakları temizle
    parts = [x.strip().strip('"').strip("'") for x in row.split(",")]
    if len(parts) == len(columns):
        data.append(parts)
    else:
        print(f"⚠️  Atlanan satır (sütun sayısı uyuşmuyor): {row}")

# ✅ pandas ile CSV yaz
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Tamamlandı: {len(df)} satır yazıldı → {output_file}")
