import re
import csv

input_file = "src/data/game_events_archive.sql"
output_file = "src/data/game_events_cleaned.csv"

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Tüm INSERT komutlarını yakala
insert_blocks = re.findall(r"INSERT INTO .*? VALUES\s*(.*?);", content, re.DOTALL)

if not insert_blocks:
    print("❌ INSERT verisi bulunamadı.")
    exit()

rows = []
for block in insert_blocks:
    values = block.strip().rstrip(";")
    split_rows = re.findall(r"\((.*?)\)", values)
    rows.extend(split_rows)

print(f" OK- Total Line Found as: {len(rows)}")

# CSV yaz
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "id", "ts", "player_id", "event_id", "locno", "win", "bet", "credit",
        "point", "denom", "promo_bet", "gameing_day", "asset", "curr_type", "avg_bet"
    ])
    for row in rows:
        writer.writerow([x.strip(" '") for x in row.split(",")])
