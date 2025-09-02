# -*- coding: utf-8 -*-
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

CSV_PATH = "./src/models/Output/unified_temporal_evolution.csv"

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="casino_research",
        user="researcher",
        password="academic_password_2024"
    )

def period_key(p: str) -> int:
    # '2023-H2' -> 2023*2 + 2  (H1=1, H2=2)
    y, h = p.split("-H")
    return int(y) * 2 + (2 if h == "2" else 1)

def build_temporal_categories_from_csv(csv_path=CSV_PATH) -> pd.DataFrame:
    """CSV'deki evolution_status'u yok say, kategoriyi burada üret."""
    df = pd.read_csv(csv_path)
    needed = {"customer_id", "period", "predicted_promotion", "confidence"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV eksik kolonlar: {miss}")

    # Sadece (customer_id, period) bazında tekilleştir (ASLA sadece customer_id değil)
    df = df.drop_duplicates(subset=["customer_id", "period"], keep="last").copy()

    # Dönem sırası ve müşteri içi sıralama
    df["period_seq"] = df["period"].astype(str).map(period_key)
    df = df.sort_values(["customer_id", "period_seq"]).reset_index(drop=True)

    # Önceki dönem değerleri
    df["prev_label"] = df.groupby("customer_id")["predicted_promotion"].shift(1)
    df["prev_conf"]  = df.groupby("customer_id")["confidence"].shift(1)

    # Kategori kuralı
    def categorize(r):
        if pd.isna(r["prev_label"]):
            return "NEW_CUSTOMER"
        c, p = r["confidence"], r["prev_conf"]
        delta_ok = pd.notna(c) and pd.notna(p)
        if r["predicted_promotion"] != r["prev_label"]:
            return "DYNAMIC_EVOLUTION"
        if delta_ok and (c > p + 0.05):
            return "SUCCESS_STORY"
        if delta_ok and (c < p - 0.05):
            return "RISK_ALERT"
        return "STABLE_CUSTOMER"

    df["temporal_category"] = df.apply(categorize, axis=1)

    # cluster_label CSV'de olmayabilir
    if "cluster_label" not in df.columns:
        df["cluster_label"] = None

    # DB şemasına uygun kolonlar
    out = df.rename(columns={"period": "period_id",
                             "predicted_promotion": "promo_label"})
    return out[["customer_id", "period_id", "cluster_label",
                "promo_label", "temporal_category", "confidence"]]

def upsert_into_db(final_df: pd.DataFrame):
    conn = get_connection()
    cur = conn.cursor()

    sql = """
        INSERT INTO casino_data.temporal_promo_evolution
        (customer_id, period_id, cluster_label, promo_label,
         temporal_category, confidence, analysis_date)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (customer_id, period_id) DO UPDATE SET
            cluster_label     = EXCLUDED.cluster_label,
            promo_label       = EXCLUDED.promo_label,
            temporal_category = EXCLUDED.temporal_category,
            confidence        = EXCLUDED.confidence,
            analysis_date     = CURRENT_TIMESTAMP;
    """
    data = list(final_df.itertuples(index=False, name=None))
    execute_batch(cur, sql, data, page_size=5000)

    # KMeans segment etiketi boşsa doldur
    cur.execute("""
        UPDATE casino_data.temporal_promo_evolution t
        SET cluster_label = k.segment_label
        FROM casino_data.kmeans_segments k
        WHERE t.customer_id = k.customer_id
          AND t.period_id   = k.period_id
          AND t.cluster_label IS NULL;
    """)

    conn.commit()
    cur.close()
    conn.close()

def quick_checks():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT temporal_category, COUNT(*) AS cnt
        FROM casino_data.temporal_promo_evolution
        GROUP BY temporal_category
        ORDER BY cnt DESC;
    """)
    print("\n[CHECK] temporal_category dağılımı:", cur.fetchall())

    cur.execute("""
        SELECT COUNT(*) AS customers_with_multi_period
        FROM (
          SELECT customer_id
          FROM casino_data.temporal_promo_evolution
          GROUP BY customer_id
          HAVING COUNT(DISTINCT period_id) > 1
        ) s;
    """)
    print("[CHECK] customers_with_multi_period:", cur.fetchone()[0])

    cur.execute("""
        SELECT
          SUM(CASE WHEN cluster_label IS NULL THEN 1 ELSE 0 END) AS null_cluster_labels,
          COUNT(*) AS total_rows
        FROM casino_data.temporal_promo_evolution;
    """)
    print("[CHECK] null_cluster_labels / total_rows:", cur.fetchone())

    cur.close()
    conn.close()

if __name__ == "__main__":
    df = build_temporal_categories_from_csv(CSV_PATH)
    upsert_into_db(df)
    quick_checks()
    print("[OK] temporal_promo_evolution upsert + checks completed")
