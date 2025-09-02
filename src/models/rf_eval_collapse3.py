"""3-Class Operational Evaluation Module

University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

Evaluates Random Forest predictions using a simplified 3-class operational framework.
Maps 5-class predictions to binary operational decisions for business implementation.
"""

import argparse
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 3-class operational space mapping (STANDARD_PROMO -> GROWTH_TARGET; risk categories -> NO_PROMOTION)
MAP3 = {
    "NO_PROMOTION": "NO_PROMOTION",
    "GROWTH_TARGET": "GROWTH_TARGET",
    "STANDARD_PROMO": "GROWTH_TARGET",
    "LOW_ENGAGEMENT": "NO_PROMOTION",
    "INTERVENTION_NEEDED": "NO_PROMOTION",
}

LABELS3 = ["NO_PROMOTION", "GROWTH_TARGET"]  # Fixed order for evaluation

def load_preds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["customer_id", "period", "pred_label"])
    df["pred3"] = df["pred_label"].map(MAP3).fillna("NO_PROMOTION").astype(str)
    return df[["customer_id", "period", "pred3"]]

def load_truth(period: str) -> pd.DataFrame:
    conn = psycopg2.connect(host="localhost", dbname="casino_research",
                            user="researcher", password="academic_password_2024", port=5432)
    q = """SELECT customer_id, period, promo_label FROM casino_data.promo_label WHERE period=%s"""
    t = pd.read_sql(q, conn, params=[period])
    conn.close()
    t["true3"] = t["promo_label"].map(MAP3).fillna("NO_PROMOTION").astype(str)
    return t[["customer_id", "period", "true3"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--period", required=True)
    ap.add_argument("--output-prefix", default=None)
    a = ap.parse_args()

    p = load_preds(a.csv)
    t = load_truth(a.period)
    m = p.merge(t, on=["customer_id", "period"], how="inner")
    y_true = m["true3"].values
    y_pred = m["pred3"].values

    acc = accuracy_score(y_true, y_pred)

    rep = classification_report(
        y_true, y_pred,
        labels=LABELS3, target_names=LABELS3,
        zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS3)

    output_path = a.output_prefix or a.csv.rsplit(".", 1)[0] + "_3cls"
    pd.DataFrame(rep).T.to_csv(f"{output_path}_report.csv", index=True)
    pd.DataFrame(cm, index=LABELS3, columns=LABELS3).to_csv(f"{output_path}_confusion.csv", index=True)
    with open(f"{output_path}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"N={len(m)}  Accuracy(3-classes)={acc:.4f}\n")

    print(f"N={len(m)}  Accuracy(3-classes)={acc:.4f}")
    print(pd.DataFrame(rep).T[["precision","recall","f1-score","support"]])

if __name__ == "__main__":
    main()
