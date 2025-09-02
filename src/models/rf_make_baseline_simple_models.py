"""Baseline Model Comparison Framework

University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

Implements baseline machine learning models for casino customer promotion prediction.
Compares multiple algorithms using a simplified 16-feature set with cross-validation.
"""
import argparse, pandas as pd, psycopg2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

SIMPLE_FEATS = [
    "total_bet","avg_bet","loss_rate","total_sessions",
    "days_since_last_visit","session_duration_volatility",
    "loss_chasing_score","sessions_last_30d","bet_trend_ratio",
    "bet_volatility","weekend_preference","late_night_player",
    "game_diversity","machine_diversity","zone_diversity","bet_std"
]

MAP3 = {
    "NO_PROMOTION":"NO_PROMOTION",
    "GROWTH_TARGET":"GROWTH_TARGET",
    "STANDARD_PROMO":"GROWTH_TARGET",
    "LOW_ENGAGEMENT":"NO_PROMOTION",
    "INTERVENTION_NEEDED":"NO_PROMOTION",
}

def load_xy(period):
    conn = psycopg2.connect(host="localhost", dbname="casino_research",
                            user="researcher", password="academic_password_2024", port=5432)
    qX = f"""
        SELECT customer_id, {", ".join(SIMPLE_FEATS)}
        FROM casino_data.customer_features
        WHERE analysis_period = %s
    """
    X = pd.read_sql(qX, conn, params=[period])

    qy = """
        SELECT customer_id, promo_label
        FROM casino_data.promo_label
        WHERE period = %s
    """
    y = pd.read_sql(qy, conn, params=[period])
    conn.close()

    df = X.merge(y, on="customer_id", how="inner")
    df["label3"] = df["promo_label"].map(MAP3)
    df = df.dropna(subset=["label3"])

    # Simple data cleaning: fill remaining NaN values with 0 (sufficient for simple feature set)
    df[SIMPLE_FEATS] = df[SIMPLE_FEATS].fillna(0.0)

    return df[SIMPLE_FEATS].values, df["label3"].values, df.shape[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True)  # e.g., 2023-H2
    ap.add_argument("--out", default="baseline_simple_results.csv")
    args = ap.parse_args()

    X, y, n = load_xy(args.period)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic_Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "Naive_Bayes": GaussianNB(),
    }

    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=None)
        rows.append({"algorithm": name, "feature_set": "simple(16)", "accuracy": scores.mean()})

    out = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"N={n}  period={args.period}")
    print(out.to_string(index=False))
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
