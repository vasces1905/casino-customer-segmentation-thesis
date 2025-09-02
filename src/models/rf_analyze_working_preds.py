"""
Period Analysis from Working Predictions
==========================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

Produces per-period analysis tables/plots from working prediction files. 
& Period analysis figures/CSV
 """


import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="working_predictions_*.csv file")
    ap.add_argument("--out-prefix", default=None, help="output file name prefix (optional)")
    args = ap.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)

    required = {"customer_id","period","pred_label","pred_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    n = len(df)
    labels = df["pred_label"].unique().tolist()

    summary = (df.groupby("pred_label")
       .agg(n=("pred_label","size"),
            mean_conf=("pred_confidence","mean"),
            std_conf=("pred_confidence","std"),
            min_conf=("pred_confidence","min"),
            max_conf=("pred_confidence","max"))
       .sort_values("n", ascending=False)
       .reset_index())

    summary["pct"] = (summary["n"] / n * 100).round(2)
    for c in ["mean_conf","std_conf","min_conf","max_conf"]:
        summary[c] = summary[c].round(3)

    output_prefix = args.out_prefix or os.path.splitext(os.path.basename(csv_path))[0]
    dist_csv = f"{output_prefix}_pred_distribution.csv"
    summary.to_csv(dist_csv, index=False)

    print(f"\nFile: {csv_path}")
    print(f"N = {n:,}")
    print("Labels:", ", ".join(str(x) for x in labels))
    print("\nPrediction distribution (saved to:", dist_csv, ")")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
