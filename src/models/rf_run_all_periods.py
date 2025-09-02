"""Multi-Period Random Forest Prediction Pipeline

University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

Automated pipeline for running Random Forest predictions across multiple time periods.
Executes prediction and evaluation workflows for comprehensive temporal analysis.
"""

import os, sys, glob, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

def run_period(period: str):
    print("\n" + "="*80)
    print(f"RUN: {period}")

    # 1) Run Random Forest prediction (set working directory to BASE -> output falls into src/models)
    cmd1 = [sys.executable,
            os.path.join(BASE, "rf_direct_working_predict_2.py"),
            "--period", period, "--min-feature-coverage", "0.95"]
    subprocess.check_call(cmd1, cwd=BASE)

    # 2) Find generated CSV file (search both BASE and root directories)
    patterns = [
        os.path.join(BASE, f"working_predictions_{period}_*.csv"),
        os.path.join(os.getcwd(), f"working_predictions_{period}_*.csv"),
    ]
    files = []
    for pat in patterns:
        files += glob.glob(pat)

    if not files:
        raise RuntimeError(
            f"No predictions CSV found for {period}\nSearched:\n" + "\n".join(patterns)
        )

    csv_path = max(files, key=os.path.getmtime)
    print(f"→ Found CSV: {csv_path}")

    # 3) 3-class operational evaluation
    output_prefix = os.path.join(BASE, f"{period}_3cls")
    cmd2 = [sys.executable,
            os.path.join(BASE, "rf_eval_collapse3.py"),
            "--csv", csv_path,
            "--period", period,
            "--output-prefix", output_prefix]
    subprocess.check_call(cmd2, cwd=BASE)

    print(f"Completed: {period}  ->  {os.path.basename(csv_path)}  and  {os.path.basename(output_prefix)}_*")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", nargs="+",
                    default=["2022-H1", "2022-H2", "2023-H1"],
                    help="e.g., 2022-H1 2022-H2 2023-H1")
    args = ap.parse_args()
    for p in args.periods:
        run_period(p)

if __name__ == "__main__":
    main()
