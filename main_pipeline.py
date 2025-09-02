
"""
Casino Customer Segmentation - Main ML Pipeline
===============================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science (Software Engineering)
Supervisor: Dr. Moody Alam
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

This script orchestrates the complete ML pipeline from data to predictions.
Supports multiple data sources for academic comparison and research validation.

Academic Contribution:
- Novel hybrid synthetic-real data approach for sensitive domains
- Ethics-first ML pipeline design for responsible gambling research
- Reproducible academic research methodology with full audit trails
"""

# main.py  — minimal orchestrator (Python 3.10+)
import argparse, subprocess, sys, shlex

def run(cmd):
    print(f"\n>>> {cmd}")
    proc = subprocess.run(shlex.split(cmd), stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def main():
    ap = argparse.ArgumentParser(description="Run thesis pipeline end-to-end.")
    ap.add_argument("--periods", nargs="+", required=True, help="e.g. 2022-H1 2022-H2 2023-H1 2023-H2")
    ap.add_argument("--do-segmentation", action="store_true", help="run segmentation_v2.py after features")
    ap.add_argument("--retrain", action="store_true", help="retrain RF before predictions")
    ap.add_argument("--min-feature-coverage", type=float, default=0.95)
    args = ap.parse_args()

    # 1) Features
    run("python src/features/complete_feature_engineering_v3.py")

    # 2) (optional) Segmentation (does not affect RF)
    if args.do_segmentation:
        run("python src/models/segmentation_v2.py")

    # 3) (optional) Train RF (updates LATEST.pkl)
    if args.retrain:
        run("python src/models/rf_clean_harmonized_training_2024.py")

    # 4–6) Per-period prediction + evaluation (+ baselines once)
    baselines_done = False
    for p in args.periods:
        run(f"python src/models/rf_direct_working_predict_2.py --period {p} "
            f"--min-feature-coverage {args.min_feature_coverage}")
        run(f"python src/models/rf_eval_collapse3.py --csv "
            f"results/raw/$(ls -t src/models/working_predictions_{p}_*.csv 2>/dev/null | head -n1) "
            f"--period {p} --out-prefix src/models/{p}_3cls")

        if not baselines_done:
            run("python src/models/rf_make_baseline_simple_models.py")
            baselines_done = True

    # 7) Final comparison tables
    run("python src/models/make_final_comparison_tables.py")

if __name__ == "__main__":
    main()
