"""Multi-Period Customer Segmentation Pipeline

University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

Automated pipeline for running customer segmentation across multiple time periods.
Executes K-means clustering workflows for comprehensive temporal segmentation analysis.
"""

import argparse, os, sys, subprocess, pathlib

def run(cmd: list[str]):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-script", required=True, help="segmentation.py path (e.g., ./src/models/segmentation.py)")
    ap.add_argument("--periods", nargs="+", required=True, help="e.g., 2022-H1 2022-H2 2023-H1 ...")
    ap.add_argument("--outdir", required=True, help="Output directory (CSV files will be saved here)")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--python-exe", default=sys.executable, help="Alternative Python executable if needed")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in args.periods:
        cmd = [
            args.python_exe,
            args.seg-script if hasattr(args, "seg-script") else args.seg_script,  # argparser dash fix
            "--period", p,
            "--n_clusters", str(args.n_clusters),
            "--output_dir", str(outdir),
        ]
        # argparse may store "--seg-script" parameter as "seg_script"; above line handles this
        if not os.path.exists(cmd[1]):
            # Try alternative attribute name
            cmd[1] = getattr(args, "seg_script", None) or args.seg_script
        run(cmd)

    print("\n All Periods Completed. For CSV's:", outdir.resolve())

if __name__ == "__main__":
    main()
