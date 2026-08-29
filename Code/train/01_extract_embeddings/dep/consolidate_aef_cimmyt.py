"""
consolidate_aef_cimmyt.py — Consolidate CIMMYT plot AEF CSV exports into a single parquet.

Run after downloading CSVs from Google Drive folder 'alpha_earth_cimmyt_plot'.

Usage:
  python3 consolidate_aef_cimmyt.py --csv_dir /path/to/downloaded/csvs

Requires: ML_env (pandas, pyarrow)
"""

import os
import argparse
import glob
import pandas as pd


def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory containing downloaded GEE CSV files')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(os.path.expanduser("~"),
                                             "Dropbox", "Projects",
                                             "Maize_prediction", "Data",
                                             "alpha_earth"))
    args =  parser.parse_args()

    csv_files =  sorted(glob.glob(os.path.join(args.csv_dir, "ae_cimmyt_*.csv")))
    print(f"Found {len(csv_files)} CSV files in {args.csv_dir}")

    if not csv_files:
        print("No files found! Check the directory path.")
        return

    dfs =  []
    for f in csv_files:
        df =  pd.read_csv(f)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} rows")

    combined =  pd.concat(dfs, ignore_index=True)

    # Ensure plot_id is string
    combined['plot_id'] =  combined['plot_id'].astype(str)

    # Drop duplicates (in case of overlapping batches)
    combined =  combined.drop_duplicates(subset=['plot_id', 'year'])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path =  os.path.join(args.out_dir, "alpha_earth_cimmyt_plot.parquet")
    combined.to_parquet(out_path, index=False)

    print(f"\nConsolidated: {len(combined):,} plot-year observations")
    print(f"  Unique plots: {combined['plot_id'].nunique():,}")
    print(f"  Years:        {sorted(combined['year'].unique())}")
    print(f"  Columns:      {list(combined.columns)}")
    print(f"  Written to:   {out_path}")


if __name__ == '__main__':
    main()
