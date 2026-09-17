"""
14_consolidate_aef_cimmyt_all.py
=============================
Consolidate CIMMYT plot AEF CSV exports (mean, hist, or binned_hist)
into a single parquet.

Run after downloading CSVs from the corresponding Google Drive folder.

Usage:
  python3 14_consolidate_aef_cimmyt_all.py --type mean --csv_dir /path/to/csvs
  python3 14_consolidate_aef_cimmyt_all.py --type hist --csv_dir /path/to/csvs
  python3 14_consolidate_aef_cimmyt_all.py --type binned_hist --csv_dir /path/to/csvs

Requires: mpc_env or ML_env (pandas, pyarrow)

Author: Jay Sayre
Date: 2026-03-21
"""

import os
import argparse
import glob
import pandas as pd


TYPE_CONFIG = {
    'mean': {
        'csv_prefix': 'ae_cimmyt_',
        'out_name':   'alpha_earth_cimmyt_plot.parquet',
    },
    'hist': {
        'csv_prefix': 'ae_cimmyt_hist_',
        'out_name':   'alpha_earth_cimmyt_plot_hist.parquet',
    },
    'binned_hist': {
        'csv_prefix': 'ae_cimmyt_binhist_',
        'out_name':   'alpha_earth_cimmyt_plot_binned_hist.parquet',
    },
}


def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True,
                        choices=['mean', 'hist', 'binned_hist'],
                        help='Type of extraction to consolidate')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory containing downloaded GEE CSV files')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(os.path.expanduser("~"),
                                             "Dropbox", "Projects",
                                             "Maize_prediction", "Data",
                                             "alpha_earth"))
    args =  parser.parse_args()

    cfg       =  TYPE_CONFIG[args.type]
    prefix    =  cfg['csv_prefix']
    out_name  =  cfg['out_name']

    csv_files =  sorted(glob.glob(os.path.join(args.csv_dir, f"{prefix}*.csv")))
    print(f"Type: {args.type}")
    print(f"Found {len(csv_files)} CSV files matching '{prefix}*.csv' "
          f"in {args.csv_dir}")

    if not csv_files:
        print("No files found! Check the directory path and prefix.")
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
    out_path =  os.path.join(args.out_dir, out_name)
    combined.to_parquet(out_path, index=False)

    print(f"\nConsolidated: {len(combined):,} plot-year observations")
    print(f"  Unique plots: {combined['plot_id'].nunique():,}")
    print(f"  Years:        {sorted(combined['year'].unique())}")
    print(f"  Columns:      {len(combined.columns)}")
    print(f"  Written to:   {out_path}")


if __name__ == '__main__':
    main()
