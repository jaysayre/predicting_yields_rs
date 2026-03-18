"""
consolidate_hist_parquet_mun.py
===============================
Consolidate per-state GEE CSV exports of distributional AEF features
(municipality level) into a single parquet file.

Run after downloading CSVs from Google Drive folder 'alpha_earth_mun_hist'.

Usage:
  python3 consolidate_hist_parquet_mun.py --csv_dir /path/to/downloaded/csvs

Author: Jay Sayre
Date: 2026-02-23
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

    csv_files =  sorted(glob.glob(os.path.join(args.csv_dir, "aef_hist_mun_*.csv")))
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

    # Drop duplicates (in case of overlapping batches)
    combined =  combined.drop_duplicates(subset=['CVE_ENT', 'CVE_MUN', 'year'])

    # Build muncode from CVE_ENT + CVE_MUN
    combined['muncode'] =  combined['CVE_ENT'].astype(str).str.zfill(2) + \
                           combined['CVE_MUN'].astype(str).str.zfill(3)

    out_path =  os.path.join(args.out_dir, "alpha_earth_mex_mun_hist.parquet")
    combined.to_parquet(out_path, index=False)

    print(f"\nConsolidated: {len(combined):,} municipality-year observations")
    print(f"  Unique muncodes: {combined['muncode'].nunique():,}")
    print(f"  Years:           {sorted(combined['year'].unique())}")
    print(f"  Columns:         {len(combined.columns)}")
    print(f"  Written to:      {out_path}")


if __name__ == '__main__':
    main()
