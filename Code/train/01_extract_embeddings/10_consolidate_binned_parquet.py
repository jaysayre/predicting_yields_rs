"""
10_consolidate_binned_parquet.py
================================
Consolidate the per-state GEE CSV exports of fixed-width BINNED AEF
histograms (6_ee_alpha_earth_binned_hist.py: 64 dims x 8 bins on [-0.8, 0.8])
into the parquet files behind the paper's "AEF Hist" model:

  --level mun   Data/alpha_earth/mun_binned_hist_csvs/aef_binhist_mun_state_*.csv
                -> alpha_earth_mex_mun_binned_hist.parquet   (512 bins, CVE_ENT, CVE_MUN, year, muncode)
  --level adc   Data/alpha_earth/adcs_binned_hist_csvs/aef_binhist_state_*.csv
                -> alpha_earth_mex_adcs_binned_hist.parquet  (512 bins, adcid, year, muncode)

Same schema as 8_/9_consolidate_hist_parquet*.py (percentile features) and
consolidate_qbin_parquet.py (quantile-bin robustness variant).

Usage:
  python3 10_consolidate_binned_parquet.py --level mun
  python3 10_consolidate_binned_parquet.py --level adc [--csv_dir DIR] [--out_dir DIR]

Author: James Sayre
Date: 2026-09-17
"""

import os
import glob
import argparse
import pandas as pd

AEF_DIR  =  os.path.join(os.path.expanduser("~"), "Dropbox", "Projects",
                         "Maize_prediction", "Data", "alpha_earth")
BIN_COLS =  [f"A{d:02d}_b{b}" for d in range(64) for b in range(8)]

LEVELS = {
    'mun': {'csv_dir': 'mun_binned_hist_csvs',  'pattern': 'aef_binhist_mun_state_*.csv',
            'out': 'alpha_earth_mex_mun_binned_hist.parquet'},
    'adc': {'csv_dir': 'adcs_binned_hist_csvs', 'pattern': 'aef_binhist_state_*.csv',
            'out': 'alpha_earth_mex_adcs_binned_hist.parquet'},
}


def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--level',   choices=list(LEVELS), required=True)
    parser.add_argument('--csv_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=AEF_DIR)
    args =  parser.parse_args()
    cfg  =  LEVELS[args.level]
    csv_dir   =  args.csv_dir or os.path.join(AEF_DIR, cfg['csv_dir'])
    csv_files =  sorted(glob.glob(os.path.join(csv_dir, cfg['pattern'])))
    print(f"Found {len(csv_files)} CSV files in {csv_dir}")
    if not csv_files:
        raise SystemExit("No files found! Check --csv_dir.")

    combined =  pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if args.level == 'mun':
        combined['CVE_ENT'] =  combined['CVE_ENT'].astype(str).str.zfill(2)
        combined['CVE_MUN'] =  combined['CVE_MUN'].astype(str).str.zfill(3)
        combined['muncode'] =  combined['CVE_ENT'] + combined['CVE_MUN']
        ids =  ['CVE_ENT', 'CVE_MUN', 'year', 'muncode']
        combined =  combined.drop_duplicates(subset=['CVE_ENT', 'CVE_MUN', 'year'])
    else:
        combined['adcid']   =  combined['adcid'].astype(str)
        combined['muncode'] =  combined['adcid'].str[:5]
        ids =  ['adcid', 'year', 'muncode']
        combined =  combined.drop_duplicates(subset=['adcid', 'year'])
    combined['year'] =  combined['year'].astype(int)
    combined =  combined[BIN_COLS + ids]

    out_path =  os.path.join(args.out_dir, cfg['out'])
    combined.to_parquet(out_path, index=False)
    print(f"Consolidated {len(combined):,} unit-years x {len(combined.columns)} cols -> {out_path}")


if __name__ == '__main__':
    main()
