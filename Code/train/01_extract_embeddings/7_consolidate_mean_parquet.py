"""
7_consolidate_mean_parquet.py
=============================
Consolidate the per-state GEE CSV exports of MEAN AEF embeddings
(1_ee_alpha_earth_download.py, MODE = "mexico_mun" / "mexico_adc") into the
two parquet files every model reads:

  --level mun   Drive folder alpha_earth_agmask/  (alpha_earth_agmask_state_*.csv)
                -> Data/alpha_earth/alpha_earth_mex_muns.parquet   (A00..A63, CVE_ENT, CVE_MUN, year)
  --level adc   Drive folder alpha_earth_adcs/    (alpha_earth_adc_state_*.csv)
                -> Data/alpha_earth/alpha_earth_mex_adcs.parquet   (A00..A63, adcid, year)

Usage:
  python3 7_consolidate_mean_parquet.py --level mun --csv_dir /path/to/alpha_earth_agmask
  python3 7_consolidate_mean_parquet.py --level adc --csv_dir /path/to/alpha_earth_adcs

Author: James Sayre
Date: 2026-09-17
"""

import os
import glob
import argparse
import pandas as pd

FEAT_COLS =  [f"A{d:02d}" for d in range(64)]

LEVELS = {
    'mun': {'pattern': 'alpha_earth_agmask_state_*.csv',
            'ids':     ['CVE_ENT', 'CVE_MUN'],
            'out':     'alpha_earth_mex_muns.parquet'},
    'adc': {'pattern': 'alpha_earth_adc_state_*.csv',
            'ids':     ['adcid'],
            'out':     'alpha_earth_mex_adcs.parquet'},
}


def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--level',   choices=list(LEVELS), required=True)
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory holding the downloaded GEE CSV exports')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(os.path.expanduser("~"), "Dropbox", "Projects",
                                             "Maize_prediction", "Data", "alpha_earth"))
    args =  parser.parse_args()
    cfg  =  LEVELS[args.level]

    csv_files =  sorted(glob.glob(os.path.join(args.csv_dir, cfg['pattern'])))
    print(f"Found {len(csv_files)} CSV files in {args.csv_dir}")
    if not csv_files:
        raise SystemExit("No files found! Check --csv_dir.")

    combined =  pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if args.level == 'mun':
        combined['CVE_ENT'] =  combined['CVE_ENT'].astype(str).str.zfill(2)
        combined['CVE_MUN'] =  combined['CVE_MUN'].astype(str).str.zfill(3)
    else:
        combined['adcid']   =  combined['adcid'].astype(str)
    combined['year'] =  combined['year'].astype(int)

    combined =  combined.drop_duplicates(subset=cfg['ids'] + ['year'])
    combined =  combined[FEAT_COLS + cfg['ids'] + ['year']]

    out_path =  os.path.join(args.out_dir, cfg['out'])
    combined.to_parquet(out_path, index=False)
    print(f"Consolidated {len(combined):,} unit-years x {len(combined.columns)} cols -> {out_path}")


if __name__ == '__main__':
    main()
