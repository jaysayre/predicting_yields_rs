"""
clean_harmonic_features.py — Parse unified harmonic-feature CSV exports
(from ls_harmonic_features.py) into training-ready parquet files.

Each exported CSV row has, per spatial unit-year:
  {id_col}, gs_year,
  h_const, h_cos1, h_sin1, h_cos2, h_sin2, h_cos3, h_sin3   (harmonic coeffs)
  h3_f_p12, h3_f_p23, h3_f_p13, h3_q_p12, h3_q_p23, h3_q_p13 (3-period 2D hists)
  h2_f_p12, h2_q_p12                                          (2-period 2D hists)
Each hist column is a GEE fixedHistogram string '[[0.0, n], [1.0, n], ...]'.

Splits into five outputs (the models Phase C trains), each id + year + features:
  <level>_harmonic_coefs.parquet      7 coefficient columns
  <level>_h3_fixed.parquet            768 cols (p12|p23|p13 x 256)
  <level>_h3_quantile.parquet         768 cols
  <level>_h2_fixed.parquet            256 cols
  <level>_h2_quantile.parquet         256 cols

Usage:
  ~/miniforge3/envs/geo_env/bin/python clean_harmonic_features.py \
      --csv_dir <downloaded_csvs> --level {muni|adc|cimmyt} [--bins 16] [--harmonics 3]
"""
import os
import re
import glob
import argparse

import numpy  as np
import pandas as pd

parser =  argparse.ArgumentParser()
parser.add_argument('--csv_dir', required=True, help='dir of GEE-exported CSVs')
parser.add_argument('--out_dir', default=None, help='defaults to Data/harmonic_features')
parser.add_argument('--level', choices=['muni', 'adc', 'cimmyt'], required=True)
parser.add_argument('--bins', type=int, default=16)
parser.add_argument('--harmonics', type=int, default=3)
args =  parser.parse_args()

BINS =  args.bins
NSQ  =  BINS * BINS
ID   =  {'muni': 'muncode', 'adc': 'adcid', 'cimmyt': 'plot_id'}[args.level]
COEF =  ['h_const'] + sum([[f'h_cos{k}', f'h_sin{k}'] for k in range(1, args.harmonics + 1)], [])

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
out_dir  =  args.out_dir or os.path.join(data_dir, "harmonic_features")
os.makedirs(out_dir, exist_ok=True)


def parse_hist(cell):
    """GEE fixedHistogram string -> normalized count vector (len NSQ), or None."""
    if pd.isna(cell) or cell in ('', None):
        return None
    s =  str(cell).replace('[[', '[').replace(']]', ']').strip('][')
    counts =  []
    for entry in s.split('], ['):
        parts =  entry.split(', ')
        if len(parts) >= 2:
            try:
                counts.append(float(parts[1]))
            except ValueError:
                pass
    if not counts:
        return None
    a =  np.zeros(NSQ, dtype=np.float32)
    a[:len(counts)] =  counts[:NSQ]
    tot =  a.sum()
    return a / tot if tot > 0 else None


HIST_COLS =  [f'h3_f_p12', 'h3_f_p23', 'h3_f_p13', 'h3_q_p12', 'h3_q_p23', 'h3_q_p13',
              'h2_f_p12', 'h2_q_p12']

files =  sorted(glob.glob(os.path.join(args.csv_dir, f"harmfeat_{args.level}_*.csv")))
print(f"{len(files)} CSVs for level={args.level}")
if not files:
    raise SystemExit("no harmfeat_*.csv found")

rows =  []
for f in files:
    df =  pd.read_csv(f)
    if ID not in df.columns:
        print(f"  skip {os.path.basename(f)} (no {ID})"); continue
    rows.append(df)
raw =  pd.concat(rows, ignore_index=True)
raw[ID]   =  raw[ID].astype(str)
raw =  raw.rename(columns={'gs_year': 'year'})
print(f"  {len(raw):,} unit-year rows")

# parse each histogram column once
parsed =  {h: raw[h].apply(parse_hist) for h in HIST_COLS}


def build(hcols, prefix, out_name):
    """Assemble id+year + flattened bin columns for the given hist columns."""
    keep =  raw[[ID, 'year']].copy()
    ok   =  np.ones(len(raw), dtype=bool)
    blocks =  []
    for h in hcols:
        arr =  parsed[h]
        ok  &=  arr.notna().values
        mat =  np.vstack([a if a is not None else np.zeros(NSQ, np.float32) for a in arr])
        pair =  h.split('_')[-1]            # p12/p23/p13
        cols =  [f"{prefix}_{pair}_b{j:03d}" for j in range(NSQ)]
        blocks.append(pd.DataFrame(mat, columns=cols, index=raw.index))
    out =  pd.concat([keep] + blocks, axis=1)[ok]
    path =  os.path.join(out_dir, out_name)
    out.to_parquet(path, index=False)
    print(f"  wrote {out_name}  {out.shape}")


# harmonic coefficients
coefs =  raw[[ID, 'year'] + COEF].dropna(subset=COEF)
coefs.to_parquet(os.path.join(out_dir, f"{args.level}_harmonic_coefs.parquet"), index=False)
print(f"  wrote {args.level}_harmonic_coefs.parquet  {coefs.shape}")

build(['h3_f_p12', 'h3_f_p23', 'h3_f_p13'], 'h3f', f"{args.level}_h3_fixed.parquet")
build(['h3_q_p12', 'h3_q_p23', 'h3_q_p13'], 'h3q', f"{args.level}_h3_quantile.parquet")
build(['h2_f_p12'], 'h2f', f"{args.level}_h2_fixed.parquet")
build(['h2_q_p12'], 'h2q', f"{args.level}_h2_quantile.parquet")
print("done")
