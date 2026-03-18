"""
clean_3period_histograms.py — Parse 3-period 2D histogram CSV exports from GEE
into training-ready pickle files.

GEE exports from ls_3period_hists.py produce CSVs with columns:
  {id_col}, hist_p12, hist_p23, hist_p13
Each histogram column is a string: '[[0.0, count], [1.0, count], ...]'

This script:
  1. Reads all CSVs, extracts year from filename
  2. Parses histogram strings and normalizes to proportions
  3. Concatenates into 768-element feature vectors (3 pairs × 256 bins)
  4. Saves per-year pickle files

Usage:
  python clean_3period_histograms.py <raw_csv_dir> <output_dir> <bins> <level>
  python clean_3period_histograms.py Data/muni_3period_hists_16bins Data/muni_3period_hists_clean 16 muni
  python clean_3period_histograms.py Data/adc_3period_hists_16bins Data/adc_3period_hists_clean 16 adc

Run from Maize_prediction/ directory.
Requires: ML_env (pandas, numpy)
"""

import os
import sys
import re

import numpy  as np
import pandas as pd

raw_dir  =  sys.argv[1]   # Directory with GEE-exported CSVs
out_dir  =  sys.argv[2]   # Output directory for pickles
bins     =  int(sys.argv[3])
level    =  sys.argv[4]   # "muni" or "adc"

if level == "muni":
    id_col =  "muncode"
elif level == "adc":
    id_col =  "adcid"
elif level == "cimmyt":
    id_col =  "plot_id"
n_bins_sq =  bins * bins   # 256 for bins=16

os.makedirs(out_dir, exist_ok=True)


def parse_hist_cell(cell):
    """
    Parse a GEE histogram string into a numpy array of bin counts.
    Input:  '[[0.0, 15.0], [1.0, 23.0], ...]'
    Output: np.array([15.0, 23.0, ...]) normalized to sum to 1
    """
    if pd.isna(cell) or cell == '' or cell is None:
        return None

    cell    =  str(cell)
    entries =  cell.replace('[[', '[').replace(']]', ']').strip('][').split('], [')
    counts  =  []
    for entry in entries:
        parts =  entry.split(', ')
        if len(parts) >= 2:
            counts.append(float(parts[1]))

    counts =  np.array(counts)
    total  =  counts.sum()
    if total > 0:
        counts =  counts / total
    return counts


# ── Read all CSVs ────────────────────────────────────────

csv_files =  [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files in {raw_dir}")

if len(csv_files) == 0:
    print("No CSV files found. Exiting.")
    sys.exit(1)

# Process per-year to manage memory (especially for ADC level)
year_dfs =  {}

for f in sorted(csv_files):
    # Extract year from filename: 3period_hist_{level}_{state}_{batch}_{year}.csv
    match =  re.search(r'_(\d{4})\.csv$', f)
    if match:
        year =  int(match.group(1))
    else:
        print(f"  Skipping {f} — can't parse year")
        continue

    df        =  pd.read_csv(os.path.join(raw_dir, f))
    df['year'] =  year

    if year not in year_dfs:
        year_dfs[year] =  []
    year_dfs[year].append(df)

print(f"Years found: {sorted(year_dfs.keys())}")


# ── Process each year ────────────────────────────────────

hist_cols =  ['hist_p12', 'hist_p23', 'hist_p13']

for year in sorted(year_dfs.keys()):
    combined =  pd.concat(year_dfs[year], ignore_index=True)
    print(f"\nYear {year}: {len(combined)} rows")

    # Parse histogram columns
    for col in hist_cols:
        if col in combined.columns:
            combined[col] =  combined[col].apply(parse_hist_cell)
        else:
            print(f"  Warning: column {col} not found, filling with zeros")
            combined[col] =  combined.apply(lambda x: np.zeros(n_bins_sq), axis=1)

    # Drop rows with any missing or wrong-length histograms
    n_before =  len(combined)
    for col in hist_cols:
        combined =  combined[combined[col].apply(
            lambda x: x is not None and isinstance(x, np.ndarray) and len(x) == n_bins_sq
        )]
    combined =  combined.reset_index(drop=True)
    n_dropped =  n_before - len(combined)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with bad histograms")

    # Concatenate into single 768-element feature vector
    combined['hist'] =  combined.apply(
        lambda row: np.concatenate([row['hist_p12'], row['hist_p23'], row['hist_p13']]),
        axis=1
    )

    # Keep only needed columns
    out_cols =  [id_col, 'year', 'hist', 'hist_p12', 'hist_p23', 'hist_p13']
    out_df   =  combined[out_cols].copy()

    # Save per-year pickle
    out_path =  os.path.join(out_dir, f"{level}_3period_hists_{year}.pkl")
    out_df.to_pickle(out_path)
    print(f"  Saved {len(out_df)} rows → {out_path}")
    print(f"  Feature vector length: {len(out_df['hist'].iloc[0])}")

    # Free memory
    del combined, out_df

print(f"\nDone. Pickles saved to {out_dir}/")
