"""
clean_monthly_histograms.py — Preprocess monthly histogram CSVs from GEE
into training-ready pickle files.

GEE exports from ls_monthly_hists.py produce CSVs with columns:
  muncode, {index}_m{month} for each index × month combination
Each histogram column contains a string of format '[[bin_start, count], ...]'

This script:
  1. Reads all exported CSVs from the GEE Drive folder
  2. Parses histogram strings into numpy arrays
  3. Normalizes histograms (counts → proportions)
  4. Reshapes into (bins, n_months, n_indices) tensors
  5. Saves per-year pickle files for training

Usage:
  python clean_monthly_histograms.py <raw_csv_dir> <output_dir> <bins> <n_months>
  python clean_monthly_histograms.py ../Data/muni_monthly_hists_32bins_8mo ../Data/monthly_hists_clean 32 8

Run from Maize_prediction/ directory.
"""

import os
import sys
import re
import pickle

import numpy as np
import pandas as pd

raw_dir   =  sys.argv[1]  # Directory with GEE-exported CSVs
out_dir   =  sys.argv[2]  # Output directory for pickles
bins      =  int(sys.argv[3])
n_months  =  int(sys.argv[4])

INDEX_NAMES =  ['ndvi', 'gcvi', 'ndti', 'evi', 'ndwi', 'bsi']
n_indices   =  len(INDEX_NAMES)

os.makedirs(out_dir, exist_ok=True)


def parse_hist_cell(cell):
    """
    Parse a GEE histogram string into a numpy array of bin counts.
    Input:  '[[0.0, 15.0], [1.0, 23.0], ...]'
    Output: np.array([15.0, 23.0, ...]) normalized to sum to 1
    """
    if pd.isna(cell) or cell == '' or cell is None:
        return None

    cell =  str(cell)
    # Strip outer brackets and split
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


def build_tensor(row, bins, n_months, index_names):
    """
    Build a (bins, n_months, n_indices) tensor from a DataFrame row.
    Each column {index}_m{month} contains a parsed histogram array.
    """
    tensor =  np.zeros((bins, n_months, len(index_names)))

    for idx_i, idx_name in enumerate(index_names):
        for m in range(n_months):
            col =  f"{idx_name}_m{m}"
            hist =  row.get(col)
            if hist is not None and isinstance(hist, np.ndarray) and len(hist) == bins:
                tensor[:, m, idx_i] =  hist
    return tensor


# ── Read all CSVs ─────────────────────────────────────────

csv_files =  [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files in {raw_dir}")

all_dfs =  []
for f in sorted(csv_files):
    # Extract year from filename: monthly_hist_{ent}_{group}_{year}.csv
    match =  re.search(r'_(\d{4})\.csv$', f)
    if match:
        year =  int(match.group(1))
    else:
        print(f"  Skipping {f} — can't parse year")
        continue

    df      =  pd.read_csv(os.path.join(raw_dir, f))
    df['year'] =  year
    all_dfs.append(df)

if len(all_dfs) == 0:
    print("No CSV files processed. Exiting.")
    sys.exit(1)

combined =  pd.concat(all_dfs, ignore_index=True)
print(f"Combined shape: {combined.shape}")

# ── Parse histogram columns ──────────────────────────────

hist_cols =  [f"{idx}_m{m}" for m in range(n_months) for idx in INDEX_NAMES]

for col in hist_cols:
    if col in combined.columns:
        combined[col] =  combined[col].apply(parse_hist_cell)
    else:
        print(f"  Warning: column {col} not found, filling with zeros")
        combined[col] =  combined.apply(lambda x: np.zeros(bins), axis=1)

# Drop rows with any missing histograms
n_before =  len(combined)
for col in hist_cols:
    combined =  combined[combined[col].apply(lambda x: x is not None and len(x) == bins)]
combined =  combined.reset_index(drop=True)
print(f"Dropped {n_before - len(combined)} rows with bad histograms")

# ── Build tensors ─────────────────────────────────────────

combined['hist_tensor'] =  combined.apply(
    lambda row: build_tensor(row, bins, n_months, INDEX_NAMES),
    axis=1
)

# ── Also extract percentile summaries ─────────────────────
# From each histogram, compute p10, p25, p50, p75, p90

INDEX_RANGES =  {
    'ndvi': (0.0, 1.0),
    'gcvi': (0.0, 12.0),
    'ndti': (0.0, 0.6),
    'evi':  (0.0, 1.0),
    'ndwi': (-0.5, 0.5),
    'bsi':  (-0.5, 0.5),
}

PERCENTILES =  [10, 25, 50, 75, 90]


def hist_percentile(counts, vmin, vmax, percentile):
    """Compute a percentile from histogram bin counts (assumed normalized)."""
    n_bins   =  len(counts)
    bin_size =  (vmax - vmin) / n_bins
    cumsum   =  np.cumsum(counts)
    target   =  percentile / 100.0
    idx      =  np.searchsorted(cumsum, target)
    idx      =  min(idx, n_bins - 1)
    return vmin + (idx + 0.5) * bin_size


def extract_percentiles(row, bins, n_months, index_names, index_ranges, percentiles):
    """Extract percentile features: shape (n_months, n_indices, n_percentiles)."""
    feats =  np.zeros((n_months, len(index_names), len(percentiles)))
    for idx_i, idx_name in enumerate(index_names):
        vmin, vmax =  index_ranges[idx_name]
        for m in range(n_months):
            col   =  f"{idx_name}_m{m}"
            hist  =  row.get(col)
            if hist is not None and isinstance(hist, np.ndarray):
                for p_i, p in enumerate(percentiles):
                    feats[m, idx_i, p_i] =  hist_percentile(hist, vmin, vmax, p)
    return feats


combined['percentiles'] =  combined.apply(
    lambda row: extract_percentiles(row, bins, n_months, INDEX_NAMES, INDEX_RANGES, PERCENTILES),
    axis=1
)

# ── Save per-year pickles ─────────────────────────────────

out_cols =  ['muncode', 'year', 'hist_tensor', 'percentiles']

for year in sorted(combined['year'].unique()):
    year_df =  combined.loc[combined['year'] == year, out_cols].copy()
    out_path =  os.path.join(out_dir, f"monthly_hists_{year}.pkl")
    year_df.to_pickle(out_path)
    print(f"  Year {year}: {len(year_df)} municipalities → {out_path}")

# Also save one combined file
combined[out_cols].to_pickle(os.path.join(out_dir, "monthly_hists_all.pkl"))
print(f"\nAll years combined: {len(combined)} rows → {out_dir}/monthly_hists_all.pkl")
print(f"Tensor shape per row: ({bins}, {n_months}, {n_indices})")
print(f"Percentile shape per row: ({n_months}, {n_indices}, {len(PERCENTILES)})")
