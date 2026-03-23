"""
ee_alpha_earth_cimmyt_binned_hist.py
=====================================
Extract binned histogram AEF features for CIMMYT farmer plots.

For each of 64 embedding dimensions, computes the fraction of cropland
pixels falling into each of 8 fixed-width bins in [-0.8, 0.8].
This gives 64 x 8 = 512 features per plot-year.

Uses the same buffered CIMMYT plot geometries as ee_alpha_earth_cimmyt.py.

Usage:
  conda activate ML_env
  python3 ee_alpha_earth_cimmyt_binned_hist.py
  python3 ee_alpha_earth_cimmyt_binned_hist.py --start_state 14 --start_year 2019

Output: CSV files exported to Google Drive folder 'alpha_earth_cimmyt_plot_binned_hist'

Author: Jay Sayre
Date: 2026-03-21
"""

import os
import sys
import time
import argparse
from ast import literal_eval

import ee
import pandas as pd


# ============================================================
# Configuration
# ============================================================
YEARS        =  range(2017, 2023)
BATCH_SIZE   =  500
SCALE        =  10
N_BINS       =  8
BIN_MIN      =  -0.8
BIN_MAX      =  0.8
DRIVE_FOLDER =  'alpha_earth_cimmyt_plot_binned_hist'
# ============================================================

BIN_WIDTH =  (BIN_MAX - BIN_MIN) / N_BINS


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names =  [f"A{add_zeros(x)}" for x in range(64)]


def build_indicator_image(img):
    """Create indicator bands: for each dim x bin, 1 if pixel in bin, else 0.

    Returns an ee.Image with 64 x N_BINS = 512 bands, each named
    A00_b0, A00_b1, ..., A63_b7.
    """
    bands =  []
    for dim_name in feat_names:
        band =  img.select(dim_name)
        for b in range(N_BINS):
            lower =  BIN_MIN + b * BIN_WIDTH
            upper =  lower + BIN_WIDTH
            if b < N_BINS - 1:
                indicator =  band.gte(lower).And(band.lt(upper))
            else:
                indicator =  band.gte(lower).And(band.lte(upper))
            bands.append(indicator.rename(f"{dim_name}_b{b}"))
    return ee.Image(bands).toFloat()


def build_out_cols():
    """Build output column names."""
    cols =  []
    for dim_name in feat_names:
        for b in range(N_BINS):
            cols.append(f"{dim_name}_b{b}")
    return cols


out_cols   =  build_out_cols()
selectors  =  out_cols + ['plot_id', 'adcid', 'muncode', 'year']


def safe_start(task, desc):
    """Start a GEE export task, waiting if queue is full."""
    for attempt in range(10):
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg =  str(e).lower()
            if 'too many tasks' in msg:
                wait =  300 * (attempt + 1)
                print(f"    Queue full, waiting {wait//60} min... ({desc})",
                      flush=True)
                time.sleep(wait)
            elif 'payload size' in msg:
                print(f"    Payload too large: {desc}", flush=True)
                return False
            else:
                print(f"    ERROR {desc}: {e}", flush=True)
                return False
    print(f"    GAVE UP after 10 retries: {desc}", flush=True)
    return False


def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--start_state', type=str, default=None,
                        help='Resume from this state code')
    parser.add_argument('--start_year', type=int, default=2017,
                        help='Resume from this year')
    args =  parser.parse_args()

    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

    alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    print(f"Binned histogram features: {len(out_cols)} "
          f"({len(feat_names)} dims x {N_BINS} bins)")
    print(f"Bin range: [{BIN_MIN}, {BIN_MAX}], width: {BIN_WIDTH:.4f}")
    print(f"Years: {list(YEARS)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Drive folder: {DRIVE_FOLDER}")

    # Load geometry CSV
    home_dir  =  os.path.expanduser("~")
    data_dir  =  os.path.join(home_dir, "Dropbox", "Projects",
                              "Maize_prediction", "Data")
    geom_file =  os.path.join(data_dir, "CIMMYT",
                              "cimmyt_plot_geometries_for_ee.csv")

    geoms =  pd.read_csv(geom_file)
    geoms['coords']  =  geoms['coords'].apply(literal_eval)
    geoms['plot_id'] =  geoms['plot_id'].astype(str)
    print(f"Loaded {len(geoms):,} CIMMYT plot geometries")

    # Export loop: state -> year -> batch
    states =  sorted(geoms['state'].astype(str).str.zfill(2).unique())
    if args.start_state:
        states =  [s for s in states if s >= args.start_state]
        print(f"Resuming from state {args.start_state} "
              f"({len(states)} states remaining)")

    total_tasks =  0

    for state in states:
        state_geoms =  geoms.loc[
            geoms['state'].astype(str).str.zfill(2) == state
        ].copy()
        print(f"\nState {state}: {len(state_geoms):,} plots")

        # Convert to EE geometries
        state_geoms['eeGeom'] =  state_geoms['coords'].apply(ee.Geometry)

        yr_start =  args.start_year if state == args.start_state else 2017
        for year in YEARS:
            if year < yr_start:
                continue

            # Build EE features for each plot
            ee_features =  []
            for _, row in state_geoms.iterrows():
                feat =  ee.Feature(row['eeGeom'], {
                    'plot_id': str(row['plot_id']),
                    'adcid':   str(row['adcid']) if pd.notna(row['adcid']) else '',
                    'muncode': str(row['muncode']),
                })
                ee_features.append(feat)

            # Get AEF image and apply ag mask
            img    =  alpha_earth.filterDate(
                           f'{year}-01-01', f'{year}-12-31').mean()
            wc     =  ee.ImageCollection("ESA/WorldCover/v200")
            agland =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()
            img    =  img.updateMask(agland)

            # Build indicator image (512 bands)
            indicator_img =  build_indicator_image(img)

            # Batch exports
            n_batches =  (len(ee_features) + BATCH_SIZE - 1) // BATCH_SIZE

            for b in range(n_batches):
                start =  b * BATCH_SIZE
                end   =  min(start + BATCH_SIZE, len(ee_features))
                batch_feats =  ee_features[start:end]

                batch_coll =  ee.FeatureCollection(batch_feats)

                # Reduce with mean() — gives fraction of pixels per bin
                results =  indicator_img.reduceRegions(
                    collection=batch_coll,
                    reducer=ee.Reducer.mean(),
                    scale=SCALE
                )
                results =  results.map(lambda f: f.set('year', year))

                batch_label =  f"_b{b}" if n_batches > 1 else ""
                desc =  f"ae_cimmyt_binhist_{state}_{year}{batch_label}"

                task =  ee.batch.Export.table.toDrive(
                    collection=results,
                    description=desc,
                    folder=DRIVE_FOLDER,
                    selectors=selectors
                )
                if safe_start(task, desc):
                    total_tasks += 1
                    print(f"  {desc} — {len(batch_feats)} plots", flush=True)

    print(f"\nAll exports submitted! Total: {total_tasks} tasks")
    print(f"  -> {DRIVE_FOLDER}/ on Google Drive")
    print(f"  Monitor at: https://code.earthengine.google.com/tasks")
    print(f"\nAfter tasks complete, download CSVs and run:")
    print(f"  python3 consolidate_aef_cimmyt_all.py --type binned_hist")


if __name__ == '__main__':
    main()
