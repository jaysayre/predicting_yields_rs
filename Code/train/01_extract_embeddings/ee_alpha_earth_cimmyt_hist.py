"""
ee_alpha_earth_cimmyt_hist.py
=============================
Extract distributional AEF embedding features (percentiles + stdDev)
for CIMMYT farmer plots.

For each of 64 embedding dimensions, extracts percentiles [10, 25, 50, 75, 90]
and stdDev across agricultural pixels within each plot polygon.
This gives 64 x 6 = 384 features per plot-year.

Uses the same buffered CIMMYT plot geometries as ee_alpha_earth_cimmyt.py.

Usage:
  conda activate ML_env
  python3 ee_alpha_earth_cimmyt_hist.py
  python3 ee_alpha_earth_cimmyt_hist.py --start_state 14 --start_year 2019

Output: CSV files exported to Google Drive folder 'alpha_earth_cimmyt_plot_hist'

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
DRIVE_FOLDER =  'alpha_earth_cimmyt_plot_hist'
# ============================================================


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names =  [f"A{add_zeros(x)}" for x in range(64)]


# Build output column names: percentiles + stdDev per dimension
pct_suffixes =  ['_p10', '_p25', '_p50', '_p75', '_p90']
std_suffix   =  '_stdDev'
out_cols     =  []
for feat in feat_names:
    for suf in pct_suffixes:
        out_cols.append(feat + suf)
    out_cols.append(feat + std_suffix)

selectors =  out_cols + ['plot_id', 'adcid', 'muncode', 'year']


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

    # Build reducer: percentiles + stdDev
    pct_reducer =  ee.Reducer.percentile([10, 25, 50, 75, 90])
    std_reducer =  ee.Reducer.stdDev()
    reducer     =  pct_reducer.combine(std_reducer, sharedInputs=True)

    print(f"Output features: {len(out_cols)} ({len(feat_names)} dims x 6 stats)")
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

            # Batch exports
            n_batches =  (len(ee_features) + BATCH_SIZE - 1) // BATCH_SIZE

            for b in range(n_batches):
                start =  b * BATCH_SIZE
                end   =  min(start + BATCH_SIZE, len(ee_features))
                batch_feats =  ee_features[start:end]

                batch_coll =  ee.FeatureCollection(batch_feats)

                # reduceRegions with percentile + stdDev reducer
                results =  img.reduceRegions(
                    collection=batch_coll,
                    reducer=reducer,
                    scale=SCALE
                )
                results =  results.map(lambda f: f.set('year', year))

                batch_label =  f"_b{b}" if n_batches > 1 else ""
                desc =  f"ae_cimmyt_hist_{state}_{year}{batch_label}"

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
    print(f"  python3 consolidate_aef_cimmyt_all.py --type hist")


if __name__ == '__main__':
    main()
