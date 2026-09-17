"""
11_ee_alpha_earth_cimmyt.py — Extract AEF satellite embeddings for CIMMYT plots.

Reads buffered CIMMYT plot geometries and extracts mean AEF embeddings
(64 features, A00-A63) per plot per year using Google Earth Engine.
Uses ESA WorldCover v200 cropland mask (class 40).

Usage:
  python 11_ee_alpha_earth_cimmyt.py
  python 11_ee_alpha_earth_cimmyt.py --start_state 14 --start_year 2019

Requires: ML_env (earthengine-api, pandas)
"""

import os
import sys
import time
import argparse
from ast import literal_eval

import ee
import pandas as pd


# ── CLI arguments ────────────────────────────────────────
parser =  argparse.ArgumentParser()
parser.add_argument('--start_state', type=str, default=None,
                    help='Resume from this state code')
parser.add_argument('--start_year', type=int, default=2017,
                    help='Resume from this year')
args =  parser.parse_args()


# ── EE setup ─────────────────────────────────────────────
ee.Authenticate()
ee.Initialize(project='avocadoyieldsdeforestation')


# ── Config ───────────────────────────────────────────────
YEARS      =  range(2017, 2023)
BATCH_SIZE =  500
DRIVE_FOLDER =  'alpha_earth_cimmyt_plot'


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names  =  [f"A{add_zeros(x)}" for x in range(0, 64)]
alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
cols        =  feat_names + ['plot_id', 'adcid', 'muncode', 'year']


def safe_start(task, desc):
    """Start a GEE export task, waiting if queue is full."""
    while True:
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg =  str(e).lower()
            if 'too many tasks' in msg:
                print(f"    Queue full, waiting 5 min... ({desc})", flush=True)
                time.sleep(300)
            elif 'payload size' in msg:
                print(f"    Payload too large: {desc}", flush=True)
                return False
            else:
                print(f"    ERROR {desc}: {e}", flush=True)
                return False


# ── Load geometry CSV ────────────────────────────────────
home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
geom_file =  os.path.join(data_dir, "CIMMYT", "cimmyt_plot_geometries_for_ee.csv")

geoms =  pd.read_csv(geom_file)
geoms['coords'] =  geoms['coords'].apply(literal_eval)
geoms['plot_id'] =  geoms['plot_id'].astype(str)
print(f"Loaded {len(geoms):,} CIMMYT plot geometries")


# ── Export loop: state → year → batch ────────────────────
states =  sorted(geoms['state'].astype(str).str.zfill(2).unique())
if args.start_state:
    states =  [s for s in states if s >= args.start_state]
    print(f"Resuming from state {args.start_state} ({len(states)} states remaining)")

total_tasks =  0

for state in states:
    state_geoms =  geoms.loc[geoms['state'].astype(str).str.zfill(2) == state].copy()
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
        img    =  alpha_earth.filterDate(f'{year}-01-01', f'{year}-12-31').mean()
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
            means =  img.reduceRegions(
                collection=batch_coll,
                reducer=ee.Reducer.mean(),
                scale=10
            )
            means =  means.map(lambda f: f.set('year', year))

            batch_label =  f"_b{b}" if n_batches > 1 else ""
            desc =  f"ae_cimmyt_{state}_{year}{batch_label}"

            task =  ee.batch.Export.table.toDrive(
                collection=means,
                description=desc,
                folder=DRIVE_FOLDER,
                selectors=cols
            )
            if safe_start(task, desc):
                total_tasks += 1
                print(f"  {desc} — {len(batch_feats)} plots", flush=True)

print(f"\nAll exports submitted! Total: {total_tasks} tasks")
print(f"  -> {DRIVE_FOLDER}/ on Google Drive")
