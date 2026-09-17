"""
ls_3period_hists_cimmyt.py — Extract 3-period 2D NDVI histograms for CIMMYT plots.

Creates pixel-level trajectory histograms across 3 phenology-based periods:
  P1: planting_month → +2 months   (early vegetative)
  P2: +2 → +4 months               (peak growth)
  P3: +4 → +6 months               (senescence/harvest)

For each plot geometry, computes 3 period-pair 2D histograms:
  hist_p12: P1 × P2 (early → peak)
  hist_p23: P2 × P3 (peak → senescence)
  hist_p13: P1 × P3 (full season)

Usage:
  python ls_3period_hists_cimmyt.py
  python ls_3period_hists_cimmyt.py 08 2019

Requires: ML_env (earthengine-api)
"""

from ast import literal_eval
import os
import sys
import time
import subprocess

import pandas as pd
import ee

# ── CLI args ─────────────────────────────────────────────
start_state =  sys.argv[1] if len(sys.argv) > 1 else None
start_year  =  int(sys.argv[2]) if len(sys.argv) > 2 else 2017

ee.Initialize()


# ── Config ───────────────────────────────────────────────
bins       =  16
id_col     =  "plot_id"
batch_n    =  500
YEARS      =  range(2017, 2023)
folder     =  "cimmyt_3period_hists_16bins"


# ── Band names by satellite ─────────────────────────────
BAND_MAP =  {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}


# ── Landsat helpers ──────────────────────────────────────

def scale_facts(img):
    opticalBands =  img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands =  img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, overwrite=True).addBands(thermalBands, overwrite=True)


def cloud_mask(img):
    qa   =  img.select(['QA_PIXEL'])
    mask =  ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4))
    )
    return img.updateMask(mask)


def calc_ndvi(img, red, nir):
    img  =  scale_facts(img)
    img  =  cloud_mask(img)
    ndvi =  img.normalizedDifference([nir, red]).rename('ndvi')
    return ndvi


def get_ndvi_composite(geom, start_date, end_date):
    """Merge Landsat 5/7/8 NDVI for a date range, return median composite."""
    collections =  []
    for sat, bands in BAND_MAP.items():
        col =  (ee.ImageCollection(sat)
                .filterBounds(geom)
                .filterDate(start_date, end_date))
        ndvi =  ee.Algorithms.If(
            col.size(),
            col.map(lambda x: calc_ndvi(x, bands['red'], bands['nir'])),
            ee.ImageCollection([])
        )
        collections.append(ee.ImageCollection(ndvi))

    merged =  collections[0].merge(collections[1]).merge(collections[2])
    merged =  ee.ImageCollection(merged)

    merged =  ee.Algorithms.If(
        merged.size(),
        merged,
        ee.ImageCollection([ee.Image.constant(0).rename('ndvi')])
    )
    return ee.ImageCollection(merged).median()


# ── Binning helpers ──────────────────────────────────────

NDVI_MIN =  0.0
NDVI_MAX =  1.0


def assign_bin(img, vmin, vmax, n_bins):
    """Assign each pixel to a bin index [0, n_bins-1]."""
    eps =  (vmax - vmin) / 1_000_000
    img =  img.max(vmin + eps)
    img =  img.min(vmax - eps)
    img =  img.subtract(vmin)
    return img.divide((vmax - vmin) / n_bins).floor()


def calc_1d_bin(bin_a, bin_b, n_bins):
    """Flatten 2D bin indices to 1D: bin_1d = bin_a + bin_b * n_bins."""
    return bin_a.add(bin_b.multiply(n_bins)).rename('hist')


# ── Main histogram function ─────────────────────────────

def compute_3period_hist(feature):
    """For a single feature, compute 3 period-pair 2D histograms."""
    geom           =  feature.geometry()
    planting_month =  ee.Number(feature.get('planting_month'))

    gs_year  =  ee.Number(feature.get('gs_year'))
    gs_start =  ee.Date.fromYMD(gs_year, planting_month, 1)

    p1_start =  gs_start
    p1_end   =  gs_start.advance(2, 'month')
    p2_start =  p1_end
    p2_end   =  gs_start.advance(4, 'month')
    p3_start =  p2_end
    p3_end   =  gs_start.advance(6, 'month')

    ndvi_p1 =  get_ndvi_composite(geom, p1_start, p1_end)
    ndvi_p2 =  get_ndvi_composite(geom, p2_start, p2_end)
    ndvi_p3 =  get_ndvi_composite(geom, p3_start, p3_end)

    bin_p1 =  assign_bin(ndvi_p1.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)
    bin_p2 =  assign_bin(ndvi_p2.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)
    bin_p3 =  assign_bin(ndvi_p3.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)

    hist_1d_p12 =  calc_1d_bin(bin_p1, bin_p2, bins).rename('hist_p12')
    hist_1d_p23 =  calc_1d_bin(bin_p2, bin_p3, bins).rename('hist_p23')
    hist_1d_p13 =  calc_1d_bin(bin_p1, bin_p3, bins).rename('hist_p13')

    combined =  hist_1d_p12.addBands(hist_1d_p23).addBands(hist_1d_p13)

    n_bins_sq =  bins * bins
    hist =  combined.reduceRegion(
        ee.Reducer.fixedHistogram(0, n_bins_sq, n_bins_sq),
        geom,
        30,
        maxPixels=500_000_000
    )

    return feature.set(hist)


# ── Queue management ────────────────────────────────────

QUEUE_LIMIT   =  3000
QUEUE_BUFFER  =  50
POLL_INTERVAL =  60


def count_active_tasks():
    """Count READY/RUNNING/PENDING tasks via earthengine CLI."""
    result =  subprocess.run(
        ['earthengine', 'task', 'list'],
        capture_output=True, text=True, timeout=120
    )
    n_active =  0
    for line in result.stdout.splitlines():
        if 'PENDING' in line or 'RUNNING' in line or 'READY' in line:
            n_active += 1
    return n_active


def wait_for_queue_space():
    """Block until GEE task queue has room for more tasks."""
    while True:
        try:
            n_active =  count_active_tasks()
            if n_active < QUEUE_LIMIT - QUEUE_BUFFER:
                print(f"    Queue has space ({n_active}/{QUEUE_LIMIT}), resuming...")
                return
            print(f"    Queue full ({n_active}/{QUEUE_LIMIT}), waiting {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"    Queue check error: {e}, retrying in {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)


_submit_count =  0


def submit_batch(batch_df, state, batch_idx, year):
    """Submit a single batch to GEE, retrying with smaller sub-batches on payload error."""
    global _submit_count
    _submit_count += 1

    if _submit_count % 100 == 0:
        wait_for_queue_space()

    ee_coll =  ee.FeatureCollection(batch_df['eeFeature'].to_list())

    print(f"  State {state}, batch {batch_idx}, year {year} — {len(batch_df)} plots")
    result =  ee_coll.map(compute_3period_hist)

    desc =  f"3period_hist_cimmyt_{state}_{batch_idx}_{year}"
    cols =  [id_col, 'hist_p12', 'hist_p23', 'hist_p13']

    try:
        task =  ee.batch.Export.table.toDrive(
            collection=result, folder=folder,
            description=desc, selectors=cols
        )
        task.start()
    except ee.ee_exception.EEException as e:
        err_msg =  str(e).lower()
        if 'payload size' in err_msg or 'exceeds the limit' in err_msg:
            mid =  len(batch_df) // 2
            if mid == 0:
                print(f"    ERROR: single feature too large, skipping")
                return
            print(f"    Payload too large, splitting into sub-batches...")
            submit_batch(batch_df.iloc[:mid], state, f"{batch_idx}a", year)
            submit_batch(batch_df.iloc[mid:], state, f"{batch_idx}b", year)
        elif 'too many tasks' in err_msg:
            print(f"    Queue full, waiting for space...")
            wait_for_queue_space()
            submit_batch(batch_df, state, batch_idx, year)
        else:
            raise


# ── Load geometries ──────────────────────────────────────

home_dir  =  os.path.expanduser("~")
data_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
geom_file =  os.path.join(data_dir, "CIMMYT", "cimmyt_plot_geometries_for_ee.csv")

geoms =  pd.read_csv(geom_file)
geoms['coords']  =  geoms['coords'].apply(literal_eval)
geoms['plot_id'] =  geoms['plot_id'].astype(str)
geoms['state']   =  geoms['state'].astype(str).str.zfill(2)
print(f"Loaded {len(geoms):,} CIMMYT plot geometries")


# ── Export loop: state × year × batch ────────────────────

states =  sorted(geoms['state'].unique())
if start_state:
    states =  [s for s in states if s >= start_state]
    print(f"Resuming from state {start_state} ({len(states)} states remaining)")

for state in states:
    state_geoms =  geoms.loc[geoms['state'] == state].copy()
    print(f"\nState {state}: {len(state_geoms):,} plots")

    # Convert to EE geometries
    state_geoms['eeGeom'] =  state_geoms['coords'].apply(ee.Geometry)

    yr_start =  start_year if state == start_state else 2017
    for year in YEARS:
        if year < yr_start:
            continue

        # Add gs_year as a property on each feature
        state_geoms['eeFeature'] =  state_geoms.apply(
            lambda row: ee.Feature(row['eeGeom'], {
                id_col:           str(row[id_col]),
                'planting_month': int(row['planting_month']),
                'gs_year':        year,
            }),
            axis=1
        )

        n_total   =  len(state_geoms)
        n_batches =  (n_total + batch_n - 1) // batch_n

        for b in range(n_batches):
            batch =  state_geoms.iloc[b * batch_n : (b + 1) * batch_n]
            submit_batch(batch, state, b, year)

print(f"\nAll tasks submitted. Outputs → Google Drive/{folder}/")
