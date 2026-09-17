"""
ls_3period_hists.py — Extract 3-period 2D NDVI histograms from Google Earth Engine.

Creates pixel-level trajectory histograms across 3 phenology-based periods:
  P1: planting_month → +2 months   (early vegetative)
  P2: +2 → +4 months               (peak growth)
  P3: +4 → +6 months               (senescence/harvest)

For each geometry, computes 3 period-pair 2D histograms:
  hist_p12: P1 × P2 (early → peak)
  hist_p23: P2 × P3 (peak → senescence)
  hist_p13: P1 × P3 (full season)

Each 2D histogram is flattened to bins² elements via:
  bin_1d = bin_p1 + bin_p2 * bins

Usage:
  python ls_3period_hists.py <level> <bins>
  python ls_3period_hists.py muni 16
  python ls_3period_hists.py adc 16

Requires: ML_env (earthengine-api)
"""

from ast import literal_eval
import os
import sys
import time

import pandas as pd
import ee

level       =  sys.argv[1]   # "muni" or "adc"
bins        =  int(sys.argv[2])  # bins per axis (e.g. 16)
start_state =  sys.argv[3] if len(sys.argv) > 3 else None  # optional: resume from this state
start_year  =  int(sys.argv[4]) if len(sys.argv) > 4 else 2003  # optional: resume from this year

ee.Initialize()

# ── Band names by satellite ──────────────────────────────
BAND_MAP =  {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}


# ── Landsat helpers (same as ls_ndvi_hists.py) ───────────

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

    # Fallback to constant zero if no images
    merged =  ee.Algorithms.If(
        merged.size(),
        merged,
        ee.ImageCollection([ee.Image.constant(0).rename('ndvi')])
    )
    return ee.ImageCollection(merged).median()


# ── Binning helpers (same as ls_ndvi_hists.py) ───────────

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


# ── Main histogram function (mapped over FeatureCollection) ──

def compute_3period_hist(feature):
    """For a single feature, compute 3 period-pair 2D histograms."""
    geom           =  feature.geometry()
    planting_month =  ee.Number(feature.get('planting_month'))

    # Compute growing season start date for this feature's year
    gs_year  =  ee.Number(feature.get('gs_year'))
    gs_start =  ee.Date.fromYMD(gs_year, planting_month, 1)

    # Period boundaries (each 2 months)
    p1_start =  gs_start
    p1_end   =  gs_start.advance(2, 'month')
    p2_start =  p1_end
    p2_end   =  gs_start.advance(4, 'month')
    p3_start =  p2_end
    p3_end   =  gs_start.advance(6, 'month')

    # NDVI median composites per period
    ndvi_p1 =  get_ndvi_composite(geom, p1_start, p1_end)
    ndvi_p2 =  get_ndvi_composite(geom, p2_start, p2_end)
    ndvi_p3 =  get_ndvi_composite(geom, p3_start, p3_end)

    # Bin each period
    bin_p1 =  assign_bin(ndvi_p1.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)
    bin_p2 =  assign_bin(ndvi_p2.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)
    bin_p3 =  assign_bin(ndvi_p3.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)

    # 2D → 1D encoding for each pair
    hist_1d_p12 =  calc_1d_bin(bin_p1, bin_p2, bins).rename('hist_p12')
    hist_1d_p23 =  calc_1d_bin(bin_p2, bin_p3, bins).rename('hist_p23')
    hist_1d_p13 =  calc_1d_bin(bin_p1, bin_p3, bins).rename('hist_p13')

    combined =  hist_1d_p12.addBands(hist_1d_p23).addBands(hist_1d_p13)

    # Reduce to fixed histograms
    n_bins_sq =  bins * bins
    hist =  combined.reduceRegion(
        ee.Reducer.fixedHistogram(0, n_bins_sq, n_bins_sq),
        geom,
        30,
        maxPixels=500_000_000
    )

    return feature.set(hist)


# ── Load geometries ──────────────────────────────────────

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")

if level == "muni":
    geom_file =  os.path.join(data_dir, "muni_geometries_for_ee.csv")
    id_col    =  "muncode"
    batch_n   =  25
elif level == "adc":
    geom_file =  os.path.join(data_dir, "adc_geometries_for_ee.csv")
    id_col    =  "adcid"
    batch_n   =  500
else:
    print(f"Unknown level: {level}. Use 'muni' or 'adc'.")
    sys.exit(1)

geoms =  pd.read_csv(geom_file)
geoms['coords'] =  geoms['coords'].apply(literal_eval)
print(f"Loaded {len(geoms)} {level}-level geometries")

# Add state code for batching
if level == "muni":
    geoms['state'] =  geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2])
else:
    geoms['state'] =  geoms['muncode'].apply(lambda x: str(x)[:2])  # adcid already string

# ── Export folder ────────────────────────────────────────
folder =  f"{level}_3period_hists_{bins}bins"

QUEUE_LIMIT   =  3000
QUEUE_BUFFER  =  50     # resume when queue has this many free slots
POLL_INTERVAL =  60     # seconds between queue checks

def count_active_tasks():
    """Count READY/RUNNING/PENDING tasks via earthengine CLI (fast)."""
    import subprocess
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

def submit_batch(batch_df, state, batch_idx, year, level, id_col, folder, bins):
    """Submit a single batch to GEE, retrying with smaller sub-batches on payload error."""
    global _submit_count
    _submit_count += 1

    # Check queue every 100 submissions
    if _submit_count % 100 == 0:
        wait_for_queue_space()

    ee_coll =  ee.FeatureCollection(batch_df['eeFeature'].to_list())

    print(f"  State {state}, batch {batch_idx}, year {year} — {len(batch_df)} {level}s")
    result =  ee_coll.map(compute_3period_hist)

    desc =  f"3period_hist_{level}_{state}_{batch_idx}_{year}"
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
            # Split batch in half and retry
            mid =  len(batch_df) // 2
            if mid == 0:
                print(f"    ERROR: single feature too large, skipping")
                return
            print(f"    Payload too large, splitting into sub-batches...")
            submit_batch(batch_df.iloc[:mid], state, f"{batch_idx}a", year, level, id_col, folder, bins)
            submit_batch(batch_df.iloc[mid:], state, f"{batch_idx}b", year, level, id_col, folder, bins)
        elif 'too many tasks' in err_msg:
            print(f"    Queue full, waiting for space...")
            wait_for_queue_space()
            submit_batch(batch_df, state, batch_idx, year, level, id_col, folder, bins)
        else:
            raise


# ── Export loop: state × year × batch ────────────────────
states =  sorted(geoms['state'].unique())
if start_state:
    states =  [s for s in states if s >= start_state]
    print(f"Resuming from state {start_state} ({len(states)} states remaining)")

for state in states:
    state_geoms =  geoms.loc[geoms['state'] == state].copy()
    print(f"\nState {state}: {len(state_geoms)} {level}s")

    # Convert to EE geometries and features
    state_geoms['eeGeom'] =  state_geoms['coords'].apply(ee.Geometry)

    yr_start =  start_year if state == start_state else 2003
    for year in range(yr_start, 2025):
        # Add gs_year as a property on each feature
        state_geoms['eeFeature'] =  state_geoms.apply(
            lambda row: ee.Feature(row['eeGeom'], {
                id_col:          row[id_col],
                'planting_month': int(row['planting_month']),
                'gs_year':        year,
            }),
            axis=1
        )

        n_total   =  len(state_geoms)
        n_batches =  (n_total + batch_n - 1) // batch_n

        for b in range(n_batches):
            batch =  state_geoms.iloc[b * batch_n : (b + 1) * batch_n]
            submit_batch(batch, state, b, year, level, id_col, folder, bins)

print(f"\nAll tasks submitted. Outputs → Google Drive/{folder}/")
