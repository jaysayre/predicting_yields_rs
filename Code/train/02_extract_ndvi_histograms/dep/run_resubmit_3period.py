"""
run_resubmit_3period.py — Manifest-driven targeted re-pull of 3-period NDVI
histograms (uniform OR quantile bins), for the coverage-boost handoff.

Only submits the (state, year) batches listed in resubmit_manifest.csv for
source == '3period', using the 32-state adc_geometries_for_ee.csv. Exports to
Google Drive (personal OAuth token, project avocadoyieldsdeforestation); pull
with the rclone 'gdrive' remote afterwards. Avoids the GCS/service-account
detour in the handoff.

Usage (laptop):
  ~/miniforge3/envs/geo_env/bin/python run_resubmit_3period.py [--quantile] [--limit N] [--bins 16]

  --quantile : use per-bin quantile edges from quantile_bin_edges_<bins>.json
               (folder adc_3period_hists_<bins>bins_quantile) instead of uniform
  --limit N  : submit only the first N batches (smoke test)
"""
import os
import sys
import json
import time
import argparse
from ast import literal_eval

import ee
import pandas as pd

parser =  argparse.ArgumentParser()
parser.add_argument('--quantile', action='store_true')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--bins', type=int, default=16)
parser.add_argument('--all_states', action='store_true',
                    help='cover all 32 states, not just the coverage-gap manifest '
                         '(quantile is new everywhere, so needs all states)')
parser.add_argument('--year', type=int, default=None,
                    help='restrict to a single year (ADC eval only needs 2022)')
args =  parser.parse_args()

bins =  args.bins

try:
    ee.Initialize(project='avocadoyieldsdeforestation')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

# Resume-safe: skip any task description that is already active or done on GEE
# (PENDING/RUNNING/SUCCEEDED). Cancelled/failed are NOT skipped so they re-run.
# NOTE: EE operation metadata does not expose the Drive folder, so uniform and
# quantile tasks must carry DISTINCT descriptions (quantile gets a '_q' suffix)
# or they would alias each other here.
ACTIVE_STATES =  {'PENDING', 'RUNNING', 'SUCCEEDED', 'COMPLETED'}
def existing_descriptions():
    seen =  set()
    try:
        for o in ee.data.listOperations():
            md =  o.get('metadata', {})
            if md.get('state') in ACTIVE_STATES and md.get('description'):
                seen.add(md['description'])
    except Exception as e:
        print(f"  (could not list existing tasks: {e})")
    return seen

EXISTING =  existing_descriptions()
DESC_SUFFIX =  "_q" if args.quantile else ""
print(f"  {len(EXISTING)} active/done tasks on GEE; will skip matching descriptions.")

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")

BAND_MAP =  {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}
NDVI_MIN, NDVI_MAX =  0.0, 1.0


def scale_facts(img):
    ob =  img.select('SR_B.').multiply(0.0000275).add(-0.2)
    tb =  img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(ob, overwrite=True).addBands(tb, overwrite=True)

def cloud_mask(img):
    qa =  img.select(['QA_PIXEL'])
    mask =  ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4)))
    return img.updateMask(mask)

def calc_ndvi(img, red, nir):
    img =  cloud_mask(scale_facts(img))
    return img.normalizedDifference([nir, red]).rename('ndvi')

def get_ndvi_composite(geom, start_date, end_date):
    cols =  []
    for sat, b in BAND_MAP.items():
        col =  ee.ImageCollection(sat).filterBounds(geom).filterDate(start_date, end_date)
        ndvi =  ee.Algorithms.If(col.size(),
                                 col.map(lambda x: calc_ndvi(x, b['red'], b['nir'])),
                                 ee.ImageCollection([]))
        cols.append(ee.ImageCollection(ndvi))
    merged =  ee.ImageCollection(cols[0].merge(cols[1]).merge(cols[2]))
    merged =  ee.Algorithms.If(merged.size(), merged,
                               ee.ImageCollection([ee.Image.constant(0).rename('ndvi')]))
    return ee.ImageCollection(merged).median()


# ── Binning ──────────────────────────────────────────────
if args.quantile:
    edges_file =  os.path.join(data_dir, f"quantile_bin_edges_{bins}.json")
    with open(edges_file) as f:
        BIN_EDGES =  json.load(f)
    assert len(BIN_EDGES) == bins + 1, f"expected {bins+1} edges, got {len(BIN_EDGES)}"
    print(f"Quantile edges: {BIN_EDGES}")

    def assign_bin(img):
        b =  ee.Image.constant(0)
        for i in range(1, len(BIN_EDGES) - 1):
            b =  b.where(img.gte(BIN_EDGES[i]), i)
        return b
    folder =  f"adc_3period_hists_{bins}bins_quantile"
else:
    def assign_bin(img):
        eps =  (NDVI_MAX - NDVI_MIN) / 1_000_000
        im =  img.max(NDVI_MIN + eps).min(NDVI_MAX - eps).subtract(NDVI_MIN)
        return im.divide((NDVI_MAX - NDVI_MIN) / bins).floor()
    folder =  f"adc_3period_hists_{bins}bins"


def calc_1d_bin(a, b):
    return a.add(b.multiply(bins)).rename('hist')

def compute_3period_hist(feature):
    geom =  feature.geometry()
    pm   =  ee.Number(feature.get('planting_month'))
    gsy  =  ee.Number(feature.get('gs_year'))
    gs0  =  ee.Date.fromYMD(gsy, pm, 1)
    n1 =  get_ndvi_composite(geom, gs0, gs0.advance(2, 'month'))
    n2 =  get_ndvi_composite(geom, gs0.advance(2, 'month'), gs0.advance(4, 'month'))
    n3 =  get_ndvi_composite(geom, gs0.advance(4, 'month'), gs0.advance(6, 'month'))
    b1, b2, b3 =  assign_bin(n1.select('ndvi')), assign_bin(n2.select('ndvi')), assign_bin(n3.select('ndvi'))
    combined =  (calc_1d_bin(b1, b2).rename('hist_p12')
                 .addBands(calc_1d_bin(b2, b3).rename('hist_p23'))
                 .addBands(calc_1d_bin(b1, b3).rename('hist_p13')))
    nsq =  bins * bins
    hist =  combined.reduceRegion(ee.Reducer.fixedHistogram(0, nsq, nsq), geom, 30,
                                  maxPixels=500_000_000)
    return feature.set(hist)


# ── Load manifest + geometries ───────────────────────────
geoms =  pd.read_csv(os.path.join(data_dir, "adc_geometries_for_ee.csv"))
geoms['coords'] =  geoms['coords'].apply(literal_eval)
geoms['state']  =  geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2])

years_wanted =  [args.year] if args.year else list(range(2017, 2025))
if args.all_states:
    # quantile is a new binning everywhere -> every state needs it
    target_states =  sorted(geoms['state'].unique())
else:
    man =  pd.read_csv(os.path.join(data_dir, "resubmit_manifest.csv"), dtype={'state': str})
    man =  man[man['source'] == '3period']
    man['state'] =  man['state'].str.zfill(2)
    target_states =  sorted(man['state'].unique())
want =  set((s, y) for s in target_states for y in years_wanted)
geoms =  geoms[geoms['state'].isin(target_states)].copy()
geoms['eeGeom'] =  geoms['coords'].apply(ee.Geometry)
print(f"{len(geoms):,} ADCs across {geoms['state'].nunique()} manifest states; "
      f"folder={folder}")

batch_n =  500
submitted =  0
for state in sorted(geoms['state'].unique()):
    sg =  geoms[geoms['state'] == state]
    for year in range(2017, 2025):
        if (state, year) not in want:
            continue
        sg2 =  sg.copy()
        sg2['eeFeature'] =  sg2.apply(lambda r: ee.Feature(r['eeGeom'], {
            'adcid': r['adcid'], 'planting_month': int(r['planting_month']),
            'gs_year': year}), axis=1)
        nb =  (len(sg2) + batch_n - 1) // batch_n
        for b in range(nb):
            batch =  sg2.iloc[b * batch_n:(b + 1) * batch_n]
            coll =  ee.FeatureCollection(batch['eeFeature'].to_list())
            desc =  f"3period_hist_adc_{state}_{b}_{year}{DESC_SUFFIX}"
            if desc in EXISTING:
                continue
            res  =  coll.map(compute_3period_hist)
            task =  ee.batch.Export.table.toDrive(
                collection=res, folder=folder, description=desc,
                selectors=['adcid', 'hist_p12', 'hist_p23', 'hist_p13'])
            for attempt in range(8):
                try:
                    task.start(); break
                except ee.ee_exception.EEException as e:
                    if 'too many tasks' in str(e).lower():
                        print("  queue full, waiting 300s..."); time.sleep(300)
                    else:
                        raise
            submitted +=  1
            print(f"  submitted {desc} ({len(batch)} ADCs) [{submitted}]", flush=True)
            if args.limit and submitted >= args.limit:
                print(f"\nLimit {args.limit} reached (smoke test). Folder: {folder}")
                sys.exit(0)

print(f"\nAll {submitted} tasks submitted → Drive/{folder}/")
