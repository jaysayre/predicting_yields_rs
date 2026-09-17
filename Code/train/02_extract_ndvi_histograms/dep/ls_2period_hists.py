"""
ls_2period_hists.py — 2-period NDVI histogram extraction (planting-anchored).

Model 1 of the Landsat NDVI-histogram family. NDVI only, TWO 3-month phenology
periods anchored to the planting month (P1: planting -> +3mo, P2: +3 -> +6mo),
forming a single P1 x P2 2D trajectory histogram (bins^2 features). Deliberately
parallel to ls_3period_hists.py so the two models differ only in temporal
granularity, and supports both uniform and quantile (per-NDVI-quantile) binning
to mirror the AEF fixed/quantile harmonization.

Exports to Google Drive (personal OAuth, project avocadoyieldsdeforestation);
pull with the rclone 'gdrive' remote.

Usage (laptop):
  ~/miniforge3/envs/geo_env/bin/python ls_2period_hists.py --level adc  --years 2022 [--quantile] [--limit N]
  ~/miniforge3/envs/geo_env/bin/python ls_2period_hists.py --level muni --years 2017-2024 [--quantile]
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
parser.add_argument('--level', choices=['muni', 'adc'], required=True)
parser.add_argument('--years', type=str, default='2022', help='e.g. 2022 or 2017-2024')
parser.add_argument('--quantile', action='store_true')
parser.add_argument('--bins', type=int, default=16)
parser.add_argument('--start_state', type=str, default='01')
parser.add_argument('--limit', type=int, default=None)
args =  parser.parse_args()

bins =  args.bins
if '-' in args.years:
    a, b =  args.years.split('-'); years =  list(range(int(a), int(b) + 1))
else:
    years =  [int(args.years)]

try:
    ee.Initialize(project='avocadoyieldsdeforestation')
except Exception:
    ee.Authenticate(); ee.Initialize(project='avocadoyieldsdeforestation')

# Resume-safe: skip descriptions already active/done on GEE (not cancelled/failed).
# Uniform vs quantile must carry distinct descriptions (quantile gets '_q').
ACTIVE_STATES =  {'PENDING', 'RUNNING', 'SUCCEEDED', 'COMPLETED'}
EXISTING =  set()
try:
    for _o in ee.data.listOperations():
        _md =  _o.get('metadata', {})
        if _md.get('state') in ACTIVE_STATES and _md.get('description'):
            EXISTING.add(_md['description'])
except Exception as _e:
    print(f"  (could not list existing tasks: {_e})")
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
    m  =  ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4)))
    return img.updateMask(m)

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


if args.quantile:
    with open(os.path.join(data_dir, f"quantile_bin_edges_{bins}.json")) as f:
        EDGES =  json.load(f)
    assert len(EDGES) == bins + 1
    def assign_bin(img):
        b =  ee.Image.constant(0)
        for i in range(1, len(EDGES) - 1):
            b =  b.where(img.gte(EDGES[i]), i)
        return b
    folder =  f"{args.level}_2period_hists_{bins}bins_quantile"
else:
    def assign_bin(img):
        eps =  (NDVI_MAX - NDVI_MIN) / 1_000_000
        im =  img.max(NDVI_MIN + eps).min(NDVI_MAX - eps).subtract(NDVI_MIN)
        return im.divide((NDVI_MAX - NDVI_MIN) / bins).floor()
    folder =  f"{args.level}_2period_hists_{bins}bins"


def compute_2period_hist(feature):
    geom =  feature.geometry()
    pm   =  ee.Number(feature.get('planting_month'))
    gsy  =  ee.Number(feature.get('gs_year'))
    gs0  =  ee.Date.fromYMD(gsy, pm, 1)
    # two 3-month periods spanning the same 6-month season as the 3-period model
    n1 =  get_ndvi_composite(geom, gs0, gs0.advance(3, 'month'))
    n2 =  get_ndvi_composite(geom, gs0.advance(3, 'month'), gs0.advance(6, 'month'))
    b1, b2 =  assign_bin(n1.select('ndvi')), assign_bin(n2.select('ndvi'))
    hist_p12 =  b1.add(b2.multiply(bins)).rename('hist_p12')
    nsq =  bins * bins
    hist =  hist_p12.reduceRegion(ee.Reducer.fixedHistogram(0, nsq, nsq), geom, 30,
                                  maxPixels=500_000_000)
    return feature.set(hist)


# ── Geometries ───────────────────────────────────────────
if args.level == 'muni':
    geom_file =  os.path.join(data_dir, "muni_geometries_for_ee.csv")
    id_col, batch_n =  "muncode", 25
else:
    geom_file =  os.path.join(data_dir, "adc_geometries_for_ee.csv")
    id_col, batch_n =  "adcid", 500

geoms =  pd.read_csv(geom_file)
geoms['coords'] =  geoms['coords'].apply(literal_eval)
geoms['state']  =  geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2]) if args.level == 'muni' \
                  else geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2])
geoms =  geoms[geoms['state'] >= args.start_state].copy()
geoms['eeGeom'] =  geoms['coords'].apply(ee.Geometry)
print(f"{len(geoms):,} {args.level}s; years={years}; folder={folder}")

submitted =  0
for state in sorted(geoms['state'].unique()):
    sg =  geoms[geoms['state'] == state]
    for year in years:
        sg2 =  sg.copy()
        sg2['eeFeature'] =  sg2.apply(lambda r: ee.Feature(r['eeGeom'], {
            id_col: r[id_col], 'planting_month': int(r['planting_month']),
            'gs_year': year}), axis=1)
        nb =  (len(sg2) + batch_n - 1) // batch_n
        for b in range(nb):
            batch =  sg2.iloc[b * batch_n:(b + 1) * batch_n]
            desc =  f"2period_hist_{args.level}_{state}_{b}_{year}{DESC_SUFFIX}"
            if desc in EXISTING:
                continue
            res  =  ee.FeatureCollection(batch['eeFeature'].to_list()).map(compute_2period_hist)
            task =  ee.batch.Export.table.toDrive(
                collection=res, folder=folder, description=desc,
                selectors=[id_col, 'hist_p12'])
            for attempt in range(8):
                try:
                    task.start(); break
                except ee.ee_exception.EEException as e:
                    if 'too many tasks' in str(e).lower():
                        print("  queue full, waiting 300s..."); time.sleep(300)
                    else:
                        raise
            submitted +=  1
            print(f"  {desc} ({len(batch)}) [{submitted}]", flush=True)
            if args.limit and submitted >= args.limit:
                print(f"\nLimit {args.limit} reached. Folder: {folder}"); sys.exit(0)

print(f"\nAll {submitted} tasks submitted -> Drive/{folder}/")
