"""
ls_harmonic_ndvi.py — Harmonic-regression NDVI features (Model 3).

Fits a 3-harmonic annual model to each unit's Landsat NDVI time series:
    NDVI(t) = c + Σ_{k=1..3} [ a_k cos(2πk t) + b_k sin(2πk t) ],   t in years
per pixel (via ee.Reducer.linearRegression over the year's Landsat 5/7/8 NDVI
collection), then spatially averages the 7 coefficients over the unit. This is a
distinct, non-binned Landsat baseline alongside the 2- and 3-period NDVI
histograms — it summarizes seasonal SHAPE (amplitude/phase of the greenness
cycle) rather than a binned distribution.

Output: 7 features per unit-year — h_const, h_cos1, h_sin1, h_cos2, h_sin2,
        h_cos3, h_sin3. Exported to Google Drive; pull via rclone 'gdrive'.

Usage (laptop):
  ~/miniforge3/envs/geo_env/bin/python ls_harmonic_ndvi.py --level muni --years 2017-2024 [--limit N]
  ~/miniforge3/envs/geo_env/bin/python ls_harmonic_ndvi.py --level adc  --years 2022
"""
import os
import sys
import time
import argparse
from ast import literal_eval

import ee
import pandas as pd

parser =  argparse.ArgumentParser()
parser.add_argument('--level', choices=['muni', 'adc'], required=True)
parser.add_argument('--years', type=str, default='2022')
parser.add_argument('--harmonics', type=int, default=3)
parser.add_argument('--start_state', type=str, default='01')
parser.add_argument('--limit', type=int, default=None)
args =  parser.parse_args()

NH =  args.harmonics
if '-' in args.years:
    a, b =  args.years.split('-'); years =  list(range(int(a), int(b) + 1))
else:
    years =  [int(args.years)]

try:
    ee.Initialize(project='avocadoyieldsdeforestation')
except Exception:
    ee.Authenticate(); ee.Initialize(project='avocadoyieldsdeforestation')

# Resume-safe: skip descriptions already active/done on GEE (not cancelled/failed).
ACTIVE_STATES =  {'PENDING', 'RUNNING', 'SUCCEEDED', 'COMPLETED'}
EXISTING =  set()
try:
    for _o in ee.data.listOperations():
        _md =  _o.get('metadata', {})
        if _md.get('state') in ACTIVE_STATES and _md.get('description'):
            EXISTING.add(_md['description'])
except Exception as _e:
    print(f"  (could not list existing tasks: {_e})")
print(f"  {len(EXISTING)} active/done tasks on GEE; will skip matching descriptions.")

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")

BAND_MAP =  {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}
COEF_NAMES =  ['h_const'] + sum([[f'h_cos{k}', f'h_sin{k}'] for k in range(1, NH + 1)], [])
N_X =  len(COEF_NAMES)            # 1 + 2*NH
folder =  f"{args.level}_harmonic_ndvi_h{NH}"


def scale_facts(img):
    ob =  img.select('SR_B.').multiply(0.0000275).add(-0.2)
    tb =  img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(ob, overwrite=True).addBands(tb, overwrite=True)

def cloud_mask(img):
    qa =  img.select(['QA_PIXEL'])
    m  =  ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4)))
    return img.updateMask(m)

def ndvi_img(img, red, nir, ref):
    """NDVI band + harmonic predictor bands from image acquisition time."""
    img  =  cloud_mask(scale_facts(img))
    ndvi =  img.normalizedDifference([nir, red]).rename('ndvi')
    # fractional years since Jan 1 of the reference year
    t    =  ee.Image(ee.Number(img.date().difference(ref, 'year'))).rename('t').toFloat()
    bands =  [ee.Image.constant(1).rename('h_const').toFloat()]
    for k in range(1, NH + 1):
        ang =  t.multiply(2 * 3.141592653589793 * k)
        bands.append(ang.cos().rename(f'h_cos{k}'))
        bands.append(ang.sin().rename(f'h_sin{k}'))
    return ee.Image.cat(bands).addBands(ndvi).set('system:time_start', img.get('system:time_start'))


def harmonic_coeffs(feature):
    geom =  feature.geometry()
    gsy  =  ee.Number(feature.get('gs_year'))
    ref  =  ee.Date.fromYMD(gsy, 1, 1)
    start, end =  ref, ref.advance(1, 'year')

    cols =  []
    for sat, b in BAND_MAP.items():
        col =  ee.ImageCollection(sat).filterBounds(geom).filterDate(start, end)
        mapped =  ee.Algorithms.If(
            col.size(),
            col.map(lambda x: ndvi_img(x, b['red'], b['nir'], ref)),
            ee.ImageCollection([]))
        cols.append(ee.ImageCollection(mapped))
    coll =  ee.ImageCollection(cols[0].merge(cols[1]).merge(cols[2]))

    # per-pixel harmonic least squares: X = predictor bands, Y = ndvi
    fit =  (coll.select(COEF_NAMES + ['ndvi'])
                .reduce(ee.Reducer.linearRegression(numX=N_X, numY=1)))
    coeffs =  (fit.select('coefficients')
                  .arrayProject([0]).arrayFlatten([COEF_NAMES]))
    # tileScale splits the reduction into smaller memory chunks; municipality
    # geometries otherwise exceed the per-task memory limit for the harmonic fit.
    means =  coeffs.reduceRegion(ee.Reducer.mean(), geom, 30,
                                 maxPixels=500_000_000, tileScale=16)
    return feature.set(means)


# ── Geometries ───────────────────────────────────────────
if args.level == 'muni':
    geom_file =  os.path.join(data_dir, "muni_geometries_for_ee.csv")
    id_col, batch_n =  "muncode", 25
else:
    geom_file =  os.path.join(data_dir, "adc_geometries_for_ee.csv")
    id_col, batch_n =  "adcid", 500

geoms =  pd.read_csv(geom_file)
geoms['coords'] =  geoms['coords'].apply(literal_eval)
geoms['state']  =  geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2])
geoms =  geoms[geoms['state'] >= args.start_state].copy()
geoms['eeGeom'] =  geoms['coords'].apply(ee.Geometry)
print(f"{len(geoms):,} {args.level}s; years={years}; {N_X} coeffs; folder={folder}")

submitted =  0
for state in sorted(geoms['state'].unique()):
    sg =  geoms[geoms['state'] == state]
    for year in years:
        sg2 =  sg.copy()
        sg2['eeFeature'] =  sg2.apply(lambda r: ee.Feature(r['eeGeom'], {
            id_col: r[id_col], 'gs_year': year}), axis=1)
        nb =  (len(sg2) + batch_n - 1) // batch_n
        for b in range(nb):
            batch =  sg2.iloc[b * batch_n:(b + 1) * batch_n]
            desc =  f"harmonic_ndvi_{args.level}_{state}_{b}_{year}"
            if desc in EXISTING:
                continue
            res  =  ee.FeatureCollection(batch['eeFeature'].to_list()).map(harmonic_coeffs)
            task =  ee.batch.Export.table.toDrive(
                collection=res, folder=folder, description=desc,
                selectors=[id_col] + COEF_NAMES)
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
