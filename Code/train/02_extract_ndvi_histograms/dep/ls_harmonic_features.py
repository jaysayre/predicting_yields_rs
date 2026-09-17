"""
ls_harmonic_features.py — Unified per-year harmonic NDVI feature extraction.

Fits a per-YEAR harmonic model to each pixel's full-year Landsat 5/7/8 NDVI
series, then derives ALL of the NDVI-family features from that single fit:

  * harmonic coefficients        -> 1 + 2*NH per unit  (the "harmonic" model)
  * 2-period reconstructed hist  -> bins^2             (P1 x P2 trajectory)
  * 3-period reconstructed hists -> 3 * bins^2         (P12, P23, P13)

The histograms bin RECONSTRUCTED NDVI (the fitted curve evaluated at
planting-anchored phase midpoints), not window composites, so a pixel only
needs enough cloud-free observations ANYWHERE in the year to fit the curve —
not clear observations inside each phenology window. This makes coverage
robust to cloud gaps (see the Chiapas proof: 69 annual obs, 0 in the peak
window, yet a valid reconstruction at every phase).

Both fixed-width and quantile bin edges are computed from the same
reconstructed values in one pass (no re-fit).

Fit is PER YEAR (uses only the target year's observations) so year-specific
growing conditions — the signal that predicts that year's yield — are retained.

Usage (laptop):
  ~/miniforge3/envs/geo_env/bin/python ls_harmonic_features.py --level adc  --years 2022 [--limit N]
  ~/miniforge3/envs/geo_env/bin/python ls_harmonic_features.py --level muni --years 2003-2022
"""
import os
import sys
import json
import math
import time
import argparse
from ast import literal_eval

import ee
import pandas as pd

parser =  argparse.ArgumentParser()
parser.add_argument('--level', choices=['muni', 'adc', 'cimmyt'], required=True)
parser.add_argument('--years', type=str, default='2022')
parser.add_argument('--bins', type=int, default=16)
parser.add_argument('--harmonics', type=int, default=3)
parser.add_argument('--start_state', type=str, default='01')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--project', type=str, default='avocadoyieldsdeforestation',
                    help='EE/Cloud project to bill compute to (quota source)')
parser.add_argument('--cropland_mask', action='store_true',
                    help='restrict to ESA WorldCover v200 class 40 (cropland), matching '
                         'the AEF extraction. Without it the histograms summarise EVERY '
                         'pixel in the ADC (forest/pasture/urban included), which both '
                         'dilutes the crop signal and makes NDVI non-comparable to AEF. '
                         'Note it also drops the ~20k small smallholder ADCs that have no '
                         'WorldCover cropland - keep the unmasked run for those.')
parser.add_argument('--skip_done_projects', type=str, default=None,
                    help='comma-sep projects whose SUCCEEDED harmfeat tasks to also '
                         'skip (cross-project dedup when moving a tail to a new project)')
args =  parser.parse_args()

BINS =  args.bins
NH   =  args.harmonics
if '-' in args.years:
    a, b =  args.years.split('-'); YEARS =  list(range(int(a), int(b) + 1))
else:
    YEARS =  [int(args.years)]

try:
    ee.Initialize(project=args.project)
except Exception:
    ee.Authenticate(); ee.Initialize(project=args.project)

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")

BAND_MAP =  {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}
COEF =  ['h_const'] + sum([[f'h_cos{k}', f'h_sin{k}'] for k in range(1, NH + 1)], [])
N_X  =  len(COEF)
TILE =  16 if args.level == 'muni' else 4

# quantile edges (per-NDVI-quantile) shared with the window-histogram variant
with open(os.path.join(data_dir, f"quantile_bin_edges_{BINS}.json")) as f:
    Q_EDGES =  json.load(f)          # length BINS+1, over [0,1]


def scale_facts(img):
    return img.addBands(img.select('SR_B.').multiply(0.0000275).add(-0.2), overwrite=True)

def cloud_mask(img):
    qa =  img.select('QA_PIXEL')
    return img.updateMask(ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4))))

def ndvi_with_harmonics(img, red, nir, ref):
    img  =  cloud_mask(scale_facts(img))
    ndvi =  img.normalizedDifference([nir, red]).rename('ndvi')
    t    =  ee.Image(ee.Number(img.date().difference(ref, 'year'))).rename('t').toFloat()
    bands =  [ee.Image.constant(1).rename('h_const').toFloat()]
    for k in range(1, NH + 1):
        ang =  t.multiply(2 * math.pi * k)
        bands.append(ang.cos().rename(f'h_cos{k}'))
        bands.append(ang.sin().rename(f'h_sin{k}'))
    return ee.Image.cat(bands).addBands(ndvi)


def bin_fixed(img):
    eps =  1e-6
    return img.max(eps).min(1 - eps).divide(1.0 / BINS).floor()

def bin_quantile(img):
    b =  ee.Image.constant(0)
    for i in range(1, BINS):
        b =  b.where(img.gte(Q_EDGES[i]), i)
    return b

def joint_hist(bin_a, bin_b, name):
    idx =  bin_a.add(bin_b.multiply(BINS)).rename(name)
    return idx


def compute_features(feature):
    geom =  feature.geometry()
    pm   =  ee.Number(feature.get('planting_month'))
    gsy  =  ee.Number(feature.get('gs_year'))
    ref  =  ee.Date.fromYMD(gsy, 1, 1)

    # cropland mask (matches the AEF extraction: ESA WorldCover v200 class 40)
    cmask =  (ee.ImageCollection("ESA/WorldCover/v200").filterBounds(geom)
              .map(lambda x: x.eq(40).rename('ag')).max()) if args.cropland_mask else None

    cols =  []
    for sat, b in BAND_MAP.items():
        col =  ee.ImageCollection(sat).filterBounds(geom).filterDate(ref, ref.advance(1, 'year'))
        _f =  (lambda x: ndvi_with_harmonics(x, b['red'], b['nir'], ref).updateMask(cmask)) \
              if cmask is not None else \
              (lambda x: ndvi_with_harmonics(x, b['red'], b['nir'], ref))
        mapped =  ee.Algorithms.If(col.size(), col.map(_f), ee.ImageCollection([]))
        cols.append(ee.ImageCollection(mapped))
    # fully-masked 8-band dummy so an ADC-year with ZERO Landsat images still has
    # the bands linearRegression needs (masked pixels are ignored by the reducer,
    # so ADCs with data are unaffected; a truly-empty ADC just yields a masked
    # fit -> empty histogram -> a null row that clean_harmonic_features.py drops).
    # band dtypes MUST match the real images EXACTLY or the merged collection is
    # rejected as heterogeneous: h_const is float, cos()/sin() promote h_cos*/
    # h_sin* to double, and normalizedDifference gives the *ranged* Float<-1,1>
    # (hence the clamp). Verified against a real Landsat image's band types.
    dummy =  ee.Image.cat(
        [(ee.Image.constant(0).rename(c).toFloat() if c == 'h_const'
          else ee.Image.constant(0).rename(c).toDouble()) for c in COEF] +
        [ee.Image.constant(0).rename('ndvi').toFloat().clamp(-1, 1)]
    ).updateMask(ee.Image.constant(0))
    coll =  ee.ImageCollection(cols[0].merge(cols[1]).merge(cols[2])).merge(ee.ImageCollection([dummy]))

    fit  =  coll.select(COEF + ['ndvi']).reduce(ee.Reducer.linearRegression(numX=N_X, numY=1))
    coef =  fit.select('coefficients').arrayProject([0]).arrayFlatten([COEF])

    # planting-anchored phase midpoints, reconstructed server-side from the
    # per-feature planting month (3-period mids at planting +1,+3,+5; 2-period +1.5,+4.5)
    def recon_ee(off):
        t =  ee.Number(pm).add(off).subtract(1).mod(12).divide(12.0)
        out =  coef.select('h_const')
        for k in range(1, NH + 1):
            ang =  t.multiply(2 * math.pi * k)
            out =  out.add(coef.select(f'h_cos{k}').multiply(ee.Number(ang).cos()))
            out =  out.add(coef.select(f'h_sin{k}').multiply(ee.Number(ang).sin()))
        return out.rename('ndvi_hat')

    p1 =  recon_ee(1); p2 =  recon_ee(3); p3 =  recon_ee(5)     # 3-period mids
    q1 =  recon_ee(1.5); q2 =  recon_ee(4.5)                    # 2-period mids

    bands =  []
    for tag, binner in [('f', bin_fixed), ('q', bin_quantile)]:
        b1, b2, b3 =  binner(p1), binner(p2), binner(p3)
        c1, c2     =  binner(q1), binner(q2)
        bands += [
            joint_hist(b1, b2, f'h3_{tag}_p12'),
            joint_hist(b2, b3, f'h3_{tag}_p23'),
            joint_hist(b1, b3, f'h3_{tag}_p13'),
            joint_hist(c1, c2, f'h2_{tag}_p12'),
        ]
    nsq =  BINS * BINS
    hist =  ee.Image(bands).reduceRegion(
        ee.Reducer.fixedHistogram(0, nsq, nsq), geom, 30,
        maxPixels=500_000_000, tileScale=TILE)
    coef_means =  coef.reduceRegion(ee.Reducer.mean(), geom, 30,
                                    maxPixels=500_000_000, tileScale=TILE)
    return feature.set(hist).set(coef_means)


HIST_COLS =  [f'h3_{t}_{p}' for t in ('f', 'q') for p in ('p12', 'p23', 'p13')] + \
            [f'h2_{t}_p12' for t in ('f', 'q')]
OUT_COLS  =  COEF + HIST_COLS
MTAG      =  "_crop" if args.cropland_mask else ""   # keep masked exports separate
folder    =  f"{args.level}_harmonic_features_{BINS}b_h{NH}{MTAG}"


# ── Geometries + resume ──────────────────────────────────
ACTIVE_STATES =  {'PENDING', 'RUNNING', 'SUCCEEDED', 'COMPLETED'}
EXISTING =  set()
try:
    for o in ee.data.listOperations():
        md =  o.get('metadata', {})
        if md.get('state') in ACTIVE_STATES and md.get('description'):
            EXISTING.add(md['description'])
except Exception as e:
    print(f"  (list ops: {e})")

# cross-project dedup: also skip anything already SUCCEEDED on other projects
# (used when moving an undone tail to a fresh project without recomputing)
if args.skip_done_projects:
    for _op in [s.strip() for s in args.skip_done_projects.split(',') if s.strip()]:
        try:
            ee.Initialize(project=_op)
            n0 =  len(EXISTING)
            for o in ee.data.listOperations():
                md =  o.get('metadata', {})
                if md.get('state') in ('SUCCEEDED', 'COMPLETED') and md.get('description'):
                    EXISTING.add(md['description'])
            print(f"  skip-scan {_op}: +{len(EXISTING) - n0} done descriptions")
        except Exception as e:
            print(f"  (skip proj {_op}: {e})")
    ee.Initialize(project=args.project)   # restore billing project for submits

if args.level == 'muni':
    geom_file =  os.path.join(data_dir, "muni_geometries_for_ee.csv"); id_col, batch_n =  "muncode", 25
elif args.level == 'adc':
    geom_file =  os.path.join(data_dir, "adc_geometries_for_ee.csv");  id_col, batch_n =  "adcid", 500
else:  # cimmyt plots
    geom_file =  os.path.join(data_dir, "CIMMYT", "cimmyt_plot_geometries_for_ee.csv")
    id_col, batch_n =  "plot_id", 500

geoms =  pd.read_csv(geom_file)
geoms['coords'] =  geoms['coords'].apply(literal_eval)
geoms[id_col]   =  geoms[id_col].astype(str)
if args.level == 'cimmyt':
    geoms['state'] =  geoms['state'].astype(str).str.zfill(2)      # explicit state column
else:
    geoms['state'] =  geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2])
geoms =  geoms[geoms['state'] >= args.start_state].copy()
geoms =  geoms.dropna(subset=['planting_month'])
geoms['eeGeom'] =  geoms['coords'].apply(ee.Geometry)
print(f"{len(geoms):,} {args.level}s; years={YEARS}; {len(OUT_COLS)} cols; folder={folder}")

submitted =  0
for state in sorted(geoms['state'].unique()):
    sg =  geoms[geoms['state'] == state]
    for year in YEARS:
        sg2 =  sg.copy()
        sg2['eeFeature'] =  sg2.apply(lambda r: ee.Feature(r['eeGeom'], {
            id_col: r[id_col], 'planting_month': int(r['planting_month']), 'gs_year': year}), axis=1)
        nb =  (len(sg2) + batch_n - 1) // batch_n
        for b in range(nb):
            desc =  f"harmfeat{MTAG}_{args.level}_{state}_{b}_{year}"
            if desc in EXISTING:
                continue
            batch =  sg2.iloc[b * batch_n:(b + 1) * batch_n]
            res   =  ee.FeatureCollection(batch['eeFeature'].to_list()).map(compute_features)
            task  =  ee.batch.Export.table.toDrive(
                collection=res, folder=folder, description=desc,
                selectors=[id_col, 'gs_year'] + OUT_COLS)
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
