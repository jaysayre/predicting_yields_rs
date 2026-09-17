"""
2_ls_cropland_features.py — AEF-mirroring NDVI features on cropland-masked pixels.

WHY THIS EXISTS
The existing `ls_harmonic_features.py` extraction is UNMASKED (every pixel in the
ADC) and its features are 16x16 = 256-bin JOINT histograms. That is not
comparable to AEF, which masks to ESA WorldCover class 40 and summarises with
64 dims x {mean, 6 percentiles, 8 quantile bins} -- i.e. MARGINAL summaries.

Applying the cropland mask to the 256-bin joint histograms does not work: median
cropland area is ~1.1% of an ADC, giving a median of ~20 Landsat pixels per ADC
(30% have zero, 73% have fewer pixels than the 256 bins they must fill). AEF
survives the same mask only because it is 10 m (~9x more pixels) AND its bins are
8-per-dimension marginals rather than 256-bin joints.

So this script mirrors AEF's FEATURE TYPES on cropland-masked NDVI, which are
robust at ~20 pixels:

  dims (10): p1, p2, p3            reconstructed NDVI at the 3 phase midpoints
             h_const, h_cos1..3, h_sin1..3   the 7 harmonic coefficients
  per dim:   mean                            -> 10   (mirrors AEF means)
             p10 p25 p50 p75 p90 stdDev      -> 60   (mirrors AEF percentiles)
             8 equal-mass quantile bins      -> 80   (mirrors AEF qhist)
  total: 150 features per unit-year

Quantile edges come from Data/cropland_qbin_edges_8.json, derived as equal-mass
cuts over the 2,344,722 already-extracted ADC-years (no extra GEE needed).

The harmonic fit itself is identical to ls_harmonic_features.py (per-year,
3-harmonic, planting-anchored phase midpoints) so the two runs differ ONLY in
(a) the cropland mask and (b) marginal-vs-joint summarisation.

Usage:
  ~/miniforge3/envs/geo_env/bin/python 2_ls_cropland_features.py --level adc \
      --years 2017-2024 --project <gcp-project-id> [--no_mask] [--limit N]
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
parser.add_argument('--harmonics', type=int, default=3)
parser.add_argument('--start_state', type=str, default='01')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--project', type=str, default=os.environ.get('EE_PROJECT'),
                    help='Earth Engine-enabled Google Cloud project id (or set EE_PROJECT)')
parser.add_argument('--no_mask', action='store_true',
                    help='skip the cropland mask (unmasked control run)')
parser.add_argument('--only_adcids', type=str, default=None,
                    help='file with one id per line; restrict extraction to these '
                         '(e.g. the 95,264 INEGI census maize ADCs, which is all the '
                         'ADC-level accuracy table needs -- 32%% of the 293k universe)')
parser.add_argument('--tag', type=str, default=None,
                    help="suffix for descriptions/Drive folder. REQUIRED with "
                         "--only_adcids: restricting changes which ADCs land in batch b, "
                         "so a restricted run must not reuse an unrestricted run's "
                         "descriptions (same desc + different members = silent corruption).")
parser.add_argument('--states', type=str, default=None,
                    help='comma-sep state codes, for cost pilots across DIVERSE states '
                         '(a single-state pilot underestimated cost 7.5x)')
parser.add_argument('--skip_done_projects', type=str, default=None)
parser.add_argument('--allow_dup_pending', action='store_true',
                    help='skip only SUCCEEDED (not PENDING) on --skip_done_projects, so a '
                         'fresh unthrottled project can re-run batches parked pending elsewhere')
args =  parser.parse_args()

NH =  args.harmonics
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
COEF  =  ['h_const'] + sum([[f'h_cos{k}', f'h_sin{k}'] for k in range(1, NH + 1)], [])
DIMS  =  ['p1', 'p2', 'p3'] + COEF                      # 10 dims
PCTS  =  [10, 25, 50, 75, 90]
NQB   =  8

# Edges MUST come from the same pixel population the features are computed on.
# The first version used edges derived from the UNMASKED panel; on masked cropland
# they put 55.6% of p1 into bin 0 and 0.2% into bin 7 -- ~3 effective levels out of
# 8, which is the exact pathology AEF's quantile-bin pilot was built to fix.
# `_masked.json` is pooled from 13,556 already-extracted masked ADC-years.
_edge_file =  ("cropland_qbin_edges_8_masked.json" if not args.no_mask
               else "cropland_qbin_edges_8.json")
with open(os.path.join(data_dir, _edge_file)) as f:
    QEDGE =  json.load(f)                                # dim -> 7 interior cuts
print(f"quantile edges: {_edge_file}")

# NOTE the '_cen' tag: restricting to a subset changes which ADCs fall in batch b,
# so a restricted run MUST NOT reuse the unrestricted run's descriptions -- same
# desc + different member ADCs would silently corrupt the panel on resume/merge.
MTAG   =  ('' if args.no_mask else '_crop') + (f'_{args.tag}' if args.tag else '')
if args.only_adcids and not args.tag:
    sys.exit("--only_adcids requires --tag (see its help)")
folder =  f"{args.level}_cropland_features{MTAG}"

OUT_COLS =  ([f"{d}_mean" for d in DIMS] +
             [f"{d}_p{p}" for d in DIMS for p in PCTS] +
             [f"{d}_sd" for d in DIMS] +
             [f"{d}_qb{i}" for d in DIMS for i in range(NQB)])


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


def _bin_index(img, edges):
    """0..NQB-1 bin index for one dim given its interior quantile cuts.

    MUST inherit img's mask. ee.Image.constant(0) is unmasked everywhere and
    .where() only overwrites where the condition holds, so without the final
    updateMask every non-cropland pixel silently lands in bin 0 -- the quantile
    features would then describe the whole ADC rather than its cropland, which is
    precisely the masking this script exists to apply. (Symptom: qb columns 0%
    null while mean/pct/sd were 1.4% null, and 93.6% of mass piled into bin 0.)
    """
    idx =  ee.Image.constant(0)
    for i, cut in enumerate(edges):
        idx =  idx.where(img.gte(cut), i + 1)
    return idx.updateMask(img.mask())


def build_cmask(batch_geom):
    """ONE cropland mask per batch (mirrors gb/6_ee_alpha_earth_binned_hist.py)."""
    if args.no_mask:
        return None
    return (ee.ImageCollection("ESA/WorldCover/v200").filterBounds(batch_geom)
            .map(lambda x: x.eq(40).rename('ag')).max())


def compute_features(feature, cmask=None):
    """cmask MUST be built once per batch and passed in (see build_cmask).

    Building it here from `feature.geometry()` re-filters and re-reduces the 10 m
    global WorldCover collection once per ADC -- 500x per task. That, plus the
    9x pixel-read of a 10 m layer on a 30 m analysis, is why the first attempt
    cost 5.97 EECU-hr/task against the unmasked harmonic run's 1.41.
    """
    geom =  feature.geometry()
    pm   =  ee.Number(feature.get('planting_month'))
    gsy  =  ee.Number(feature.get('gs_year'))
    ref  =  ee.Date.fromYMD(gsy, 1, 1)

    cols =  []
    for sat, b in BAND_MAP.items():
        col =  ee.ImageCollection(sat).filterBounds(geom).filterDate(ref, ref.advance(1, 'year'))
        _f =  (lambda x: ndvi_with_harmonics(x, b['red'], b['nir'], ref).updateMask(cmask)) \
              if cmask is not None else \
              (lambda x: ndvi_with_harmonics(x, b['red'], b['nir'], ref))
        cols.append(ee.ImageCollection(ee.Algorithms.If(col.size(), col.map(_f),
                                                        ee.ImageCollection([]))))
    # dtype-matched masked dummy so a zero-image ADC still has the 8 bands
    # linearRegression needs (see ls_harmonic_features.py for the dtype rules)
    dummy =  ee.Image.cat(
        [(ee.Image.constant(0).rename(c).toFloat() if c == 'h_const'
          else ee.Image.constant(0).rename(c).toDouble()) for c in COEF] +
        [ee.Image.constant(0).rename('ndvi').toFloat().clamp(-1, 1)]
    ).updateMask(ee.Image.constant(0))
    coll =  ee.ImageCollection(cols[0].merge(cols[1]).merge(cols[2])).merge(
        ee.ImageCollection([dummy]))

    fit  =  coll.select(COEF + ['ndvi']).reduce(ee.Reducer.linearRegression(numX=len(COEF), numY=1))
    coef =  fit.select('coefficients').arrayProject([0]).arrayFlatten([COEF])

    def recon(off):
        t   =  ee.Number(pm).add(off).subtract(1).mod(12).divide(12.0)
        out =  coef.select('h_const')
        for k in range(1, NH + 1):
            ang =  t.multiply(2 * math.pi * k)
            out =  (out.add(coef.select(f'h_cos{k}').multiply(ee.Number(ang.cos())))
                       .add(coef.select(f'h_sin{k}').multiply(ee.Number(ang.sin()))))
        return out

    dim_img =  ee.Image.cat([recon(1).rename('p1'), recon(3).rename('p2'),
                             recon(5).rename('p3')] +
                            [coef.select(c).rename(c) for c in COEF])

    # mean + percentiles + stdDev, all in one reduceRegion
    red =  (ee.Reducer.mean()
            .combine(ee.Reducer.percentile(PCTS), None, True)
            .combine(ee.Reducer.stdDev(), None, True))
    stats =  dim_img.reduceRegion(reducer=red, geometry=geom, scale=30,
                                  maxPixels=1e9, bestEffort=True, tileScale=16)

    props =  {}
    for d in DIMS:
        props[f"{d}_mean"] =  stats.get(f"{d}_mean")
        props[f"{d}_sd"]   =  stats.get(f"{d}_stdDev")
        for p in PCTS:
            props[f"{d}_p{p}"] =  stats.get(f"{d}_p{p}")

    # equal-mass quantile bins per dim -> fraction of pixels in each of 8 bins.
    # ONE reduceRegion over a 10-band bin-index image. Doing this per dim (10
    # extra reduceRegions on top of the stats one) blew EE's memory limit on 2 of
    # 3 pilot tasks -- fixedHistogram on a multi-band image returns one histogram
    # per band, so the whole thing costs a single reduction.
    qidx =  ee.Image.cat([
        _bin_index(dim_img.select(d), QEDGE[d]).rename(d) for d in DIMS])
    hists =  qidx.reduceRegion(
        reducer=ee.Reducer.fixedHistogram(0, NQB, NQB), geometry=geom,
        scale=30, maxPixels=1e9, bestEffort=True, tileScale=16)

    for d in DIMS:
        h   =  hists.get(d)
        arr =  ee.Array(ee.Algorithms.If(h, h, ee.Array([[0, 0]] * NQB))) \
                 .slice(1, 1, 2).project([0]).toList()
        tot =  ee.Number(ee.List(arr).reduce(ee.Reducer.sum()))
        for i in range(NQB):
            props[f"{d}_qb{i}"] =  ee.Algorithms.If(
                tot, ee.Number(ee.List(arr).get(i)).divide(tot), None)

    return feature.set(props)


def main():
    # ── geometries + resume ──────────────────────────────────
    ACTIVE =  {'PENDING', 'RUNNING', 'SUCCEEDED', 'COMPLETED'}
    EXISTING =  set()
    try:
        for o in ee.data.listOperations():
            md =  o.get('metadata', {})
            if md.get('state') in ACTIVE and md.get('description'):
                EXISTING.add(md['description'])
    except Exception as e:
        print(f"  (list ops: {e})")
    if args.skip_done_projects:
        # Normally skip anything SUCCEEDED **or in-flight (PENDING/RUNNING)** on the
        # named projects, so a fresh project only picks up genuinely-unqueued batches.
        # With --allow_dup_pending, skip only DONE (succeeded) — lets a fresh, un-
        # throttled project re-run batches still parked PENDING on quota-locked
        # projects (harmless duplicates; the pull dedupes by description).
        _skip_states =  ('SUCCEEDED', 'COMPLETED') if args.allow_dup_pending else ACTIVE
        for _op in [s.strip() for s in args.skip_done_projects.split(',') if s.strip()]:
            try:
                ee.Initialize(project=_op)
                for o in ee.data.listOperations():
                    md =  o.get('metadata', {})
                    if md.get('state') in _skip_states and md.get('description'):
                        EXISTING.add(md['description'])
            except Exception as e:
                print(f"  (skip proj {_op}: {e})")
        ee.Initialize(project=args.project)

    if args.level == 'muni':
        geom_file =  os.path.join(data_dir, "muni_geometries_for_ee.csv"); id_col, batch_n =  "muncode", 25
    elif args.level == 'adc':
        geom_file =  os.path.join(data_dir, "adc_geometries_for_ee.csv");  id_col, batch_n =  "adcid", 500
    else:
        geom_file =  os.path.join(data_dir, "CIMMYT", "cimmyt_plot_geometries_for_ee.csv")
        id_col, batch_n =  "plot_id", 500

    geoms =  pd.read_csv(geom_file)
    geoms['coords'] =  geoms['coords'].apply(literal_eval)
    geoms[id_col]   =  geoms[id_col].astype(str)
    if args.level == 'cimmyt':
        geoms['state'] =  geoms['state'].astype(str).str.zfill(2)
    else:
        geoms['state'] =  geoms['muncode'].apply(lambda x: str(x).zfill(5)[:2])
    geoms =  geoms[geoms['state'] >= args.start_state].copy()
    if args.only_adcids:
        keep =  set(x.strip() for x in open(args.only_adcids) if x.strip())
        n0 =  len(geoms)
        geoms =  geoms[geoms[id_col].isin(keep)].copy()
        print(f"  restricted to {len(geoms):,} of {n0:,} {args.level}s "
              f"from {args.only_adcids}")
    geoms =  geoms.dropna(subset=['planting_month'])
    geoms['eeGeom'] =  geoms['coords'].apply(ee.Geometry)
    print(f"{len(geoms):,} {args.level}s; years={YEARS}; {len(OUT_COLS)} cols; folder={folder}")

    submitted =  0
    states =  sorted(geoms['state'].unique())
    if args.states:
        want =  {x.strip().zfill(2) for x in args.states.split(',')}
        states =  [s for s in states if s in want]
    for state in states:
        sg =  geoms[geoms['state'] == state]
        for year in YEARS:
            sg2 =  sg.copy()
            sg2['eeFeature'] =  sg2.apply(lambda r: ee.Feature(r['eeGeom'], {
                id_col: r[id_col], 'planting_month': int(r['planting_month']), 'gs_year': year}), axis=1)
            nb =  (len(sg2) + batch_n - 1) // batch_n
            for b in range(nb):
                desc =  f"cropfeat{MTAG}_{args.level}_{state}_{b}_{year}"
                if desc in EXISTING:
                    continue
                batch =  sg2.iloc[b * batch_n:(b + 1) * batch_n]
                fc    =  ee.FeatureCollection(batch['eeFeature'].to_list())
                cmask =  build_cmask(fc.geometry())      # ONCE per batch, not per ADC
                res   =  fc.map(lambda f: compute_features(f, cmask))
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
                if args.limit is not None and submitted >= args.limit:
                    print(f"\nLimit {args.limit} reached. Folder: {folder}"); sys.exit(0)

    print(f"\nAll {submitted} tasks submitted -> Drive/{folder}/")


if __name__ == '__main__':
    main()
