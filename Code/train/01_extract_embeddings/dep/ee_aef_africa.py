"""
ee_aef_africa.py
================
Extract AEF embedding features for HarvestStat Africa admin regions via GEE.

Extracts three feature types at admin_2 (boundary) level for all 1,109
regions across 33 countries:
  1. Mean embeddings:       64 features  (A00-A63)
  2. Percentile + stdDev:  384 features  (64 dims x 6 stats)
  3. Binned histograms:    512 features  (64 dims x 8 bins)

Uses ESA WorldCover v2 cropland mask (class 40) and AEF annual composites
(GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) at 10m scale.

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate ML_env
  python3 ee_aef_africa.py                        # all feature types, all years
  python3 ee_aef_africa.py --type binned           # only binned histograms
  python3 ee_aef_africa.py --type hist             # only percentile + stdDev
  python3 ee_aef_africa.py --type mean             # only mean embeddings
  python3 ee_aef_africa.py --years 2022,2023       # subset of years
  python3 ee_aef_africa.py --start_country NG      # resume from country

Author: Jay Sayre
"""

import os, sys, time, argparse
from ast import literal_eval

import ee
import geopandas as gpd
from shapely import to_geojson

sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# Configuration
# ============================================================
ALL_YEARS  =  [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
N_BINS     =  8
BIN_MIN    =  -0.8
BIN_MAX    =  0.8
SCALE      =  10

# Google Drive export folders
FOLDER_MEAN    =  'aef_africa_adm2_mean'
FOLDER_HIST    =  'aef_africa_adm2_hist'
FOLDER_BINNED  =  'aef_africa_adm2_binned_hist'
# ============================================================

BIN_WIDTH  =  (BIN_MAX - BIN_MIN) / N_BINS

home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
hvstat_dir  =  os.path.join(proj_dir, "Data", "HarvestStat_Africa")

feat_names  =  [f"A{d:02d}" for d in range(64)]


# -- Feature column names --------------------------------------
mean_selectors  =  feat_names + ['fnid', 'year']

pct_cols  =  []
for d in feat_names:
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"{d}{s}")
pct_selectors  =  pct_cols + ['fnid', 'year']

bin_cols  =  [f"{d}_b{b}" for d in feat_names for b in range(N_BINS)]
bin_selectors  =  bin_cols + ['fnid', 'year']


def build_indicator_image(img):
    """512-band indicator image: 1 if pixel in bin, 0 otherwise."""
    bands  =  []
    for dim_name in feat_names:
        band  =  img.select(dim_name)
        for b in range(N_BINS):
            lo  =  BIN_MIN + b * BIN_WIDTH
            hi  =  lo + BIN_WIDTH
            if b < N_BINS - 1:
                indicator  =  band.gte(lo).And(band.lt(hi))
            else:
                indicator  =  band.gte(lo).And(band.lte(hi))
            bands.append(indicator.rename(f"{dim_name}_b{b}"))
    return ee.Image(bands).toFloat()


def submit_task(results, desc, folder, selectors):
    """Submit GEE export task with retry on queue-full."""
    task  =  ee.batch.Export.table.toDrive(
        collection=results, description=desc,
        folder=folder, selectors=selectors
    )
    for attempt in range(10):
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg  =  str(e).lower()
            if 'too many tasks' in msg:
                wait  =  300 * (attempt + 1)
                print(f"  queue full, wait {wait // 60}m...",
                      end='', flush=True)
                time.sleep(wait)
            elif 'payload size' in msg:
                print(f"  PAYLOAD TOO LARGE")
                return False
            else:
                print(f"  ERROR: {e}")
                return False
    print(f"  GAVE UP after 10 retries")
    return False


def main():
    parser  =  argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'mean', 'hist', 'binned'],
                        help='Feature type to extract')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated years (default: all)')
    parser.add_argument('--start_country', type=str, default=None,
                        help='Resume from this country code')
    args  =  parser.parse_args()

    years  =  ALL_YEARS
    if args.years:
        years  =  [int(y) for y in args.years.split(',')]

    do_mean    =  args.type in ('all', 'mean')
    do_hist    =  args.type in ('all', 'hist')
    do_binned  =  args.type in ('all', 'binned')

    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

    alpha_earth  =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    print(f"Years: {years}")
    print(f"Features: mean={do_mean}, hist={do_hist}, binned={do_binned}")
    print(f"Bins: {N_BINS} in [{BIN_MIN}, {BIN_MAX}], width={BIN_WIDTH:.3f}")

    # -- Load boundary gpkg ------------------------------------
    print("\nLoading boundary gpkg...")
    gdf  =  gpd.read_file(
        os.path.join(hvstat_dir, "hvstat_africa_boundary_v1.1.gpkg")
    )
    gdf['geometry']  =  gdf['geometry'].simplify(0.001)
    gdf['geometry']  =  gdf['geometry'].buffer(0)
    gdf['coords']    =  gdf['geometry'].apply(
        lambda g: literal_eval(to_geojson(g))
    )
    print(f"  {len(gdf)} regions across {gdf['country_code'].nunique()} countries")

    # Build EE features
    gdf['eeGeom']     =  gdf['coords'].apply(ee.Geometry)
    gdf['eeFeature']  =  gdf.apply(
        lambda r: ee.Feature(r['eeGeom'], {'fnid': r['fnid']}), axis=1
    )

    counts  =  {'mean': 0, 'hist': 0, 'binned': 0}

    for cc in sorted(gdf['country_code'].unique()):
        if args.start_country and cc < args.start_country:
            continue

        sub   =  gdf[gdf['country_code'] == cc]
        coll  =  ee.FeatureCollection(sub['eeFeature'].tolist())

        # Spatial filters (shared across years for this country)
        ae_filt  =  alpha_earth.filterBounds(coll.geometry())
        wc       =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(
                         coll.geometry())
        agland   =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

        print(f"\n  {cc}: {len(sub)} regions")

        for year in years:
            img  =  ae_filt.filterDate(
                f'{year}-01-01', f'{year}-12-31'
            ).mean()
            img  =  img.updateMask(agland)

            # -- Mean embeddings -------------------------------
            if do_mean:
                desc  =  f'aef_africa_mean_{cc}_{year}'
                print(f"    {desc}...", end='', flush=True)
                results  =  img.reduceRegions(
                    collection=coll,
                    reducer=ee.Reducer.mean(),
                    scale=SCALE
                ).map(lambda f: f.set('year', year))
                if submit_task(results, desc, FOLDER_MEAN, mean_selectors):
                    counts['mean'] += 1
                    print(f" #{counts['mean']}")
                else:
                    print()

            # -- Percentile + stdDev ---------------------------
            if do_hist:
                desc  =  f'aef_africa_hist_{cc}_{year}'
                print(f"    {desc}...", end='', flush=True)
                reducer  =  (ee.Reducer.percentile([10, 25, 50, 75, 90])
                             .combine(ee.Reducer.stdDev(),
                                      sharedInputs=True))
                results  =  img.reduceRegions(
                    collection=coll,
                    reducer=reducer,
                    scale=SCALE
                ).map(lambda f: f.set('year', year))
                if submit_task(results, desc, FOLDER_HIST, pct_selectors):
                    counts['hist'] += 1
                    print(f" #{counts['hist']}")
                else:
                    print()

            # -- Binned histogram ------------------------------
            if do_binned:
                desc  =  f'aef_africa_binned_{cc}_{year}'
                print(f"    {desc}...", end='', flush=True)
                indicator_img  =  build_indicator_image(img)
                results  =  indicator_img.reduceRegions(
                    collection=coll,
                    reducer=ee.Reducer.mean(),
                    scale=SCALE
                ).map(lambda f: f.set('year', year))
                if submit_task(results, desc, FOLDER_BINNED, bin_selectors):
                    counts['binned'] += 1
                    print(f" #{counts['binned']}")
                else:
                    print()

    print(f"\n{'=' * 55}")
    print(f"  Tasks submitted: mean={counts['mean']}, "
          f"hist={counts['hist']}, binned={counts['binned']}")
    print(f"  Total: {sum(counts.values())}")
    print(f"  Monitor: https://code.earthengine.google.com/tasks")
    print(f"{'=' * 55}")
    print(f"\nAfter tasks complete, download CSVs from Drive and run:")
    print(f"  python3 consolidate_aef_africa.py --drive_dir <path>")


if __name__ == '__main__':
    main()
