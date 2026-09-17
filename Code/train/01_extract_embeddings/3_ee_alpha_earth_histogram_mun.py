"""
3_ee_alpha_earth_histogram_mun.py
===============================
Extract distributional AEF embedding features at the municipality level.

Same reducer as the ADC-level script (percentiles + stdDev), but using
municipality polygons from the 2007 MGM shapefile. This gives 64 × 6 = 384
features per municipality, capturing the within-municipality distribution
of embeddings across agricultural pixels.

Extracts year 2022 only (for proof of concept).

Usage:
  conda activate ML_env   # needs earthengine-api
  python3 3_ee_alpha_earth_histogram_mun.py

Output: CSV files exported to Google Drive folder 'alpha_earth_mun_hist'
        (one file per state, year 2022)

Author: Jay Sayre
Date: 2026-02-23
"""

import os
import time
from ast import literal_eval

import ee
EE_PROJECT =  os.environ.get("EE_PROJECT")   # an Earth Engine-enabled Google Cloud project id
if not EE_PROJECT:
    raise SystemExit("Set EE_PROJECT=<gcp-project-id> before running this script")

import pandas as pd
import geopandas as gpd
from shapely import to_geojson


# ============================================================
# Configuration
# ============================================================
YEARS  =  [2022]          # Proof of concept: 2022 only
SCALE  =  10              # 10m resolution
FOLDER =  'alpha_earth_mun_hist'
# ============================================================


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names =  [f"A{add_zeros(x)}" for x in range(64)]


def main():
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    # Build reducer: percentiles + stdDev
    pct_reducer =  ee.Reducer.percentile([10, 25, 50, 75, 90])
    std_reducer =  ee.Reducer.stdDev()
    reducer     =  pct_reducer.combine(std_reducer, sharedInputs=True)

    # Build output column names
    pct_suffixes =  ['_p10', '_p25', '_p50', '_p75', '_p90']
    std_suffix   =  '_stdDev'
    out_cols     =  []
    for feat in feat_names:
        for suf in pct_suffixes:
            out_cols.append(feat + suf)
        out_cols.append(feat + std_suffix)
    selectors =  out_cols + ['CVE_ENT', 'CVE_MUN', 'year']

    print(f"Output features: {len(out_cols)} ({len(feat_names)} dims × 6 stats)")
    print(f"Selectors: {len(selectors)} columns")
    print(f"Years: {YEARS}")

    # Load municipality shapefile
    muns =  gpd.read_file(
        os.path.expanduser("~") + "/Dropbox/Projects/Maize_prediction/Data/muncodes/shp/"
        "MUNICIPIOS.shp"
    )
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.simplify(100))
    muns =  muns.to_crs("EPSG:4326")
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.buffer(0))
    muns['coords']   =  muns['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"\nLoaded {len(muns)} municipalities across {muns['CVE_ENT'].nunique()} states")

    total_tasks =  0

    for state in sorted(muns['CVE_ENT'].unique()):
        state_muns =  muns.loc[muns['CVE_ENT'] == state].copy()
        print(f"\nState {state}: {len(state_muns)} municipalities")

        # Convert to EE geometries
        state_muns['eeGeom']    =  state_muns['coords'].apply(ee.Geometry)
        state_muns['eeFeature'] =  state_muns.apply(
            lambda row: ee.Feature(row['eeGeom'], {
                'CVE_ENT': row['CVE_ENT'],
                'CVE_MUN': row['CVE_MUN']
            }),
            axis=1
        )

        mun_coll   =  ee.FeatureCollection(state_muns['eeFeature'].to_list())
        state_coll =  alpha_earth.filterBounds(mun_coll.geometry())

        # Agricultural land mask (ESA WorldCover v2, class 40 = cropland)
        wc     =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(
                       mun_coll.geometry())
        agland =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

        for year in YEARS:
            desc =  f'aef_hist_mun_state_{state}_year_{year}'
            print(f"  {desc} — submitting...", end='', flush=True)

            img =  state_coll.filterDate(
                f'{year}-01-01', f'{year}-12-31').mean()
            img =  img.updateMask(agland)

            # reduceRegions with percentile + stdDev reducer
            results =  img.reduceRegions(
                collection=mun_coll,
                reducer=reducer,
                scale=SCALE
            )
            results =  results.map(lambda f: f.set('year', year))

            try:
                task =  ee.batch.Export.table.toDrive(
                    collection=results,
                    description=desc,
                    folder=FOLDER,
                    selectors=selectors
                )
                task.start()
                total_tasks += 1
                print(f" started (task #{total_tasks})")
            except ee.ee_exception.EEException as e:
                msg =  str(e).lower()
                if 'too many tasks' in msg:
                    print(f" queue full, waiting 5 min...")
                    time.sleep(300)
                    task.start()
                    total_tasks += 1
                    print(f"  retried: started (task #{total_tasks})")
                elif 'payload size' in msg:
                    print(f" PAYLOAD TOO LARGE — try smaller batch size")
                else:
                    print(f" ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"  Total GEE export tasks submitted: {total_tasks}")
    print(f"  Output folder on Google Drive: {FOLDER}")
    print(f"  Monitor at: https://code.earthengine.google.com/tasks")
    print(f"{'='*60}")
    print(f"\nAfter tasks complete, download CSVs from Drive and run:")
    print(f"  python3 9_consolidate_hist_parquet_mun.py --csv_dir /path/to/csvs")


if __name__ == '__main__':
    main()
