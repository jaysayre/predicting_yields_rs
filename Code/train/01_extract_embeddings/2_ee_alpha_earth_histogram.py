"""
2_ee_alpha_earth_histogram.py
===========================
Extract distributional AEF embedding features at the ADC level.

Instead of just the mean across agricultural pixels (ee.Reducer.mean()),
extracts percentiles [10, 25, 50, 75, 90] and stdDev for each of the 64
embedding dimensions. This gives 64 × 6 = 384 features per ADC, capturing
the within-ADC distribution of embeddings.

Extracts year 2022 only (for proof of concept / thought experiment).
Run for all years by changing YEARS below.

Usage:
  conda activate ML_env   # needs earthengine-api
  python3 2_ee_alpha_earth_histogram.py

Output: CSV files exported to Google Drive folder 'alpha_earth_adcs_hist'
        (one file per state, year 2022)

Author: Jay Sayre
Date: 2026-02-23
"""

import os
import time
from ast import literal_eval

import ee
import pandas as pd
import geopandas as gpd
from shapely import to_geojson, force_2d


# ============================================================
# Configuration
# ============================================================
YEARS      =  [2022]          # Proof of concept: 2022 only
BATCH_SIZE =  2000            # ADCs per GEE task (smaller than mean-only)
SCALE      =  10              # 10m resolution
FOLDER     =  'alpha_earth_adcs_hist'
# ============================================================


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names =  [f"A{add_zeros(x)}" for x in range(64)]


def main():
    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

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
    selectors =  out_cols + ['adcid', 'year']

    print(f"Output features: {len(out_cols)} ({len(feat_names)} dims × 6 stats)")
    print(f"Selectors: {len(selectors)} columns")
    print(f"Years: {YEARS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load ADC shapefile
    adcs =  gpd.read_file(
        os.path.expanduser("~") + "/Dropbox/Projects/Maize_prediction/Data/Shapefiles/"
        "adc_shapefile.shp"
    )
    adcs =  adcs.dropna(subset=['adcid'])
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.simplify(0.0001))
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.buffer(0))
    adcs['coords']   =  adcs['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"\nLoaded {len(adcs)} ADCs across {adcs['est'].nunique()} states")

    total_tasks =  0

    for state in sorted(adcs['est'].unique()):
        state_adcs =  adcs.loc[adcs['est'] == state].copy()
        print(f"\nState {state}: {len(state_adcs)} ADCs")

        # Convert to EE geometries
        state_adcs['eeGeom']    =  state_adcs['coords'].apply(ee.Geometry)
        state_adcs['eeFeature'] =  state_adcs.apply(
            lambda row: ee.Feature(row['eeGeom'], {'adcid': row['adcid']}),
            axis=1
        )

        n_batches =  (len(state_adcs) + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(n_batches):
            batch       =  state_adcs.iloc[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            batch_label =  f"_b{b}" if n_batches > 1 else ""

            adc_coll   =  ee.FeatureCollection(batch['eeFeature'].to_list())
            state_coll =  alpha_earth.filterBounds(adc_coll.geometry())

            # Agricultural land mask (ESA WorldCover v2, class 40 = cropland)
            wc     =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(
                           adc_coll.geometry())
            agland =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

            for year in YEARS:
                desc =  f'aef_hist_state_{state}{batch_label}_year_{year}'
                print(f"  {desc} — submitting...", end='', flush=True)

                img =  state_coll.filterDate(
                    f'{year}-01-01', f'{year}-12-31').mean()
                img =  img.updateMask(agland)

                # reduceRegions with percentile + stdDev reducer
                results =  img.reduceRegions(
                    collection=adc_coll,
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
    print(f"  python3 8_consolidate_hist_parquet.py")


if __name__ == '__main__':
    main()
