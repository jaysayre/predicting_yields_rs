"""
4_ee_alpha_earth_histogram_allyears.py
====================================
Extract distributional AEF features for all remaining years (2017-2021, 2023-2024).
Year 2022 already extracted. Submits both ADC-level and municipality-level tasks.

Uses smaller batch size (500) for Oaxaca (state 20) ADCs to avoid OOM.

Usage:
  conda activate ML_env
  python3 4_ee_alpha_earth_histogram_allyears.py

Author: Jay Sayre
Date: 2026-02-25
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
YEARS          =  [2017, 2018, 2019, 2020, 2021, 2023, 2024]
ADC_BATCH      =  2000
ADC_BATCH_OAX  =  500   # Smaller for Oaxaca (state 20)
SCALE          =  10
ADC_FOLDER     =  'alpha_earth_adcs_hist'
MUN_FOLDER     =  'alpha_earth_mun_hist'
# ============================================================


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names =  [f"A{add_zeros(x)}" for x in range(64)]


def build_reducer_and_selectors():
    pct_reducer =  ee.Reducer.percentile([10, 25, 50, 75, 90])
    std_reducer =  ee.Reducer.stdDev()
    reducer     =  pct_reducer.combine(std_reducer, sharedInputs=True)

    pct_suffixes =  ['_p10', '_p25', '_p50', '_p75', '_p90']
    std_suffix   =  '_stdDev'
    out_cols     =  []
    for feat in feat_names:
        for suf in pct_suffixes:
            out_cols.append(feat + suf)
        out_cols.append(feat + std_suffix)

    return reducer, out_cols


def submit_task(results, desc, folder, selectors):
    """Submit a GEE export task with retry on queue-full."""
    task =  ee.batch.Export.table.toDrive(
        collection=results,
        description=desc,
        folder=folder,
        selectors=selectors
    )
    for attempt in range(10):
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg =  str(e).lower()
            if 'too many tasks' in msg:
                wait =  300 * (attempt + 1)  # 5, 10, 15 min...
                print(f" queue full, waiting {wait//60} min (attempt {attempt+1})...",
                      end='', flush=True)
                time.sleep(wait)
            elif 'payload size' in msg:
                print(f" PAYLOAD TOO LARGE")
                return False
            else:
                print(f" ERROR: {e}")
                return False
    print(f" GAVE UP after 10 retries")
    return False


def main():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument('--skip_mun', action='store_true',
                        help='Skip municipality tasks (already submitted)')
    parser.add_argument('--start_state', type=str, default='01',
                        help='Start ADC tasks from this state (skip earlier)')
    args =  parser.parse_args()

    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    reducer, out_cols =  build_reducer_and_selectors()

    adc_selectors =  out_cols + ['adcid', 'year']
    mun_selectors =  out_cols + ['CVE_ENT', 'CVE_MUN', 'year']

    print(f"Output features: {len(out_cols)} ({len(feat_names)} dims × 6 stats)")
    print(f"Years: {YEARS}")
    print(f"ADC batch size: {ADC_BATCH} (Oaxaca: {ADC_BATCH_OAX})")

    # ============================================================
    # Load shapefiles
    # ============================================================

    print("\n--- Loading ADC shapefile ---")
    adcs =  gpd.read_file(
        os.path.expanduser("~") + "/Dropbox/Projects/Maize_prediction/Data/Shapefiles/"
        "adc_shapefile.shp"
    )
    adcs =  adcs.dropna(subset=['adcid'])
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.simplify(0.0001))
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.buffer(0))
    adcs['coords']   =  adcs['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"  {len(adcs)} ADCs across {adcs['est'].nunique()} states")

    print("\n--- Loading municipality shapefile ---")
    muns =  gpd.read_file(
        os.path.expanduser("~") + "/Dropbox/Projects/Maize_prediction/Data/muncodes/shp/"
        "MUNICIPIOS.shp"
    )
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.simplify(100))
    muns =  muns.to_crs("EPSG:4326")
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.buffer(0))
    muns['coords']   =  muns['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"  {len(muns)} municipalities across {muns['CVE_ENT'].nunique()} states")

    total_adc =  0
    total_mun =  0

    # ============================================================
    # Municipality-level tasks (one per state per year)
    # ============================================================

    if args.skip_mun:
        print("\n  Skipping municipality tasks (--skip_mun)")
    else:
        print("\n" + "=" * 60)
        print("  Submitting MUNICIPALITY-level tasks")
        print("=" * 60)

    for state in sorted(muns['CVE_ENT'].unique()):
        if args.skip_mun:
            break
        state_muns =  muns.loc[muns['CVE_ENT'] == state].copy()
        state_muns['eeGeom']    =  state_muns['coords'].apply(ee.Geometry)
        state_muns['eeFeature'] =  state_muns.apply(
            lambda row: ee.Feature(row['eeGeom'], {
                'CVE_ENT': row['CVE_ENT'],
                'CVE_MUN': row['CVE_MUN']
            }), axis=1)

        mun_coll   =  ee.FeatureCollection(state_muns['eeFeature'].to_list())
        state_coll =  alpha_earth.filterBounds(mun_coll.geometry())
        wc         =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(
                           mun_coll.geometry())
        agland     =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

        for year in YEARS:
            desc =  f'aef_hist_mun_state_{state}_year_{year}'
            print(f"  {desc} ({len(state_muns)} muns)...", end='', flush=True)

            img =  state_coll.filterDate(f'{year}-01-01', f'{year}-12-31').mean()
            img =  img.updateMask(agland)
            results =  img.reduceRegions(
                collection=mun_coll, reducer=reducer, scale=SCALE)
            results =  results.map(lambda f: f.set('year', year))

            if submit_task(results, desc, MUN_FOLDER, mun_selectors):
                total_mun += 1
                print(f" #{total_mun}")
            else:
                print()

    print(f"\n  Municipality tasks submitted: {total_mun}")

    # ============================================================
    # ADC-level tasks (batched per state per year)
    # ============================================================

    print("\n" + "=" * 60)
    print("  Submitting ADC-level tasks")
    print("=" * 60)

    for state in sorted(adcs['est'].unique()):
        if state < args.start_state:
            continue
        state_adcs =  adcs.loc[adcs['est'] == state].copy()
        batch_size =  ADC_BATCH_OAX if state == '20' else ADC_BATCH

        print(f"\n  State {state}: {len(state_adcs)} ADCs (batch={batch_size})")

        state_adcs['eeGeom']    =  state_adcs['coords'].apply(ee.Geometry)
        state_adcs['eeFeature'] =  state_adcs.apply(
            lambda row: ee.Feature(row['eeGeom'], {'adcid': row['adcid']}),
            axis=1)

        n_batches =  (len(state_adcs) + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch       =  state_adcs.iloc[b * batch_size : (b + 1) * batch_size]
            batch_label =  f"_b{b}" if n_batches > 1 else ""

            adc_coll   =  ee.FeatureCollection(batch['eeFeature'].to_list())
            state_coll =  alpha_earth.filterBounds(adc_coll.geometry())
            wc         =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(
                               adc_coll.geometry())
            agland     =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

            for year in YEARS:
                desc =  f'aef_hist_state_{state}{batch_label}_year_{year}'
                print(f"    {desc}...", end='', flush=True)

                img =  state_coll.filterDate(
                    f'{year}-01-01', f'{year}-12-31').mean()
                img =  img.updateMask(agland)
                results =  img.reduceRegions(
                    collection=adc_coll, reducer=reducer, scale=SCALE)
                results =  results.map(lambda f: f.set('year', year))

                if submit_task(results, desc, ADC_FOLDER, adc_selectors):
                    total_adc += 1
                    print(f" #{total_adc}")
                else:
                    print()

    print(f"\n{'='*60}")
    print(f"  ADC tasks submitted:  {total_adc}")
    print(f"  Mun tasks submitted:  {total_mun}")
    print(f"  Total:                {total_adc + total_mun}")
    print(f"  Monitor at: https://code.earthengine.google.com/tasks")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
