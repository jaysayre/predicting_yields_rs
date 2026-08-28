"""
ee_alpha_earth_binned_hist.py
=============================
Extract true histogram-binned AEF features: for each of 64 embedding
dimensions, compute the fraction of cropland pixels falling into each of
N_BINS fixed-width bins. This captures distribution shape (bimodality,
heavy tails) that percentile summaries miss.

Features: 64 dims × 8 bins = 512 per spatial unit
Value range: [-0.8, 0.8], bin width = 0.2
Each feature = fraction of agricultural pixels in that bin (sums to 1 per dim)

Submits both ADC-level and municipality-level tasks.

Usage:
  conda activate ML_env
  python3 ee_alpha_earth_binned_hist.py
  python3 ee_alpha_earth_binned_hist.py --skip_mun
  python3 ee_alpha_earth_binned_hist.py --start_state 15
  python3 ee_alpha_earth_binned_hist.py --years 2022

Author: Jay Sayre
Date: 2026-03-15
"""

import os
import time
import argparse
from ast import literal_eval

import ee
import pandas as pd
import geopandas as gpd
from shapely import to_geojson


# ============================================================
# Configuration
# ============================================================
ALL_YEARS      =  [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
N_BINS         =  8
BIN_MIN        =  -0.8
BIN_MAX        =  0.8
ADC_BATCH      =  2000
ADC_BATCH_OAX  =  500
SCALE          =  10
ADC_FOLDER     =  'alpha_earth_adcs_binned_hist'
MUN_FOLDER     =  'alpha_earth_mun_binned_hist'
# ============================================================

BIN_WIDTH =  (BIN_MAX - BIN_MIN) / N_BINS


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


feat_names =  [f"A{add_zeros(x)}" for x in range(64)]


def build_indicator_image(img):
    """Create indicator bands: for each dim × bin, 1 if pixel in bin, else 0.

    Returns an ee.Image with 64 × N_BINS = 512 bands, each named
    A00_b0, A00_b1, ..., A63_b7.
    """
    bands =  []
    for dim_name in feat_names:
        band =  img.select(dim_name)
        for b in range(N_BINS):
            lower =  BIN_MIN + b * BIN_WIDTH
            upper =  lower + BIN_WIDTH
            if b < N_BINS - 1:
                # [lower, upper)
                indicator =  band.gte(lower).And(band.lt(upper))
            else:
                # [lower, upper] for last bin (inclusive)
                indicator =  band.gte(lower).And(band.lte(upper))
            bands.append(indicator.rename(f"{dim_name}_b{b}"))
    return ee.Image(bands).toFloat()


def build_out_cols():
    """Build output column names."""
    cols =  []
    for dim_name in feat_names:
        for b in range(N_BINS):
            cols.append(f"{dim_name}_b{b}")
    return cols


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
                wait =  300 * (attempt + 1)
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
    parser =  argparse.ArgumentParser()
    parser.add_argument('--skip_mun', action='store_true',
                        help='Skip municipality tasks')
    parser.add_argument('--start_state', type=str, default='01',
                        help='Start ADC tasks from this state')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated years (default: all)')
    args =  parser.parse_args()

    years =  ALL_YEARS
    if args.years:
        years =  [int(y) for y in args.years.split(',')]

    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

    alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    out_cols    =  build_out_cols()

    adc_selectors =  out_cols + ['adcid', 'year']
    mun_selectors =  out_cols + ['CVE_ENT', 'CVE_MUN', 'year']

    print(f"Binned histogram features: {len(out_cols)} ({len(feat_names)} dims × {N_BINS} bins)")
    print(f"Bin range: [{BIN_MIN}, {BIN_MAX}], width: {BIN_WIDTH:.4f}")
    print(f"Years: {years}")
    print(f"ADC batch size: {ADC_BATCH} (Oaxaca: {ADC_BATCH_OAX})")

    # ============================================================
    # Load shapefiles
    # ============================================================

    print("\n--- Loading ADC shapefile ---")
    adcs =  gpd.read_file(
        "/home/jsayre/Dropbox/Projects/Maize_prediction/Data/Shapefiles/"
        "adc_shapefile.shp"
    )
    adcs =  adcs.dropna(subset=['adcid'])
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.simplify(0.0001))
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.buffer(0))
    adcs['coords']   =  adcs['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"  {len(adcs)} ADCs across {adcs['est'].nunique()} states")

    print("\n--- Loading municipality shapefile ---")
    muns =  gpd.read_file(
        "/home/jsayre/Dropbox/Projects/Avocado_Deforestation/Data/raw/"
        "spatial/Municipality_shp/MUNICIPIOS.shp"
    )
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.simplify(100))
    muns =  muns.to_crs("EPSG:4326")
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.buffer(0))
    muns['coords']   =  muns['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"  {len(muns)} municipalities across {muns['CVE_ENT'].nunique()} states")

    total_adc =  0
    total_mun =  0

    # ============================================================
    # Municipality-level tasks
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

        for year in years:
            desc =  f'aef_binhist_mun_state_{state}_year_{year}'
            print(f"  {desc} ({len(state_muns)} muns)...", end='', flush=True)

            img =  state_coll.filterDate(f'{year}-01-01', f'{year}-12-31').mean()
            img =  img.updateMask(agland)

            # Build indicator image (512 bands)
            indicator_img =  build_indicator_image(img)

            # Reduce with mean() — gives fraction of pixels per bin
            results =  indicator_img.reduceRegions(
                collection=mun_coll,
                reducer=ee.Reducer.mean(),
                scale=SCALE
            )
            results =  results.map(lambda f: f.set('year', year))

            if submit_task(results, desc, MUN_FOLDER, mun_selectors):
                total_mun += 1
                print(f" #{total_mun}")
            else:
                print()

    print(f"\n  Municipality tasks submitted: {total_mun}")

    # ============================================================
    # ADC-level tasks
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

            for year in years:
                desc =  f'aef_binhist_state_{state}{batch_label}_year_{year}'
                print(f"    {desc}...", end='', flush=True)

                img =  state_coll.filterDate(
                    f'{year}-01-01', f'{year}-12-31').mean()
                img =  img.updateMask(agland)

                indicator_img =  build_indicator_image(img)

                results =  indicator_img.reduceRegions(
                    collection=adc_coll,
                    reducer=ee.Reducer.mean(),
                    scale=SCALE
                )
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
