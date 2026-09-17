"""
Combined AEF satellite embedding download script.
Set MODE below to select which variant to run.

Modes:
  "mexico_mun"  - Mexico municipalities, cropland-masked  -> Drive alpha_earth_agmask/
  "mexico_adc"  - Mexico ADCs, cropland-masked            -> Drive alpha_earth_adcs/
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
from shapely import to_geojson, force_2d


# ============================================================
# CONFIGURATION — set MODE to control which variant runs
# ============================================================
MODE =  "mexico_mun"
# ============================================================


# ===== Shared setup =====

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)


def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


def safe_start(task, desc):
    """Start a GEE export task, waiting if queue is full."""
    while True:
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg =  str(e).lower()
            if 'too many tasks' in msg:
                print(f"    Queue full, waiting 5 min... ({desc})", flush=True)
                time.sleep(300)
            elif 'payload size' in msg:
                print(f"    Payload too large: {desc}", flush=True)
                return False
            else:
                print(f"    ERROR {desc}: {e}", flush=True)
                return False


def to_ee_collection(gdf, props_dict, props_from_row=None):
    """Convert GeoDataFrame rows to EE FeatureCollection."""
    gdf['coords']    =  gdf['geometry'].apply(lambda x: literal_eval(to_geojson(force_2d(x))))
    gdf['eeGeom']    =  gdf['coords'].apply(ee.Geometry)
    gdf['eeFeature'] =  gdf.apply(
        lambda row: ee.Feature(
            row['eeGeom'],
            {**props_dict, **{k: row[k] for k in (props_from_row or []) if k in row.index}}
        ),
        axis=1
    )
    return ee.FeatureCollection(gdf['eeFeature'].to_list())


def extract_and_export(img, collection, aef_year, description, folder, selectors):
    """Run reduceRegions and start export task."""
    means =  img.reduceRegions(
        collection=collection,
        reducer=ee.Reducer.mean(),
        scale=10
    )
    means =  means.map(lambda f: f.set('year', aef_year))
    task  =  ee.batch.Export.table.toDrive(
        collection=means,
        description=description,
        folder=folder,
        selectors=selectors
    )
    try:
        task.start()
        print(f"  Export started: {description}")
    except ee.ee_exception.EEException as e:
        if 'payload size' in str(e).lower():
            print(f"  WARNING: payload too large for {description}, skipping")
        else:
            raise


feat_names  =  [f"A{add_zeros(x)}" for x in range(0, 64)]
alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")


# ===== Mode-specific functions =====

def run_mexico_mun():
    """Mexico municipalities, ag-masked (from ee_alpha_earth.py)."""
    cols =  feat_names + ['CVE_ENT', 'CVE_MUN', 'year']

    muns =  gpd.read_file("../raw_data/mgm2007/Municipios_2007.shp")
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.simplify(100))
    muns =  muns.to_crs("EPSG:4326")
    muns['geometry'] =  muns['geometry'].apply(lambda x: x.buffer(0))
    muns['coords']   =  muns['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    muns['eeGeom']   =  muns['coords'].apply(ee.Geometry)
    muns['eeFeature'] =  muns.apply(
        lambda row: ee.Feature(row['eeGeom'], {'CVE_ENT': row['CVE_ENT'], 'CVE_MUN': row['CVE_MUN']}),
        axis=1
    )

    for state in muns['CVE_ENT'].unique():
        mun_coll   =  ee.FeatureCollection(muns.loc[muns['CVE_ENT'] == state, 'eeFeature'].to_list())
        print(mun_coll.size().getInfo())
        state_coll =  alpha_earth.filterBounds(mun_coll.geometry())
        wc         =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(mun_coll.geometry())
        agland     =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

        for year in range(2017, 2025):
            print(f"Processing state {state} for year {year}...")

            img       =  state_coll.filterDate(f'{year}-01-01', f'{year}-12-31').mean()
            img       =  img.updateMask(agland)
            mun_means =  img.reduceRegions(
                collection=mun_coll,
                reducer=ee.Reducer.mean(),
                scale=10
            )
            mun_means =  mun_means.map(lambda f: f.set('year', year))
            task =  ee.batch.Export.table.toDrive(
                collection=mun_means,
                description=f'alpha_earth_agmask_state_{state}_year_{year}',
                folder='alpha_earth_agmask',
                selectors=cols
            )
            task.start()


def run_mexico_adc():
    """Mexico ADCs, ag-masked (from ee_alpha_earth_adcs.py)."""
    out_dir =  os.path.join(os.path.dirname(__file__), "alpha_earth_adcs")
    os.makedirs(out_dir, exist_ok=True)

    adcs =  gpd.read_file(os.path.expanduser("~") + "/Dropbox/Projects/Maize_prediction/Data/Shapefiles/adc_shapefile.shp")
    adcs =  adcs.dropna(subset=['adcid'])
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.simplify(0.0001))
    adcs['geometry'] =  adcs['geometry'].apply(lambda x: x.buffer(0))
    adcs['coords']   =  adcs['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    print(f"Loaded {len(adcs)} ADCs")

    cols       =  feat_names + ['adcid', 'year']
    BATCH_SIZE =  3000

    for state in sorted(adcs['est'].unique()):
        state_adcs =  adcs.loc[adcs['est'] == state].copy()
        print(f"\nState {state}: {len(state_adcs)} ADCs — converting geometries...")

        state_adcs['eeGeom']    =  state_adcs['coords'].apply(ee.Geometry)
        state_adcs['eeFeature'] =  state_adcs.apply(
            lambda row: ee.Feature(row['eeGeom'], {'adcid': row['adcid']}), axis=1)

        n_batches =  (len(state_adcs) + BATCH_SIZE - 1) // BATCH_SIZE

        for b in range(n_batches):
            batch       =  state_adcs.iloc[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            batch_label =  f"_b{b}" if n_batches > 1 else ""

            adc_coll   =  ee.FeatureCollection(batch['eeFeature'].to_list())
            state_coll =  alpha_earth.filterBounds(adc_coll.geometry())
            wc         =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(adc_coll.geometry())
            agland     =  wc.map(lambda x: x.eq(40).rename('ag_area')).max()

            for year in range(2017, 2025):
                desc =  f'alpha_earth_adc_state_{state}{batch_label}_year_{year}'
                print(f"  {desc} — submitting export task...")

                img       =  state_coll.filterDate(f'{year}-01-01', f'{year}-12-31').mean()
                img       =  img.updateMask(agland)
                adc_means =  img.reduceRegions(
                    collection=adc_coll,
                    reducer=ee.Reducer.mean(),
                    scale=10
                )
                adc_means =  adc_means.map(lambda f: f.set('year', year))

                try:
                    task =  ee.batch.Export.table.toDrive(
                        collection=adc_means,
                        description=desc,
                        folder='alpha_earth_adcs',
                        selectors=cols
                    )
                    task.start()
                except ee.ee_exception.EEException as e:
                    print(f"  ERROR: {desc} — {e}")
                    continue


# ===== Main dispatch =====

if __name__ == '__main__':
    modes =  {
        'mexico_mun': run_mexico_mun,
        'mexico_adc': run_mexico_adc,
    }
    if MODE not in modes:
        raise ValueError(f"Unknown MODE '{MODE}'. Choose from: {list(modes.keys())}")

    print(f"Running mode: {MODE}")
    print(f"{'='*60}\n")
    modes[MODE]()
