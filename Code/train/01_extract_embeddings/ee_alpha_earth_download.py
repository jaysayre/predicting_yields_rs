"""
Combined AEF satellite embedding download script.
Set MODE below to select which variant to run.

Modes:
  "mexico_mun"             - Mexico municipalities, ag-masked
  "california"             - CA LandIQ crops, county + field level
  "mexico_adc"             - Mexico ADCs, ag-masked
  "mexico_huerto"          - Avocado huertos, mun + field level
  "mexico_huerto_buffered" - Buffered huertos, mun level only
"""
import os
import time
from ast import literal_eval

import ee
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
ee.Initialize(project='avocadoyieldsdeforestation')


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


def run_california():
    """CA LandIQ crops, county + field level (from ee_alpha_earth_ca.py)."""

    # ---------- Configuration ----------
    landiq_base =  '/home/jsayre/Dropbox/Projects/Avocado_Project/Data/LandIQ'
    landiq_files =  {
        2018: f'{landiq_base}/2018/i15_Crop_Mapping_2018.shp',
        2020: f'{landiq_base}/2020/i15_Crop_Mapping_2020.shp',
        2022: f'{landiq_base}/2022/i15_Crop_Mapping_2022_Provisional.shp',
    }
    aef_to_landiq =  {
        2017: 2018, 2018: 2018,
        2019: 2020, 2020: 2020,
        2021: 2022, 2022: 2022, 2023: 2022, 2024: 2022,
    }
    dwr_codes =  {
        'Avocados': 'C5',
        'Almonds':  'D12',
        'Tomatoes': 'T15',
        'Corn':     'F16',
    }
    county_cols =  feat_names + ['county', 'crop', 'year']
    field_cols  =  feat_names + ['field_id', 'county', 'crop', 'year', 'acres']

    # ---------- Helper functions ----------

    def filter_crop_parcels(gdf, crop_name, dwr_code, landiq_year):
        """Filter LandIQ parcels for a specific crop."""
        if 'MAIN_CROP' in gdf.columns:
            mask =  gdf['MAIN_CROP'] == dwr_code
        else:
            croptyp_cols =  [c for c in gdf.columns if c.startswith('CROPTYP')]
            mask =  pd.Series(False, index=gdf.index)
            for col in croptyp_cols:
                mask =  mask | (gdf[col] == dwr_code)
        filtered =  gdf[mask].copy()
        print(f"  {crop_name} ({dwr_code}): {len(filtered)} parcels in {landiq_year} LandIQ")
        return filtered

    def load_and_filter_landiq(landiq_year, crop_name, dwr_code):
        """Load LandIQ shapefile, filter to crop, standardize columns."""
        shp_path =  landiq_files[landiq_year]
        gdf      =  gpd.read_file(shp_path)
        col_map  =  {}
        if 'COUNTY' in gdf.columns:
            col_map['COUNTY'] =  'county'
        elif 'County' in gdf.columns:
            col_map['County'] =  'county'
        if 'ACRES' in gdf.columns:
            col_map['ACRES'] =  'acres'
        elif 'Acres' in gdf.columns:
            col_map['Acres'] =  'acres'
        gdf      =  gdf.rename(columns=col_map)
        filtered =  filter_crop_parcels(gdf, crop_name, dwr_code, landiq_year)
        if len(filtered) == 0:
            return None
        if 'UniqueID' in filtered.columns:
            filtered['field_id'] =  filtered['UniqueID'].astype(str)
        else:
            filtered['field_id'] =  [f"{landiq_year}_{i}" for i in range(len(filtered))]
        filtered =  filtered.to_crs("EPSG:4326")
        return filtered

    def prep_county_level(filtered):
        """Dissolve parcels by county for county-level extraction."""
        dissolved =  filtered.dissolve(by='county').reset_index()
        dissolved =  dissolved[['county', 'geometry']].copy()
        dissolved['geometry'] =  dissolved['geometry'].apply(lambda x: x.simplify(0.001))
        dissolved['geometry'] =  dissolved['geometry'].apply(lambda x: x.buffer(0))
        print(f"  County-level: {len(dissolved)} counties")
        return dissolved

    def prep_field_level(filtered, county):
        """Prepare individual field parcels for a single county."""
        county_fields =  filtered[filtered['county'] == county].copy()
        county_fields =  county_fields[['field_id', 'county', 'acres', 'geometry']].copy()
        county_fields['geometry'] =  county_fields['geometry'].apply(lambda x: x.simplify(0.0001))
        county_fields['geometry'] =  county_fields['geometry'].apply(lambda x: x.buffer(0))
        return county_fields

    # ---------- Main loop ----------
    parcel_cache =  {}

    for crop_name, dwr_code in dwr_codes.items():
        for aef_year in range(2017, 2025):
            landiq_year =  aef_to_landiq[aef_year]
            cache_key   =  (landiq_year, crop_name)

            print(f"\n{'='*60}")
            print(f"{crop_name}, AEF year {aef_year} (LandIQ {landiq_year})")
            print(f"{'='*60}")

            if cache_key not in parcel_cache:
                filtered =  load_and_filter_landiq(landiq_year, crop_name, dwr_code)
                parcel_cache[cache_key] =  filtered
            else:
                filtered =  parcel_cache[cache_key]
                if filtered is not None:
                    print(f"  Using cached {len(filtered)} parcels")

            if filtered is None:
                print(f"  No parcels found, skipping")
                continue

            img        =  alpha_earth.filterDate(f'{aef_year}-01-01', f'{aef_year}-12-31').mean()
            crop_label =  crop_name.lower()

            # County-level export (batched per county to avoid payload limit)
            dissolved =  prep_county_level(filtered)
            for _, row in dissolved.iterrows():
                county       =  row['county']
                county_label =  county.lower().replace(' ', '_')
                single       =  dissolved[dissolved['county'] == county].copy()
                county_coll  =  to_ee_collection(single, {'crop': crop_name}, props_from_row=['county'])
                extract_and_export(
                    img, county_coll, aef_year,
                    description=f'ae_ca_county_{crop_label}_{county_label}_{aef_year}',
                    folder='alpha_earth_ca_county',
                    selectors=county_cols
                )

            # Field-level export (batched by county)
            for county in sorted(filtered['county'].unique()):
                fields =  prep_field_level(filtered, county)
                if len(fields) == 0:
                    continue
                field_coll   =  to_ee_collection(
                    fields, {'crop': crop_name},
                    props_from_row=['field_id', 'county', 'acres']
                )
                county_label =  county.lower().replace(' ', '_')
                extract_and_export(
                    img, field_coll, aef_year,
                    description=f'ae_ca_field_{crop_label}_{county_label}_{aef_year}',
                    folder='alpha_earth_ca_field',
                    selectors=field_cols
                )


def run_mexico_adc():
    """Mexico ADCs, ag-masked (from ee_alpha_earth_adcs.py)."""
    out_dir =  os.path.join(os.path.dirname(__file__), "alpha_earth_adcs")
    os.makedirs(out_dir, exist_ok=True)

    adcs =  gpd.read_file("/home/jsayre/Dropbox/Projects/Maize_prediction/Data/Shapefiles/adc_shapefile.shp")
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


def run_mexico_huerto():
    """Avocado huertos, mun + field level (from ee_alpha_earth_huertos.py)."""
    shp_path    =  '/home/jsayre/Dropbox/Projects/Avocado_Deforestation/Data/build/BICOS_shapefiles/avo_full_kml_and_point.shp'
    BATCH_SIZE  =  500
    mun_cols    =  feat_names + ['muncode', 'year']
    huerto_cols =  feat_names + ['huerto', 'muncode', 'ha_hue', 'year']

    print("Loading avocado huerto shapefile...", flush=True)
    gdf =  gpd.read_file(shp_path)
    print(f"  {len(gdf)} huertos, {gdf['muncode'].nunique()} municipalities", flush=True)

    gdf['geometry'] =  gdf['geometry'].apply(lambda x: force_2d(x.buffer(0)))

    # Municipality-level: dissolve by muncode
    print("\nPreparing municipality-level geometries...", flush=True)
    dissolved =  gdf.dissolve(by='muncode').reset_index()
    dissolved =  dissolved[['muncode', 'geometry']].copy()
    dissolved['geometry'] =  dissolved['geometry'].apply(lambda x: force_2d(x.simplify(0.001).buffer(0)))
    print(f"  {len(dissolved)} municipality polygons", flush=True)

    dissolved['coords']    =  dissolved['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    dissolved['eeGeom']    =  dissolved['coords'].apply(ee.Geometry)
    dissolved['eeFeature'] =  dissolved.apply(
        lambda row: ee.Feature(row['eeGeom'], {'muncode': row['muncode']}),
        axis=1
    )

    total_mun_tasks    =  0
    total_huerto_tasks =  0

    for year in range(2017, 2025):
        print(f"\n{'='*60}", flush=True)
        print(f"Year {year}", flush=True)
        print(f"{'='*60}", flush=True)

        img =  alpha_earth.filterDate(f'{year}-01-01', f'{year}-12-31').mean()

        # Municipality-level exports (batch by state)
        dissolved['state'] =  dissolved['muncode'].str[:2]
        mun_tasks =  0

        for state in sorted(dissolved['state'].unique()):
            state_muns =  dissolved[dissolved['state'] == state]
            mun_coll   =  ee.FeatureCollection(state_muns['eeFeature'].to_list())

            means =  img.reduceRegions(collection=mun_coll, reducer=ee.Reducer.mean(), scale=10)
            means =  means.map(lambda f: f.set('year', year))

            desc =  f'ae_huerto_mun_state{state}_{year}'
            task =  ee.batch.Export.table.toDrive(
                collection=means,
                description=desc,
                folder='alpha_earth_huerto_mun',
                selectors=mun_cols
            )
            if safe_start(task, desc):
                mun_tasks +=  1
            else:
                # Payload too large — fall back to one municipality at a time
                for _, row in state_muns.iterrows():
                    single_coll =  ee.FeatureCollection([row['eeFeature']])
                    means =  img.reduceRegions(collection=single_coll, reducer=ee.Reducer.mean(), scale=10)
                    means =  means.map(lambda f: f.set('year', year))
                    desc  =  f'ae_huerto_mun_{row["muncode"]}_{year}'
                    task  =  ee.batch.Export.table.toDrive(
                        collection=means,
                        description=desc,
                        folder='alpha_earth_huerto_mun',
                        selectors=mun_cols
                    )
                    if safe_start(task, desc):
                        mun_tasks +=  1
                print(f"    State {state}: split into individual muns (payload too large)", flush=True)

        total_mun_tasks +=  mun_tasks
        print(f"  Municipality-level: {mun_tasks} export tasks", flush=True)

        # Huerto-level exports (batch by municipality, 500 per task)
        huerto_tasks =  0

        for muncode in sorted(gdf['muncode'].unique()):
            mun_huertos =  gdf[gdf['muncode'] == muncode].copy()
            mun_huertos =  mun_huertos[['huerto', 'muncode', 'ha_hue', 'geometry']].copy()
            mun_huertos['geometry'] =  mun_huertos['geometry'].apply(lambda x: force_2d(x.simplify(0.0001).buffer(0)))

            # Drop empty or invalid geometries
            mun_huertos =  mun_huertos[~mun_huertos['geometry'].is_empty & mun_huertos['geometry'].is_valid].copy()
            if len(mun_huertos) == 0:
                continue

            n_huertos =  len(mun_huertos)
            n_batches =  (n_huertos + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_i in range(n_batches):
                start =  batch_i * BATCH_SIZE
                end   =  min(start + BATCH_SIZE, n_huertos)
                batch =  mun_huertos.iloc[start:end].copy()

                batch['coords'] =  batch['geometry'].apply(lambda x: literal_eval(to_geojson(x)))

                # Convert to EE, skipping any invalid geometries
                ee_features =  []
                for _, r in batch.iterrows():
                    try:
                        geom =  ee.Geometry(r['coords'])
                        feat =  ee.Feature(geom, {
                            'huerto':  r['huerto'],
                            'muncode': r['muncode'],
                            'ha_hue':  float(r['ha_hue']) if pd.notna(r['ha_hue']) else 0.0,
                        })
                        ee_features.append(feat)
                    except ee.ee_exception.EEException:
                        pass  # skip invalid geometry

                if len(ee_features) == 0:
                    continue

                field_coll =  ee.FeatureCollection(ee_features)

                means =  img.reduceRegions(collection=field_coll, reducer=ee.Reducer.mean(), scale=10)
                means =  means.map(lambda f: f.set('year', year))

                suffix =  f'_b{batch_i+1}' if n_batches > 1 else ''
                desc   =  f'ae_huerto_{muncode}_{year}{suffix}'

                task =  ee.batch.Export.table.toDrive(
                    collection=means,
                    description=desc,
                    folder='alpha_earth_huerto_field',
                    selectors=huerto_cols
                )
                if safe_start(task, desc):
                    huerto_tasks +=  1

            if n_batches > 1:
                print(f"    {muncode}: {n_huertos} huertos -> {n_batches} batches", flush=True)

        total_huerto_tasks +=  huerto_tasks
        print(f"  Huerto-level: {huerto_tasks} export tasks", flush=True)

    print(f"\n\nAll exports submitted!", flush=True)
    print(f"  Municipality-level: {total_mun_tasks} total tasks -> alpha_earth_huerto_mun/", flush=True)
    print(f"  Huerto-level: {total_huerto_tasks} total tasks -> alpha_earth_huerto_field/", flush=True)


def run_mexico_huerto_buffered():
    """Buffered huertos, mun level only (from ee_alpha_earth_huertos_buffered.py)."""
    shp_path      =  '/home/jsayre/Dropbox/Projects/Avocado_Deforestation/Data/build/BICOS_shapefiles/avo_full_kml_and_point.shp'
    BUFFER_METERS =  500
    mun_cols      =  feat_names + ['muncode', 'year']

    print("Loading avocado huerto shapefile...", flush=True)
    gdf =  gpd.read_file(shp_path)
    print(f"  {len(gdf)} huertos, {gdf['muncode'].nunique()} municipalities", flush=True)

    # Reproject to metric CRS for accurate buffering, buffer, then back to 4326
    print(f"  Buffering each huerto by {BUFFER_METERS}m...", flush=True)
    gdf =  gdf.to_crs("EPSG:6372")  # Mexico ITRF2008 / LCC (meters)
    gdf['geometry'] =  gdf['geometry'].buffer(BUFFER_METERS)
    gdf =  gdf.to_crs("EPSG:4326")
    gdf['geometry'] =  gdf['geometry'].apply(lambda x: force_2d(x.buffer(0)))

    # Dissolve by municipality
    print("  Dissolving by municipality...", flush=True)
    dissolved =  gdf.dissolve(by='muncode').reset_index()
    dissolved =  dissolved[['muncode', 'geometry']].copy()
    dissolved['geometry'] =  dissolved['geometry'].apply(lambda x: force_2d(x.simplify(0.001).buffer(0)))
    print(f"  {len(dissolved)} municipality polygons", flush=True)

    # Convert to EE features
    dissolved['coords']    =  dissolved['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
    dissolved['eeGeom']    =  dissolved['coords'].apply(ee.Geometry)
    dissolved['eeFeature'] =  dissolved.apply(
        lambda row: ee.Feature(row['eeGeom'], {'muncode': row['muncode']}),
        axis=1
    )

    total_tasks =  0

    for year in range(2017, 2025):
        print(f"\n{'='*60}", flush=True)
        print(f"Year {year}", flush=True)
        print(f"{'='*60}", flush=True)

        img =  alpha_earth.filterDate(f'{year}-01-01', f'{year}-12-31').mean()

        dissolved['state'] =  dissolved['muncode'].str[:2]
        year_tasks =  0

        for state in sorted(dissolved['state'].unique()):
            state_muns =  dissolved[dissolved['state'] == state]
            mun_coll   =  ee.FeatureCollection(state_muns['eeFeature'].to_list())

            means =  img.reduceRegions(collection=mun_coll, reducer=ee.Reducer.mean(), scale=10)
            means =  means.map(lambda f: f.set('year', year))

            desc =  f'ae_huerto_buf{BUFFER_METERS}m_state{state}_{year}'
            task =  ee.batch.Export.table.toDrive(
                collection=means,
                description=desc,
                folder=f'alpha_earth_huerto_buf{BUFFER_METERS}m',
                selectors=mun_cols
            )
            if safe_start(task, desc):
                year_tasks +=  1
            else:
                # Payload too large — fall back to one municipality at a time
                for _, row in state_muns.iterrows():
                    single_coll =  ee.FeatureCollection([row['eeFeature']])
                    means =  img.reduceRegions(collection=single_coll, reducer=ee.Reducer.mean(), scale=10)
                    means =  means.map(lambda f: f.set('year', year))
                    desc  =  f'ae_huerto_buf{BUFFER_METERS}m_{row["muncode"]}_{year}'
                    task  =  ee.batch.Export.table.toDrive(
                        collection=means,
                        description=desc,
                        folder=f'alpha_earth_huerto_buf{BUFFER_METERS}m',
                        selectors=mun_cols
                    )
                    if safe_start(task, desc):
                        year_tasks +=  1
                print(f"    State {state}: split into individual muns (payload too large)", flush=True)

        total_tasks +=  year_tasks
        print(f"  Submitted {year_tasks} export tasks", flush=True)

    print(f"\n\nAll buffered exports submitted! Total: {total_tasks} tasks", flush=True)
    print(f"  -> alpha_earth_huerto_buf{BUFFER_METERS}m/ on Google Drive", flush=True)


# ===== Main dispatch =====

if __name__ == '__main__':
    modes =  {
        'mexico_mun':             run_mexico_mun,
        'california':             run_california,
        'mexico_adc':             run_mexico_adc,
        'mexico_huerto':          run_mexico_huerto,
        'mexico_huerto_buffered': run_mexico_huerto_buffered,
    }
    if MODE not in modes:
        raise ValueError(f"Unknown MODE '{MODE}'. Choose from: {list(modes.keys())}")

    print(f"Running mode: {MODE}")
    print(f"{'='*60}\n")
    modes[MODE]()
