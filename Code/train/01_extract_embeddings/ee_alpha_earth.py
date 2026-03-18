from ast import literal_eval

import ee
import pandas as pd
import shapely.geometry as shg
from shapely import to_geojson
import geopandas as gpd

ee.Authenticate(force=True)
ee.Initialize(project='ee-joeldferg')

def add_zeros(x, n=2):
    x = str(x)
    while len(x) < n:
        x = '0' + x
    return x

muns = gpd.read_file("../raw_data/mgm2007/Municipios_2007.shp")
muns['geometry'] = muns['geometry'].apply(lambda x: x.simplify(100)) # Simplify geometries
muns = muns.to_crs("EPSG:4326")
muns['geometry'] = muns['geometry'].apply(lambda x: x.buffer(0))
muns['coords'] = muns['geometry'].apply(lambda x: literal_eval(to_geojson(x)))
muns['eeGeom'] = muns['coords'].apply(ee.Geometry)
muns['eeFeature'] = muns.apply(lambda row: ee.Feature(row['eeGeom'], {'CVE_ENT': row['CVE_ENT'],'CVE_MUN': row['CVE_MUN']}), axis=1)



alpha_earth = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

cols = [f"A{add_zeros(x)}" for x in range(0,64)]+['CVE_ENT','CVE_MUN','year']

for state in muns['CVE_ENT'].unique():
    mun_coll = ee.FeatureCollection(muns.loc[muns['CVE_ENT'] == state, 'eeFeature'].to_list())
    print(mun_coll.size().getInfo())
    state_coll = alpha_earth.filterBounds(mun_coll.geometry())
    wc = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(mun_coll.geometry())
    agland = wc.map(lambda x : x.eq(40).rename('ag_area')).max()
    for year in range(2017, 2025):
        print(f"Processing state {state} for year {year}...")
        

        img = state_coll.filterDate(f'{year}-01-01', f'{year}-12-31').mean()
        img = img.updateMask(agland)
        mun_means = img.reduceRegions(
            collection=mun_coll,
            reducer=ee.Reducer.mean(),
            scale=10
        )

        mun_means = mun_means.map(lambda f: f.set('year', year))
        task = ee.batch.Export.table.toDrive(
            collection=mun_means,
            description=f'alpha_earth_agmask_state_{state}_year_{year}',
            folder='alpha_earth_agmask',
            selectors=cols)
        task.start()

       