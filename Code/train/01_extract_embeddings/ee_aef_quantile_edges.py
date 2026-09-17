"""
ee_aef_quantile_edges.py
========================
Compute per-dimension quantile bin edges for AEF embeddings over Mexican
cropland (ESA WorldCover class 40). Samples pixels state by state from the
2022 AEF mosaic, pools them, and stores the 7 interior edges per dim that
define 8 equal-mass bins.

Motivation: the fixed-width bins in 6_ee_alpha_earth_binned_hist.py span
[-0.8, 0.8] but 99.4% of cropland pixel mass falls in [-0.4, 0.4] (median
effective bins per dim ~2). Quantile edges restore true 8-level resolution.

Output: ~/Dropbox/Projects/Maize_prediction/Data/alpha_earth/
        aef_quantile_bin_edges_8.json   ({"A00": [7 edges], ...})

Usage:
  laptop:  ~/miniforge3/envs/ml_cuda/bin/python ee_aef_quantile_edges.py
  server:  conda activate ML_env && python3 ee_aef_quantile_edges.py
"""

import os
import json
from ast import literal_eval

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import to_geojson

# ============================================================
# Configuration
# ============================================================
YEAR        =  2022     # sample year (edges held fixed across all years)
N_BINS      =  8
PX_PER_ST   =  20000    # sampled locations per state (masked -> fewer valid returns)
SCALE       =  10
SEED        =  42
# ============================================================

home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
mun_shp   =  os.path.join(home_dir, "Dropbox", "Projects", "Avocado_Deforestation",
                          "Data", "raw", "spatial", "Municipality_shp", "MUNICIPIOS.shp")
out_json  =  os.path.join(proj_dir, "Data", "alpha_earth", "aef_quantile_bin_edges_8.json")

feat_names =  [f"A{d:02d}" for d in range(64)]

try:
    ee.Initialize(project='avocadoyieldsdeforestation')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

alpha_earth =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

print("Loading municipality shapefile (state outlines)...")
muns =  gpd.read_file(mun_shp)
muns['geometry'] =  muns['geometry'].apply(lambda x: x.simplify(100))
muns =  muns.to_crs("EPSG:4326")
muns['geometry'] =  muns['geometry'].apply(lambda x: x.buffer(0))

chunks =  []
for state in sorted(muns['CVE_ENT'].unique()):
    sm    =  muns.loc[muns['CVE_ENT'] == state]
    geoms =  [ee.Geometry(literal_eval(to_geojson(g))) for g in sm.geometry]
    region =  ee.FeatureCollection([ee.Feature(g) for g in geoms]).geometry()

    wc     =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(region)
    agland =  wc.map(lambda x: x.eq(40).rename('ag')).max()
    img    =  (alpha_earth.filterBounds(region)
                          .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
                          .mean().updateMask(agland))

    sample =  img.sample(region=region, scale=SCALE, numPixels=PX_PER_ST,
                         seed=SEED, tileScale=4, geometries=False)
    try:
        pages =  ee.data.computeFeatures({
            'expression': sample,
            'fileFormat': 'PANDAS_DATAFRAME',
        })
        df =  pd.concat(list(pages)) if not isinstance(pages, pd.DataFrame) else pages
    except Exception as e:
        print(f"  state {state}: FAILED ({e})")
        continue
    df =  df[[c for c in df.columns if c in feat_names]]
    chunks.append(df)
    print(f"  state {state}: {len(df):,} pixels")

pix =  pd.concat(chunks, ignore_index=True)
print(f"\nTotal sampled cropland pixels: {len(pix):,}")

qs    =  np.linspace(0, 1, N_BINS + 1)[1:-1]     # 7 interior quantiles
edges =  {}
for f in feat_names:
    v =  pix[f].dropna().values
    edges[f] =  [round(float(x), 6) for x in np.quantile(v, qs)]

with open(out_json, 'w') as fh:
    json.dump({'year': YEAR, 'n_bins': N_BINS, 'n_pixels': int(len(pix)),
               'edges': edges}, fh, indent=1)
print(f"Wrote {out_json}")
print("example A00 edges:", edges['A00'])
