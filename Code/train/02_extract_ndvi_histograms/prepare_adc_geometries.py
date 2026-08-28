"""
prepare_adc_geometries.py — Convert ADC polygons into EE-ready geometry CSVs.

One-time preprocessing. Reads ADC polygons from adc_shapefile.shp
(the canonical ADC shapefile), drops NaN adcid rows (localities),
applies simplify + buffer(0) for GEE compatibility, and writes two CSVs:
  1. adc_geometries_for_ee.csv  — one row per ADC polygon
  2. muni_geometries_for_ee.csv — one row per municipality (dissolved from ADCs)

Both CSVs include planting_month from harmonic regression phenology data.

Usage:
  python prepare_adc_geometries.py

Run from Maize_prediction/ directory (or set paths accordingly).
Requires: mpc_env (geopandas, shapely)
"""

import os
from collections import defaultdict

import pandas    as pd
import geopandas as gpd
import shapely
from shapely     import to_geojson, force_2d
from shapely.ops import unary_union

# ── Directories ──────────────────────────────────────────
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir  =  os.path.join(proj_dir, "Data")

# ── Inputs ───────────────────────────────────────────────
adc_shp_file         =  os.path.join(home_dir, "Dropbox", "Projects",
                                      "Maize_prediction", "Data", "Shapefiles",
                                      "adc_shapefile.shp")
planting_months_file =  os.path.join(data_dir, "planting_months_harmonic_regression.csv")
# SIAP monthly planting months — fallback for municipalities absent from the
# harmonic-regression file (states 11, 27, 28 are missing there entirely).
siap_plant_file      =  os.path.join(data_dir, "SIAP_monthly", "Output", "max_harv_mnth.dta")

# ── Outputs ──────────────────────────────────────────────
adc_out_file   =  os.path.join(data_dir, "adc_geometries_for_ee.csv")
muni_out_file  =  os.path.join(data_dir, "muni_geometries_for_ee.csv")


def add_zeros(x, n):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


def geom_to_coords(geom):
    """Convert a shapely geometry to GeoJSON coordinate dict string (force 2D)."""
    return to_geojson(force_2d(geom))


# ── Load planting months ─────────────────────────────────
pm =  pd.read_csv(planting_months_file)
pm['muncode'] =  (
    pm['CVE_ENT'].apply(lambda x: add_zeros(x, 2)) +
    pm['CVE_MUN'].apply(lambda x: add_zeros(x, 3))
)
pm_lookup      =  dict(zip(pm['muncode'], pm['planting_month']))
print(f"Loaded {len(pm)} municipalities with harmonic-regression planting months")

# ── Impute planting months for municipalities absent from the harmonic file ──
# The harmonic-regression product covers 29 states (missing 11 Guanajuato,
# 27 Tabasco, 28 Tamaulipas). Fill those from the SIAP monthly planting month
# (max_plants_month_median for maize) — the same quantity the regression
# approximates — so their ADCs are not dropped downstream.
siap_pm =  pd.read_stata(siap_plant_file)
siap_pm =  siap_pm[siap_pm['Crop'].astype(str).str.contains('Ma', na=False)].copy()
siap_pm['muncode'] =  siap_pm['muncode'].astype(str).str.zfill(5)
siap_pm =  siap_pm.dropna(subset=['max_plants_month_median'])
siap_lookup =  (siap_pm.groupby('muncode')['max_plants_month_median']
                       .median().round().astype(int).to_dict())

n_imputed =  0
for mc, mo in siap_lookup.items():
    if mc not in pm_lookup:
        pm_lookup[mc] =  mo
        n_imputed +=  1
print(f"Imputed {n_imputed} municipalities from SIAP monthly planting data")

valid_muncodes =  set(pm_lookup.keys())
print(f"Total municipalities with a planting month: {len(valid_muncodes)}")


# ── Load ADC shapefile ───────────────────────────────────
print(f"Reading {adc_shp_file}...")
gdf =  gpd.read_file(adc_shp_file)
print(f"  Total polygons: {len(gdf)}")

# Drop rows without adcid (localities, not ADCs)
gdf =  gdf.dropna(subset=['adcid'])
print(f"  After dropping NaN adcid: {len(gdf)}")

# Build muncode from est + mun columns
gdf['muncode'] =  gdf['est'] + gdf['mun']

# Filter to municipalities with planting month data
gdf =  gdf[gdf['muncode'].isin(valid_muncodes)].copy()
print(f"  After filtering to valid planting months: {len(gdf)}")
print(f"  Unique municipalities: {gdf['muncode'].nunique()}")

# Simplify + buffer(0) for GEE compatibility (same as ee_alpha_earth_download.py)
print("Simplifying geometries...")
gdf['geometry'] =  gdf['geometry'].apply(lambda x: x.simplify(0.0001))
gdf['geometry'] =  gdf['geometry'].apply(lambda x: x.buffer(0))

# Drop empty geometries
gdf =  gdf[~gdf['geometry'].is_empty].copy()

# Assign planting month
gdf['planting_month'] =  gdf['muncode'].map(pm_lookup)

# Convert to GeoJSON coords
print("Converting to GeoJSON coordinates...")
gdf['coords'] =  gdf['geometry'].apply(geom_to_coords)


# ── Save ADC geometries ─────────────────────────────────
adc_df =  gdf[['adcid', 'muncode', 'planting_month', 'coords']].copy()
adc_df.to_csv(adc_out_file, index=False)
print(f"\nSaved {len(adc_df)} ADC geometries to {adc_out_file}")


# ── Dissolve to municipality level ──────────────────────
print("\nDissolving ADC polygons to municipality level...")

muni_geoms =  defaultdict(list)
for _, row in gdf.iterrows():
    muni_geoms[row['muncode']].append(row['geometry'])

muni_rows =  []
for muncode in sorted(muni_geoms.keys()):
    dissolved =  unary_union(muni_geoms[muncode])
    dissolved =  dissolved.simplify(0.0001)
    dissolved =  dissolved.buffer(0)
    coords    =  geom_to_coords(dissolved)

    muni_rows.append({
        'muncode':        muncode,
        'planting_month': pm_lookup[muncode],
        'coords':         coords,
    })

muni_df =  pd.DataFrame(muni_rows)
muni_df.to_csv(muni_out_file, index=False)
print(f"Saved {len(muni_df)} municipality geometries to {muni_out_file}")
print("\nDone.")
