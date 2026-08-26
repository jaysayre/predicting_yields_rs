"""
clean_harveststat_africa.py
===========================
Clean HarvestStat Africa crop data -> maize yields overlapping AEF (2017-2024).

Joins country-level CSVs with the boundary GeoPackage on fnid, computes annual
maize yields (total production / total area, aggregated across seasons), and
saves panel data at both admin_2 (boundary) and admin_1 (dissolved) levels.

Inputs:
  Data/HarvestStat_Africa/adm_crop_production_*.csv
  Data/HarvestStat_Africa/hvstat_africa_boundary_v1.1.gpkg

Outputs (in Data/HarvestStat_Africa/):
  hvstat_africa_maize_adm2.parquet   -- boundary-level yields (fnid x year)
  hvstat_africa_maize_adm1.parquet   -- admin_1 aggregated yields
  hvstat_africa_fnid_mapping.parquet -- fnid -> admin1_id lookup
  hvstat_africa_adm1_dissolved.gpkg  -- dissolved admin_1 geometries

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 clean_harveststat_africa.py
"""

import os, glob, sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# Configuration
# ============================================================
MIN_YEAR  =  2017   # first year of AEF embeddings
MAX_YEAR  =  2024   # last year of AEF embeddings


# -- Directories -----------------------------------------------
home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
hvstat_dir  =  os.path.join(proj_dir, "Data", "HarvestStat_Africa")


# ============================================================
# 1. Load all country CSVs
# ============================================================
print("Loading HarvestStat Africa CSVs...")
csvs  =  sorted(glob.glob(os.path.join(hvstat_dir, "adm_crop_production_*.csv")))
raw   =  pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
print(f"  {len(raw):,} rows from {len(csvs)} files")


# ============================================================
# 2. Filter for maize products
# ============================================================
maize_mask  =  raw['product'].str.lower().str.contains('maize', na=False)
maize       =  raw[maize_mask].copy()
print(f"\nMaize products found: {sorted(maize['product'].unique())}")
print(f"  {len(maize):,} rows across {maize['country_code'].nunique()} countries")


# ============================================================
# 3. Compute annual yields per fnid (production / area)
# ============================================================
# Sum production across seasons/variants per fnid x year
prod  =  (maize[maize['indicator'] == 'production']
          .groupby(['fnid', 'country_code', 'harvest_year'])['value']
          .sum().reset_index()
          .rename(columns={'value': 'production'}))

# Sum area across seasons/variants per fnid x year
area  =  (maize[maize['indicator'] == 'area']
          .groupby(['fnid', 'country_code', 'harvest_year'])['value']
          .sum().reset_index()
          .rename(columns={'value': 'area'}))

# Fallback: direct yield indicator (where production/area unavailable)
yld_direct  =  (maize[maize['indicator'] == 'yield']
                .groupby(['fnid', 'country_code', 'harvest_year'])['value']
                .mean().reset_index()
                .rename(columns={'value': 'yield_direct'}))

# Merge production + area, compute yield
annual  =  prod.merge(area, on=['fnid', 'country_code', 'harvest_year'], how='outer')
annual  =  annual.merge(yld_direct, on=['fnid', 'country_code', 'harvest_year'], how='outer')

annual['yield']  =  np.where(
    annual['production'].notna() & annual['area'].notna() & (annual['area'] > 0),
    annual['production'] / annual['area'],
    annual['yield_direct']
)
annual  =  annual.drop(columns=['yield_direct'])
annual  =  annual[annual['yield'].notna() & (annual['yield'] > 0)
                   & np.isfinite(annual['yield'])].copy()
annual.rename(columns={'harvest_year': 'year'}, inplace=True)

print(f"\nAnnual yields computed: {len(annual):,} fnid-year obs")


# ============================================================
# 4. Filter to AEF years
# ============================================================
annual  =  annual[(annual['year'] >= MIN_YEAR) & (annual['year'] <= MAX_YEAR)].copy()
print(f"AEF overlap ({MIN_YEAR}-{MAX_YEAR}): {len(annual):,} obs, "
      f"{annual['fnid'].nunique()} fnids, {annual['country_code'].nunique()} countries")

if len(annual) == 0:
    print("\nWARNING: No maize yield data overlaps with AEF years.")
    print("Countries with data outside AEF range:")
    outside  =  raw[maize_mask].groupby('country_code')['harvest_year'].agg(['min', 'max'])
    print(outside.to_string())
    sys.exit(1)


# ============================================================
# 5. Load boundary gpkg & merge
# ============================================================
print("\nLoading boundary gpkg...")
gdf  =  gpd.read_file(os.path.join(hvstat_dir, "hvstat_africa_boundary_v1.1.gpkg"))
print(f"  {len(gdf)} regions, {gdf['country_code'].nunique()} countries")

csv_fnids   =  set(annual['fnid'].unique())
gpkg_fnids  =  set(gdf['fnid'].unique())
overlap     =  csv_fnids & gpkg_fnids
print(f"  fnid overlap: {len(overlap)} of {len(csv_fnids)} CSV fnids "
      f"({len(gpkg_fnids)} in boundary)")

if len(overlap) == 0:
    # Try matching via country_code + admin_1 name
    print("\n  No fnid overlap -- attempting name-based match...")
    gdf_lookup  =  gdf[['fnid', 'country_code', 'admin_1', 'admin_2']].copy()
    gdf_lookup.rename(columns={'admin_1': 'adm1_gpkg', 'admin_2': 'adm2_gpkg',
                               'fnid': 'fnid_gpkg'}, inplace=True)
    # Get CSV admin names
    csv_names   =  (maize[maize['indicator'] == 'yield']
                    [['fnid', 'country_code', 'admin_1', 'admin_2']]
                    .drop_duplicates(subset=['fnid']))
    name_match  =  csv_names.merge(gdf_lookup,
                                    left_on=['country_code', 'admin_1'],
                                    right_on=['country_code', 'adm1_gpkg'],
                                    how='inner')
    print(f"  Name-based matches: {len(name_match)}")

merged  =  annual.merge(
    gdf[['fnid', 'admin_1', 'admin_2']].rename(
        columns={'admin_1': 'adm1_name', 'admin_2': 'adm2_name'}),
    on='fnid', how='inner'
)
merged['admin1_id']  =  merged['country_code'] + '_' + merged['adm1_name']

print(f"\nAfter merge: {len(merged):,} obs, {merged['fnid'].nunique()} regions, "
      f"{merged['country_code'].nunique()} countries")


# ============================================================
# 6. Coverage diagnostics
# ============================================================
print(f"\n{'CC':<5s} {'Regions':>8s} {'Years':>6s} {'Obs':>6s} "
      f"{'Adm2':>5s} {'Adm1':>5s} {'YearRange':>12s}")
print("-" * 50)
for cc in sorted(merged['country_code'].unique()):
    sub     =  merged[merged['country_code'] == cc]
    n_reg   =  sub['fnid'].nunique()
    n_yr    =  sub['year'].nunique()
    n_obs   =  len(sub)
    n_a1    =  sub['admin1_id'].nunique()
    yr_lo   =  sub['year'].min()
    yr_hi   =  sub['year'].max()
    print(f"  {cc:<3s} {n_reg:>8d} {n_yr:>6d} {n_obs:>6d} "
          f"{n_reg:>5d} {n_a1:>5d} {yr_lo:>5d}-{yr_hi:<5d}")


# ============================================================
# 7. Save admin_2 (boundary-level) yields
# ============================================================
adm2_out  =  merged[['fnid', 'country_code', 'admin1_id', 'adm1_name',
                      'adm2_name', 'year', 'yield', 'area', 'production']].copy()
adm2_path =  os.path.join(hvstat_dir, "hvstat_africa_maize_adm2.parquet")
adm2_out.to_parquet(adm2_path, index=False)
print(f"\nSaved: {adm2_path}")
print(f"  {len(adm2_out):,} obs, {adm2_out['fnid'].nunique()} regions")


# ============================================================
# 8. Aggregate to admin_1 level (area-weighted)
# ============================================================
adm1_sub  =  merged.dropna(subset=['production', 'area'])
adm1_sub  =  adm1_sub[adm1_sub['area'] > 0]
adm1  =  adm1_sub.groupby(['admin1_id', 'country_code', 'year']).agg(
    production =  ('production', 'sum'),
    area       =  ('area', 'sum'),
    n_regions  =  ('fnid', 'nunique')
).reset_index()
adm1['yield']  =  adm1['production'] / adm1['area']
adm1  =  adm1[adm1['yield'].notna() & (adm1['yield'] > 0)]

adm1_path  =  os.path.join(hvstat_dir, "hvstat_africa_maize_adm1.parquet")
adm1.to_parquet(adm1_path, index=False)
print(f"\nSaved: {adm1_path}")
print(f"  {len(adm1):,} admin1-year obs, {adm1['admin1_id'].nunique()} groups")

# Admin_1 groups with >1 admin_2 region (needed for within-R2)
multi  =  adm1[adm1['n_regions'] > 1]
print(f"  Admin_1 groups with >1 admin_2: {multi['admin1_id'].nunique()}")


# ============================================================
# 9. Save fnid -> admin1_id mapping
# ============================================================
mapping  =  gdf[['fnid', 'country_code', 'admin_1', 'admin_2']].copy()
mapping['admin1_id']  =  mapping['country_code'] + '_' + mapping['admin_1']
map_path  =  os.path.join(hvstat_dir, "hvstat_africa_fnid_mapping.parquet")
mapping.to_parquet(map_path, index=False)
print(f"\nSaved: {map_path}  ({len(mapping)} regions)")


# ============================================================
# 10. Save dissolved admin_1 geometries
# ============================================================
gdf['admin1_id']  =  gdf['country_code'] + '_' + gdf['admin_1']
adm1_geo  =  gdf.dissolve(by='admin1_id', as_index=False)[
    ['admin1_id', 'country_code', 'geometry']
]
adm1_geo_path  =  os.path.join(hvstat_dir, "hvstat_africa_adm1_dissolved.gpkg")
adm1_geo.to_file(adm1_geo_path, driver="GPKG")
print(f"Saved: {adm1_geo_path}  ({len(adm1_geo)} admin_1 regions)")

print("\nDone.")
