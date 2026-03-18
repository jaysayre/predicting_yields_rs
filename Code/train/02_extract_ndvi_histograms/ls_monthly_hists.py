"""
ls_monthly_hists.py — Extract monthly spectral-index histograms from Landsat
Phase 2 redesign: monthly composites across the growing season instead of
2-period max composites.

Output per municipality-year:
  6 indices × 8 months × 32 bins = 6 separate 1D histograms per month
  Stored as columns: {index}_{month} (e.g., ndvi_m0, ndvi_m1, ..., ndvi_m7)

Indices:
  NDVI  — Normalized Difference Vegetation Index    [0.0, 1.0]
  GCVI  — Green Chlorophyll Vegetation Index        [0.0, 12.0]
  NDTI  — Normalized Difference Tillage Index       [0.0, 0.6]
  EVI   — Enhanced Vegetation Index                 [0.0, 1.0]
  NDWI  — Normalized Difference Water Index         [-0.5, 0.5]
  BSI   — Bare Soil Index                           [-0.5, 0.5]

Usage:
  python ls_monthly_hists.py <bins> <n_months>
  python ls_monthly_hists.py 32 8

Requires: Earth Engine Python API authenticated, phenology data.
"""

import sys
import os
from ast import literal_eval

import pandas as pd
import ee

bins     =  int(sys.argv[1])  # Number of histogram bins (e.g. 32)
n_months =  int(sys.argv[2])  # Number of monthly composites (e.g. 8)

ee.Initialize()


# ── Index ranges ──────────────────────────────────────────

INDEX_RANGES =  {
    'ndvi': (0.0, 1.0),
    'gcvi': (0.0, 12.0),
    'ndti': (0.0, 0.6),
    'evi':  (0.0, 1.0),
    'ndwi': (-0.5, 0.5),
    'bsi':  (-0.5, 0.5),
}

INDEX_NAMES =  list(INDEX_RANGES.keys())


# ── Landsat helpers ───────────────────────────────────────

def scale_facts(img):
    """Apply Landsat Collection 2 scale factors."""
    optical =  img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal =  img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)


def cloud_mask(img):
    """Mask cloudy pixels using QA_PIXEL bits (same across LS 5/7/8)."""
    qa   =  img.select(['QA_PIXEL'])
    mask =  ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4))
    )
    return img.updateMask(mask)


def prep_image(img):
    """Scale and cloud-mask a Landsat image."""
    return cloud_mask(scale_facts(img))


# ── Spectral index functions ─────────────────────────────

def calc_indices_57(img):
    """Compute all 6 indices for Landsat 5/7 band names."""
    img  =  prep_image(img)
    blue, green, red, nir, swir1, swir2 = (
        img.select('SR_B1'), img.select('SR_B2'), img.select('SR_B3'),
        img.select('SR_B4'), img.select('SR_B5'), img.select('SR_B7')
    )
    ndvi =  nir.subtract(red).divide(nir.add(red)).rename('ndvi')
    gcvi =  nir.divide(green).subtract(1).rename('gcvi')
    ndti =  swir1.subtract(swir2).divide(swir1.add(swir2)).rename('ndti')
    evi  =  nir.subtract(red).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).multiply(2.5).rename('evi')
    ndwi =  green.subtract(nir).divide(green.add(nir)).rename('ndwi')
    bsi  =  (swir1.add(red).subtract(nir).subtract(blue)).divide(
        swir1.add(red).add(nir).add(blue)
    ).rename('bsi')
    return ndvi.addBands(gcvi).addBands(ndti).addBands(evi).addBands(ndwi).addBands(bsi)


def calc_indices_8(img):
    """Compute all 6 indices for Landsat 8/9 band names."""
    img  =  prep_image(img)
    blue, green, red, nir, swir1, swir2 = (
        img.select('SR_B2'), img.select('SR_B3'), img.select('SR_B4'),
        img.select('SR_B5'), img.select('SR_B6'), img.select('SR_B7')
    )
    ndvi =  nir.subtract(red).divide(nir.add(red)).rename('ndvi')
    gcvi =  nir.divide(green).subtract(1).rename('gcvi')
    ndti =  swir1.subtract(swir2).divide(swir1.add(swir2)).rename('ndti')
    evi  =  nir.subtract(red).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).multiply(2.5).rename('evi')
    ndwi =  green.subtract(nir).divide(green.add(nir)).rename('ndwi')
    bsi  =  (swir1.add(red).subtract(nir).subtract(blue)).divide(
        swir1.add(red).add(nir).add(blue)
    ).rename('bsi')
    return ndvi.addBands(gcvi).addBands(ndti).addBands(evi).addBands(ndwi).addBands(bsi)


# ── Monthly composite builder ────────────────────────────

def monthly_composite(muni_geom, start_date, end_date):
    """
    Build a median composite of all 6 indices for a single month window.
    Merges Landsat 5, 7, and 8 imagery. Uses median (more robust than max).
    """
    ls5 =  ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(muni_geom).filterDate(start_date, end_date)
    ls7 =  ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(muni_geom).filterDate(start_date, end_date)
    ls8 =  ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(muni_geom).filterDate(start_date, end_date)

    ls5_idx =  ee.Algorithms.If(ls5.size(), ls5.map(calc_indices_57), ee.ImageCollection([]))
    ls7_idx =  ee.Algorithms.If(ls7.size(), ls7.map(calc_indices_57), ee.ImageCollection([]))
    ls8_idx =  ee.Algorithms.If(ls8.size(), ls8.map(calc_indices_8),  ee.ImageCollection([]))

    all_idx =  ee.ImageCollection(ls5_idx).merge(ee.ImageCollection(ls7_idx)).merge(ee.ImageCollection(ls8_idx))

    # Fallback: if no images, return constant zeros
    all_idx =  ee.Algorithms.If(
        all_idx.size(),
        all_idx.median(),
        ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(INDEX_NAMES)
    )
    return ee.Image(all_idx)


# ── Bin assignment ────────────────────────────────────────

def assign_bin(img, vmin, vmax, n_bins):
    """Clamp image to [vmin, vmax] and assign integer bin [0, n_bins-1]."""
    eps =  (vmax - vmin) / 1_000_000
    img =  img.max(vmin + eps).min(vmax - eps)
    return img.subtract(vmin).divide((vmax - vmin) / n_bins).floor()


# ── Per-municipality monthly histogram extraction ────────

def muni_monthly_hists(muni):
    """
    For a single municipality feature, compute 1D histograms of each
    spectral index for each of n_months monthly windows.

    Municipality feature properties must include:
      - 'gs_start': growing season start date (ISO string, e.g. '2010-04-01')
      - 'muncode': municipality code

    Returns the feature with histogram properties set:
      {index}_m{month}: list of [bin_start, count] pairs
    """
    geom     =  muni.geometry()
    gs_start =  ee.Date(muni.get('gs_start'))

    result =  muni

    # Loop over months (server-side iteration via ee.List.sequence + iterate)
    def add_month_hists(month_idx, feat):
        feat      =  ee.Feature(feat)
        month_idx =  ee.Number(month_idx)
        m_start   =  gs_start.advance(month_idx, 'month')
        m_end     =  gs_start.advance(month_idx.add(1), 'month')

        composite =  monthly_composite(geom, m_start, m_end)

        def add_index_hist(idx_name, feat_inner):
            feat_inner =  ee.Feature(feat_inner)
            idx_name   =  ee.String(idx_name)

            # Get the range for this index (encoded as properties on the muni feature)
            # We pass ranges as constants since ee.Dictionary isn't ideal here
            # Instead, compute all bins in one image and reduce

            band_img  =  composite.select(idx_name)
            binned    =  assign_bin_server(band_img, idx_name)

            hist =  binned.reduceRegion(
                ee.Reducer.fixedHistogram(0, bins, bins),
                geom,
                30,
                maxPixels=500_000_000
            )

            prop_name =  idx_name.cat('_m').cat(month_idx.format('%d'))
            return feat_inner.set(prop_name, hist.get(idx_name))

        feat =  ee.List(INDEX_NAMES).iterate(add_index_hist, feat)
        return feat

    result =  ee.List.sequence(0, n_months - 1).iterate(add_month_hists, result)
    return ee.Feature(result)


def assign_bin_server(img, idx_name):
    """
    Server-side bin assignment using precomputed ranges.
    Since we can't easily do dictionary lookups in EE map functions,
    we chain conditionals for each index.
    """
    # Build a lookup via ee.Algorithms.If chains
    def make_binned(name, vmin, vmax):
        eps =  (vmax - vmin) / 1_000_000
        clamped =  img.max(vmin + eps).min(vmax - eps)
        return clamped.subtract(vmin).divide((vmax - vmin) / bins).floor().rename(name)

    # Chain: check idx_name against each known index
    binned =  ee.Image(
        ee.Algorithms.If(idx_name.equals('ndvi'), make_binned('ndvi', 0.0, 1.0),
        ee.Algorithms.If(idx_name.equals('gcvi'), make_binned('gcvi', 0.0, 12.0),
        ee.Algorithms.If(idx_name.equals('ndti'), make_binned('ndti', 0.0, 0.6),
        ee.Algorithms.If(idx_name.equals('evi'),  make_binned('evi',  0.0, 1.0),
        ee.Algorithms.If(idx_name.equals('ndwi'), make_binned('ndwi', -0.5, 0.5),
                                                   make_binned('bsi',  -0.5, 0.5)
        )))))
    )
    return binned


# ── Simpler approach: pre-compute all months × indices ───
# The nested iterate above can hit GEE compute limits for many features.
# Alternative: flatten to one reduceRegion call per municipality-year
# by stacking all months × indices into a single multi-band image.

def muni_monthly_hists_flat(muni):
    """
    Flat approach: build one multi-band image with all months × indices,
    then do a single reduceRegion per municipality.

    Each band is named {index}_m{i} and contains the bin assignment.
    """
    geom     =  muni.geometry()
    gs_start =  ee.Date(muni.get('gs_start'))

    # Build one band per (month, index) pair
    all_bands =  ee.Image.constant(0).rename('_dummy')

    for m in range(n_months):
        m_start =  gs_start.advance(m, 'month')
        m_end   =  gs_start.advance(m + 1, 'month')
        comp    =  monthly_composite(geom, m_start, m_end)

        for idx_name, (vmin, vmax) in INDEX_RANGES.items():
            band     =  comp.select(idx_name)
            binned   =  assign_bin(band, vmin, vmax, bins)
            band_name =  f"{idx_name}_m{m}"
            all_bands =  all_bands.addBands(binned.rename(band_name))

    # Drop the dummy band
    band_names =  [f"{idx}_m{m}" for m in range(n_months) for idx in INDEX_NAMES]
    all_bands  =  all_bands.select(band_names)

    # Compute histograms for all bands in one reduceRegion call
    reducer =  ee.Reducer.fixedHistogram(0, bins, bins)
    # Apply same reducer to each band
    multi_reducer =  reducer
    for _ in range(len(band_names) - 1):
        multi_reducer =  multi_reducer.combine(reducer)

    hists =  all_bands.reduceRegion(
        multi_reducer,
        geom,
        30,
        maxPixels=500_000_000
    )

    return muni.set(hists)


# ── Data loading ──────────────────────────────────────────

def add_zeros(x, n):
    x =  str(x)
    while len(x) < n:
        x =  "0" + x
    return x


# Municipality geometries with ag mask
munis            =  pd.read_csv("../processed_data/munis_agmask.csv")
munis['coords']  =  munis['coords'].apply(literal_eval)
munis['eeGeom']  =  munis['coords'].apply(ee.Geometry)

# Phenology: peak dates → derive growing season start
peak_dates             =  pd.read_csv("../processed_data/muni_ndvi_peak_dates.csv")
peak_dates['CVE_ENT']  =  peak_dates['CVE_ENT'].apply(lambda x: add_zeros(x, 2))
peak_dates['CVE_MUN']  =  peak_dates['CVE_MUN'].apply(lambda x: add_zeros(x, 3))
peak_dates['muncode']  =  (peak_dates['CVE_ENT'] + peak_dates['CVE_MUN']).apply(int)

# Planting months (from harmonic regression)
plant_months            =  pd.read_csv("../processed_data/planting_months_harmonic_regression.csv")
plant_months['CVE_ENT'] =  plant_months['CVE_ENT'].apply(lambda x: add_zeros(x, 2))
plant_months['CVE_MUN'] =  plant_months['CVE_MUN'].apply(lambda x: add_zeros(x, 3))
plant_months['muncode'] =  (plant_months['CVE_ENT'] + plant_months['CVE_MUN']).apply(int)

# Merge phenology data
munis =  pd.merge(munis, peak_dates[['muncode', 'year', 'peak_day_of_year']], on=['muncode'])
munis =  pd.merge(munis, plant_months[['muncode', 'planting_month']], on='muncode')

# Compute growing season start: planting_month of the given year
# The growing season runs from planting month for n_months months
munis['gs_start'] =  munis.apply(
    lambda x: f"{int(x['year'])}-{int(x['planting_month']):02d}-01",
    axis=1
)

# Build EE features with growing season info
munis['eeFeature'] =  munis.apply(
    lambda x: ee.Feature(x['eeGeom'], {
        'muncode':  x['muncode'],
        'gs_start': x['gs_start'],
    }),
    axis=1
)

print("Municipalities uploaded")
print(f"Bins: {bins}, Months: {n_months}, Indices: {len(INDEX_NAMES)}")
print(f"Output columns per muni-year: {len(INDEX_NAMES) * n_months}")

# ── Export loop ───────────────────────────────────────────

# Output column names
cols =  ["muncode"] + [f"{idx}_m{m}" for m in range(n_months) for idx in INDEX_NAMES]

n_batch =  10  # Fewer per batch since each muni now has more computation

ents =  munis['CVE_ENT'].unique()

for e in range(len(ents)):
    ent =  ents[e]

    for year in range(2003, 2025):
        munis_year =  munis.loc[munis['year'] == year]
        ent_munis  =  munis_year.loc[munis_year['CVE_ENT'] == ent]

        for i in range(0, (len(ent_munis.index) // n_batch) + 1):
            batch =  ent_munis['eeFeature'].iloc[i * n_batch : (i + 1) * n_batch].to_list()
            if len(batch) == 0:
                continue

            ee_munis =  ee.FeatureCollection(batch)

            print(f"State: {ent}, Group: {i}, Year: {year}")
            munis_hists =  ee_munis.map(lambda x: muni_monthly_hists_flat(x))

            folder =  f"muni_monthly_hists_{bins}bins_{n_months}mo"
            desc   =  f"monthly_hist_{ent}_{i}_{year}"

            task =  ee.batch.Export.table.toDrive(
                collection=munis_hists,
                folder=folder,
                description=desc,
                selectors=cols
            )
            task.start()
