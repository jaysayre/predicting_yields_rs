"""
GEE extraction template: NIR/RED band-level 2D histograms.

Extends ls_ndvi_hists.py to extract 5 paired 2D histograms per municipality-year:
  H1: NDVI_p1 × NDVI_p2   (existing — NDVI trajectory)
  H2: NIR_p1  × NIR_p2    (NIR reflectance trajectory)
  H3: RED_p1  × RED_p2    (RED reflectance trajectory)
  H4: NIR_p1  × RED_p1    (early-season spectral space)
  H5: NIR_p2  × RED_p2    (late-season spectral space)

Each histogram is 32×32 = 1024 bins, yielding 5×1024 = 5120 features per obs.

Usage:
  python3 ls_band_hists.py <bins> <agg_func>

  bins:     number of bins per axis (e.g., 32)
  agg_func: temporal aggregation ('max', 'med', 'mean')

Band ranges for Landsat surface reflectance (after scale_facts):
  NIR:  [0.0, 0.6]
  RED:  [0.0, 0.3]
  NDVI: [0.2, 1.0]
"""
from ast import literal_eval
import os
import sys

import pandas as pd
import ee

# ── Command-line arguments ───────────────────────────────
bins     = int(sys.argv[1])      # bins per axis (e.g., 32)
agg_func = sys.argv[2]           # 'max', 'med', or 'mean'

# Band ranges
ndvi_min, ndvi_max = 0.2, 1.0
nir_min,  nir_max  = 0.0, 0.6
red_min,  red_max  = 0.0, 0.3

ee.Initialize()


# ── Shared helpers (from ls_ndvi_hists.py) ───────────────

def scale_facts(img):
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, overwrite=True).addBands(thermalBands, overwrite=True)


def cloud_mask(img):
    qa   = img.select(['QA_PIXEL'])
    mask = ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4))
    )
    return img.updateMask(mask)


def assign_bin(img, val_min, val_max, n_bins):
    """Assign each pixel to a bin index in [0, n_bins-1]."""
    eps = (val_max - val_min) / 1_000_000
    img = img.max(val_min + eps)
    img = img.min(val_max - eps)
    img = img.subtract(val_min)
    return img.divide((val_max - val_min) / n_bins).floor()


def calc_1d_bin(binned, n_bins):
    """Encode 2D bin pair as 1D index: bin_x + bin_y * n_bins."""
    out = binned.select(0).add(binned.select(1).multiply(n_bins))
    return out


def add_zeros(x, n):
    x = str(x)
    while len(x) < n:
        x = "0" + x
    return x


# ── Band extraction ──────────────────────────────────────

def extract_bands(img, red_name, nir_name):
    """Extract scaled, cloud-masked NIR, RED, and NDVI bands."""
    img  = scale_facts(img)
    img  = cloud_mask(img)
    nir  = img.select(nir_name).rename('nir')
    red  = img.select(red_name).rename('red')
    ndvi = img.normalizedDifference([nir_name, red_name]).rename('ndvi')
    return nir.addBands(red).addBands(ndvi)


def aggregate_collection(collection, func):
    """Temporally aggregate an image collection."""
    collection = ee.Algorithms.If(
        collection.size(),
        collection,
        ee.ImageCollection([ee.Image.constant(0).rename('nir')
                            .addBands(ee.Image.constant(0).rename('red'))
                            .addBands(ee.Image.constant(0).rename('ndvi'))])
    )
    collection = ee.ImageCollection(collection)

    if func == "max":
        return collection.max()
    elif func == "med":
        return collection.median()
    elif func == "mean":
        return collection.mean()
    elif func == "min":
        return collection.min()
    else:
        raise ValueError(f"agg_func must be 'max', 'med', or 'mean', got '{func}'")


def get_period_bands(muni, start_date, end_date):
    """Get aggregated NIR, RED, NDVI for a municipality and date range."""
    geom = muni.geometry()

    ls5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(geom).filterDate(start_date, end_date)
    ls7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(geom).filterDate(start_date, end_date)
    ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(geom).filterDate(start_date, end_date)

    # Landsat 5/7: RED=SR_B3, NIR=SR_B4
    ls5_bands = ee.Algorithms.If(ls5.size(),
                                  ls5.map(lambda x: extract_bands(x, 'SR_B3', 'SR_B4')),
                                  ee.ImageCollection([]))
    ls7_bands = ee.Algorithms.If(ls7.size(),
                                  ls7.map(lambda x: extract_bands(x, 'SR_B3', 'SR_B4')),
                                  ee.ImageCollection([]))
    # Landsat 8: RED=SR_B4, NIR=SR_B5
    ls8_bands = ee.Algorithms.If(ls8.size(),
                                  ls8.map(lambda x: extract_bands(x, 'SR_B4', 'SR_B5')),
                                  ee.ImageCollection([]))

    all_bands = (ee.ImageCollection(ls5_bands)
                 .merge(ee.ImageCollection(ls7_bands))
                 .merge(ee.ImageCollection(ls8_bands)))

    agg = aggregate_collection(all_bands, agg_func)
    return ee.Image(agg)


def muni_band_hists(muni):
    """Compute 5 paired 2D histograms for a municipality-year.

    Returns the feature with 5 histogram properties set.
    """
    # Get period 1 and period 2 bands
    p1 = get_period_bands(muni, muni.get('start_p1'), muni.get('end_p1'))
    p1 = ee.Image(p1).rename(['nir_p1', 'red_p1', 'ndvi_p1'])

    p2 = get_period_bands(muni, muni.get('start_p2'), muni.get('end_p2'))
    p2 = ee.Image(p2).rename(['nir_p2', 'red_p2', 'ndvi_p2'])

    all_bands = p1.addBands(p2)
    geom      = muni.geometry()

    # ── H1: NDVI_p1 × NDVI_p2 (existing) ────────────────
    ndvi_b1 = assign_bin(all_bands.select('ndvi_p1'), ndvi_min, ndvi_max, bins)
    ndvi_b2 = assign_bin(all_bands.select('ndvi_p2'), ndvi_min, ndvi_max, bins)
    h1_1d   = calc_1d_bin(ndvi_b1.addBands(ndvi_b2), bins).rename('h1_ndvi')

    # ── H2: NIR_p1 × NIR_p2 ─────────────────────────────
    nir_b1 = assign_bin(all_bands.select('nir_p1'), nir_min, nir_max, bins)
    nir_b2 = assign_bin(all_bands.select('nir_p2'), nir_min, nir_max, bins)
    h2_1d  = calc_1d_bin(nir_b1.addBands(nir_b2), bins).rename('h2_nir')

    # ── H3: RED_p1 × RED_p2 ─────────────────────────────
    red_b1 = assign_bin(all_bands.select('red_p1'), red_min, red_max, bins)
    red_b2 = assign_bin(all_bands.select('red_p2'), red_min, red_max, bins)
    h3_1d  = calc_1d_bin(red_b1.addBands(red_b2), bins).rename('h3_red')

    # ── H4: NIR_p1 × RED_p1 (early spectral space) ──────
    h4_1d = calc_1d_bin(nir_b1.addBands(red_b1), bins).rename('h4_nir_red_p1')

    # ── H5: NIR_p2 × RED_p2 (late spectral space) ───────
    h5_1d = calc_1d_bin(nir_b2.addBands(red_b2), bins).rename('h5_nir_red_p2')

    # Combine all 5 channels into one image
    combined = h1_1d.addBands(h2_1d).addBands(h3_1d).addBands(h4_1d).addBands(h5_1d)

    # Compute histograms for all 5 channels at once
    hist = combined.reduceRegion(
        ee.Reducer.fixedHistogram(0, bins**2, bins**2),
        geom,
        30,
        maxPixels=500_000_000
    )

    return muni.set(hist)


# ── Load municipality geometries and phenology dates ─────

munis = pd.read_csv("../processed_data/munis_agmask.csv")
munis['coords']   = munis['coords'].apply(literal_eval)
munis['eeGeom']   = munis['coords'].apply(ee.Geometry)

peak_dates = pd.read_csv("../processed_data/muni_ndvi_peak_dates.csv")
peak_dates['CVE_ENT'] = peak_dates['CVE_ENT'].apply(lambda x: add_zeros(x, 2))
peak_dates['CVE_MUN'] = peak_dates['CVE_MUN'].apply(lambda x: add_zeros(x, 3))
peak_dates['muncode'] = (peak_dates['CVE_ENT'] + peak_dates['CVE_MUN']).apply(int)

munis = pd.merge(munis, peak_dates, on=['muncode'])

munis['eeFeature'] = munis.apply(lambda x: ee.Feature(x['eeGeom'], {
    'muncode':  x['muncode'],
    'start_p1': x['start_p1'],
    'end_p1':   x['end_p1'],
    'start_p2': x['start_p2'],
    'end_p2':   x['end_p2'],
}), axis=1)

print("Municipalities uploaded")

# ── Export tasks ─────────────────────────────────────────

n    = 25  # batch size (same as ls_ndvi_hists.py)
cols = ["muncode", "h1_ndvi", "h2_nir", "h3_red", "h4_nir_red_p1", "h5_nir_red_p2"]

ents = munis['CVE_ENT'].unique()
for e in range(len(ents)):
    ent = ents[e]

    for year in range(2003, 2025):
        munis_year = munis.loc[munis['year'] == year]
        ent_munis  = munis_year.loc[munis_year['CVE_ENT'] == ent]

        for i in range((len(ent_munis.index) // n) + 1):
            batch = ent_munis['eeFeature'].iloc[i * n : min((i + 1) * n, len(ent_munis.index) + 1)]

            if len(batch) == 0:
                continue

            ee_munis = ee.FeatureCollection(batch.to_list())

            print(f"State: {ent}, Group: {i}, Year: {year}")
            munis_hists = ee_munis.map(lambda x: muni_band_hists(x))

            folder = f"muni_band_hists_{bins}_{agg_func}"
            desc   = f"muni_band_hist_{ent}_{i}_{year}"

            task = ee.batch.Export.table.toDrive(
                collection=munis_hists,
                folder=folder,
                description=desc,
                selectors=cols
            )
            task.start()
