"""
Resubmit 5 failed 3-period ADC tasks for state 18 (Nayarit) with smaller batch size.

Failed tasks (computation timeout at batch_n=500):
  3period_hist_adc_18_0_2017  (batch 0 = ADCs 0-499, year 2017)
  3period_hist_adc_18_0_2019  (batch 0 = ADCs 0-499, year 2019)
  3period_hist_adc_18_0_2022  (batch 0 = ADCs 0-499, year 2022)
  3period_hist_adc_18_1_2022  (batch 1 = ADCs 500-999, year 2022)
  3period_hist_adc_18_4_2015  (batch 4 = ADCs 2000-2293, year 2015)

Resubmits with batch_n=250 so each old batch becomes 2 sub-batches.
"""

from ast import literal_eval
import os
import time

import pandas as pd
import ee

ee.Initialize()

# ── Config ───────────────────────────────────────────────
bins       =  16
level      =  "adc"
id_col     =  "adcid"
folder     =  "adc_3period_hists_16bins"
batch_n    =  250  # halved from 500
OLD_BATCH  =  500

FAILED =  [
    (0, 2017),
    (0, 2019),
    (0, 2022),
    (1, 2022),
    (4, 2015),
]

# Already submitted in previous run:
SKIP =  {("1b", 2022)}  # 3period_hist_adc_18_1b_2022

QUEUE_LIMIT =  3000

# ── Band names by satellite ──────────────────────────────
BAND_MAP =  {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}

NDVI_MIN =  0.0
NDVI_MAX =  1.0


def scale_facts(img):
    opticalBands =  img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands =  img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, overwrite=True).addBands(thermalBands, overwrite=True)


def cloud_mask(img):
    qa   =  img.select(['QA_PIXEL'])
    mask =  ee.Image.constant(1).subtract(
        qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4))
    )
    return img.updateMask(mask)


def calc_ndvi(img, red, nir):
    img  =  scale_facts(img)
    img  =  cloud_mask(img)
    ndvi =  img.normalizedDifference([nir, red]).rename('ndvi')
    return ndvi


def get_ndvi_composite(geom, start_date, end_date):
    collections =  []
    for sat, bands in BAND_MAP.items():
        col =  (ee.ImageCollection(sat)
                .filterBounds(geom)
                .filterDate(start_date, end_date))
        ndvi =  ee.Algorithms.If(
            col.size(),
            col.map(lambda x: calc_ndvi(x, bands['red'], bands['nir'])),
            ee.ImageCollection([])
        )
        collections.append(ee.ImageCollection(ndvi))
    merged =  collections[0].merge(collections[1]).merge(collections[2])
    merged =  ee.ImageCollection(merged)
    merged =  ee.Algorithms.If(
        merged.size(),
        merged,
        ee.ImageCollection([ee.Image.constant(0).rename('ndvi')])
    )
    return ee.ImageCollection(merged).median()


def assign_bin(img, vmin, vmax, n_bins):
    eps =  (vmax - vmin) / 1_000_000
    img =  img.max(vmin + eps)
    img =  img.min(vmax - eps)
    img =  img.subtract(vmin)
    return img.divide((vmax - vmin) / n_bins).floor()


def calc_1d_bin(bin_a, bin_b, n_bins):
    return bin_a.add(bin_b.multiply(n_bins)).rename('hist')


def compute_3period_hist(feature):
    geom           =  feature.geometry()
    planting_month =  ee.Number(feature.get('planting_month'))
    gs_year        =  ee.Number(feature.get('gs_year'))
    gs_start       =  ee.Date.fromYMD(gs_year, planting_month, 1)

    p1_start =  gs_start
    p1_end   =  gs_start.advance(2, 'month')
    p2_start =  p1_end
    p2_end   =  gs_start.advance(4, 'month')
    p3_start =  p2_end
    p3_end   =  gs_start.advance(6, 'month')

    ndvi_p1 =  get_ndvi_composite(geom, p1_start, p1_end)
    ndvi_p2 =  get_ndvi_composite(geom, p2_start, p2_end)
    ndvi_p3 =  get_ndvi_composite(geom, p3_start, p3_end)

    bin_p1 =  assign_bin(ndvi_p1.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)
    bin_p2 =  assign_bin(ndvi_p2.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)
    bin_p3 =  assign_bin(ndvi_p3.select('ndvi'), NDVI_MIN, NDVI_MAX, bins)

    hist_1d_p12 =  calc_1d_bin(bin_p1, bin_p2, bins).rename('hist_p12')
    hist_1d_p23 =  calc_1d_bin(bin_p2, bin_p3, bins).rename('hist_p23')
    hist_1d_p13 =  calc_1d_bin(bin_p1, bin_p3, bins).rename('hist_p13')

    combined =  hist_1d_p12.addBands(hist_1d_p23).addBands(hist_1d_p13)
    n_bins_sq =  bins * bins
    hist =  combined.reduceRegion(
        ee.Reducer.fixedHistogram(0, n_bins_sq, n_bins_sq),
        geom, 30, maxPixels=500_000_000
    )
    return feature.set(hist)


# ── Load state 18 geometries ─────────────────────────────
home_dir  =  os.path.expanduser("~")
data_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
geom_file =  os.path.join(data_dir, "adc_geometries_for_ee.csv")

geoms =  pd.read_csv(geom_file)
geoms['coords'] =  geoms['coords'].apply(literal_eval)
geoms['state']  =  geoms['muncode'].apply(lambda x: str(x)[:2])

state_geoms =  geoms.loc[geoms['state'] == '18'].copy()
print(f"State 18: {len(state_geoms)} ADCs")

state_geoms['eeGeom'] =  state_geoms['coords'].apply(ee.Geometry)

# ── Resubmit failed tasks ────────────────────────────────
submitted =  0
for old_batch_idx, year in FAILED:
    old_start =  old_batch_idx * OLD_BATCH
    old_end   =  min(old_start + OLD_BATCH, len(state_geoms))
    old_batch =  state_geoms.iloc[old_start:old_end].copy()

    print(f"\nOld batch {old_batch_idx}, year {year}: {len(old_batch)} ADCs → sub-batches of {batch_n}")

    old_batch['eeFeature'] =  old_batch.apply(
        lambda row: ee.Feature(row['eeGeom'], {
            id_col:           row[id_col],
            'planting_month': int(row['planting_month']),
            'gs_year':        year,
        }),
        axis=1
    )

    n_sub =  (len(old_batch) + batch_n - 1) // batch_n
    for s in range(n_sub):
        sub     =  old_batch.iloc[s * batch_n : (s + 1) * batch_n]
        suffix  =  chr(ord('a') + s)
        sub_idx =  f"{old_batch_idx}{suffix}"
        desc    =  f"3period_hist_{level}_18_{sub_idx}_{year}"

        ee_coll =  ee.FeatureCollection(sub['eeFeature'].to_list())
        result  =  ee_coll.map(compute_3period_hist)
        cols    =  [id_col, 'hist_p12', 'hist_p23', 'hist_p13']

        if (sub_idx, year) in SKIP:
            print(f"  Skipping {desc} (already submitted)")
            continue

        print(f"  Submitting {desc} ({len(sub)} ADCs)")
        for attempt in range(20):
            try:
                task =  ee.batch.Export.table.toDrive(
                    collection=result, folder=folder,
                    description=desc, selectors=cols
                )
                task.start()
                submitted += 1
                break
            except ee.ee_exception.EEException as e:
                if 'too many tasks' in str(e).lower():
                    wait =  60 * (attempt + 1)
                    print(f"    Queue full, waiting {wait}s (attempt {attempt+1})...")
                    time.sleep(wait)
                else:
                    print(f"    ERROR: {e}")
                    break

print(f"\nDone. Submitted {submitted} sub-batch tasks.")
