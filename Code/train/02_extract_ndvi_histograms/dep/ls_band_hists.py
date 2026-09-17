"""
ls_band_hists.py — Extract NIR/RED/NDVI paired 2D histograms from Google Earth Engine.

Stage 3 of histogram_improvement.md: instead of NDVI alone, preserve absolute
NIR and RED reflectance (which NDVI's ratio discards) so the GB model can learn
soil-brightness / canopy-density / spectral-boundary signal that lifts the RAW
(uncorrected) histogram performance.

For each geometry, two phenology-based periods are formed from the planting month:
  P1 (vegetative)   : planting + 0..3 months
  P2 (reproductive) : planting + 3..6 months
and FIVE paired 2D histograms are built (each bins x bins, flattened to bins^2):
  H1  NDVI_p1 x NDVI_p2   trajectory (the existing 3-period signal)
  H2  NIR_p1  x NIR_p2    NIR reflectance trajectory
  H3  RED_p1  x RED_p2    RED reflectance trajectory
  H4  NIR_p1  x RED_p1    early-season spectral space
  H5  NIR_p2  x RED_p2    late-season spectral space
Total: 5 x bins^2 features (e.g. 5 x 1024 = 5120 at bins=32).

Band ranges (Landsat C2 L2 surface reflectance, scaled):
  NDVI [0.0, 1.0]   NIR [0.0, 0.6]   RED [0.0, 0.3]

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate ML_env
  python3 ls_band_hists.py <level: muni|adc> [bins=32] [start_state] [start_year]

Requires Earth Engine authentication (earthengine authenticate). Outputs are
exported to Google Drive/<level>_band_hists_<bins>bins/ as CSV shards, then
consolidated by clean_band_histograms.py (analogous to clean_3period_histograms.py).
"""
import os, sys, time
from ast import literal_eval
import pandas as pd
import ee

ee.Initialize()

level       =  sys.argv[1] if len(sys.argv) > 1 else "muni"
bins        =  int(sys.argv[2]) if len(sys.argv) > 2 else 32
start_state =  sys.argv[3] if len(sys.argv) > 3 else None
start_year  =  int(sys.argv[4]) if len(sys.argv) > 4 else 2003

# ── Band names by satellite ──────────────────────────────
BAND_MAP = {
    'LANDSAT/LT05/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LE07/C02/T1_L2': {'red': 'SR_B3', 'nir': 'SR_B4'},
    'LANDSAT/LC08/C02/T1_L2': {'red': 'SR_B4', 'nir': 'SR_B5'},
}

# ── Landsat helpers (same as ls_3period_hists.py) ────────
def scale_facts(img):
    optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    return img.addBands(optical, None, True)

def cloud_mask(img):
    mask = ee.Image.constant(1).subtract(
        img.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).neq(0))
    return img.updateMask(mask)

def band_image(img, red, nir):
    """Return an image with harmonized 'red','nir','ndvi' bands."""
    r = img.select(red).rename('red')
    n = img.select(nir).rename('nir')
    ndvi = n.subtract(r).divide(n.add(r)).rename('ndvi')
    return r.addBands(n).addBands(ndvi)

def get_band_composite(geom, start_date, end_date):
    """Median composite of red/nir/ndvi over a date range (L5/7/8 merged)."""
    collections = []
    for sat, b in BAND_MAP.items():
        col = (ee.ImageCollection(sat)
               .filterBounds(geom).filterDate(start_date, end_date)
               .map(scale_facts).map(cloud_mask))
        imgs = ee.Algorithms.If(col.size().gt(0),
                                col.map(lambda x: band_image(x, b['red'], b['nir'])),
                                ee.ImageCollection([]))
        collections.append(ee.ImageCollection(imgs))
    merged = collections[0].merge(collections[1]).merge(collections[2])
    merged = ee.Algorithms.If(
        ee.ImageCollection(merged).size().gt(0),
        ee.ImageCollection(merged),
        ee.ImageCollection([ee.Image.constant([0, 0, 0]).rename(['red', 'nir', 'ndvi'])]))
    return ee.ImageCollection(merged).median()

# ── Binning ──────────────────────────────────────────────
RANGES = {'ndvi': (0.0, 1.0), 'nir': (0.0, 0.6), 'red': (0.0, 0.3)}

def assign_bin(img, vmin, vmax, n_bins):
    scaled = img.subtract(vmin).divide(vmax - vmin).multiply(n_bins)
    return scaled.floor().clamp(0, n_bins - 1).toInt()

def calc_1d_bin(bin_a, bin_b, n_bins):
    return bin_a.add(bin_b.multiply(n_bins))

# ── Main histogram function (mapped over FeatureCollection) ──
def compute_band_hist(feature):
    geom = feature.geometry()
    planting_month = ee.Number(feature.get('planting_month'))
    gs_year  = ee.Number(feature.get('gs_year'))
    gs_start = ee.Date.fromYMD(gs_year, planting_month, 1)

    p1 = get_band_composite(geom, gs_start, gs_start.advance(3, 'month'))
    p2 = get_band_composite(geom, gs_start.advance(3, 'month'), gs_start.advance(6, 'month'))

    def b(img, band):
        vmin, vmax = RANGES[band]
        return assign_bin(img.select(band), vmin, vmax, bins)

    # five paired 2D -> 1D encodings
    h1 = calc_1d_bin(b(p1, 'ndvi'), b(p2, 'ndvi'), bins).rename('hist_ndvi')
    h2 = calc_1d_bin(b(p1, 'nir'),  b(p2, 'nir'),  bins).rename('hist_nir')
    h3 = calc_1d_bin(b(p1, 'red'),  b(p2, 'red'),  bins).rename('hist_red')
    h4 = calc_1d_bin(b(p1, 'nir'),  b(p1, 'red'),  bins).rename('hist_sp1')
    h5 = calc_1d_bin(b(p2, 'nir'),  b(p2, 'red'),  bins).rename('hist_sp2')
    combined = h1.addBands(h2).addBands(h3).addBands(h4).addBands(h5)

    n_sq = bins * bins
    hist = combined.reduceRegion(
        ee.Reducer.fixedHistogram(0, n_sq, n_sq), geom, 30, maxPixels=500_000_000)
    return feature.set(hist)

# ── Load geometries ──────────────────────────────────────
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
if level == "muni":
    geom_file, id_col, batch_n = os.path.join(data_dir, "muni_geometries_for_ee.csv"), "muncode", 25
elif level == "adc":
    geom_file, id_col, batch_n = os.path.join(data_dir, "adc_geometries_for_ee.csv"), "adcid", 500
else:
    print(f"Unknown level: {level}. Use 'muni' or 'adc'."); sys.exit(1)

geoms = pd.read_csv(geom_file)
geoms['coords'] = geoms['coords'].apply(literal_eval)
geoms['state'] = geoms[('muncode' if level == 'muni' else 'muncode')].apply(lambda x: str(x).zfill(5)[:2] if level == 'muni' else str(x)[:2])
print(f"Loaded {len(geoms)} {level}-level geometries")

folder = f"{level}_band_hists_{bins}bins"
QUEUE_LIMIT, QUEUE_BUFFER, POLL_INTERVAL = 3000, 50, 60

def count_active_tasks():
    import subprocess
    r = subprocess.run(['earthengine', 'task', 'list'], capture_output=True, text=True, timeout=120)
    return sum(1 for ln in r.stdout.splitlines() if any(s in ln for s in ('PENDING', 'RUNNING', 'READY')))

def wait_for_queue_space():
    while True:
        try:
            n = count_active_tasks()
            if n < QUEUE_LIMIT - QUEUE_BUFFER:
                print(f"    Queue has space ({n}/{QUEUE_LIMIT}), resuming..."); return
            print(f"    Queue full ({n}/{QUEUE_LIMIT}), waiting {POLL_INTERVAL}s..."); time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"    Queue check error: {e}, retry in {POLL_INTERVAL}s..."); time.sleep(POLL_INTERVAL)

_submit_count = 0
HIST_COLS = ['hist_ndvi', 'hist_nir', 'hist_red', 'hist_sp1', 'hist_sp2']

def submit_batch(batch_df, state, batch_idx, year):
    global _submit_count
    _submit_count += 1
    if _submit_count % 100 == 0:
        wait_for_queue_space()
    ee_coll = ee.FeatureCollection(batch_df['eeFeature'].to_list())
    print(f"  State {state}, batch {batch_idx}, year {year} — {len(batch_df)} {level}s")
    result = ee_coll.map(compute_band_hist)
    desc = f"band_hist_{level}_{state}_{batch_idx}_{year}"
    try:
        ee.batch.Export.table.toDrive(
            collection=result, folder=folder, description=desc,
            selectors=[id_col] + HIST_COLS).start()
    except ee.ee_exception.EEException as e:
        msg = str(e).lower()
        if 'payload size' in msg or 'exceeds the limit' in msg:
            mid = len(batch_df) // 2
            if mid == 0:
                print("    ERROR: single feature too large, skipping"); return
            print("    Payload too large, splitting...")
            submit_batch(batch_df.iloc[:mid], state, f"{batch_idx}a", year)
            submit_batch(batch_df.iloc[mid:], state, f"{batch_idx}b", year)
        elif 'too many tasks' in msg:
            wait_for_queue_space(); submit_batch(batch_df, state, batch_idx, year)
        else:
            raise

# ── Export loop: state × year × batch ────────────────────
states = sorted(geoms['state'].unique())
if start_state:
    states = [s for s in states if s >= start_state]
    print(f"Resuming from state {start_state} ({len(states)} states remaining)")

for state in states:
    state_geoms = geoms.loc[geoms['state'] == state].copy()
    print(f"\nState {state}: {len(state_geoms)} {level}s")
    state_geoms['eeGeom'] = state_geoms['coords'].apply(ee.Geometry)
    yr_start = start_year if state == start_state else 2003
    for year in range(yr_start, 2025):
        state_geoms['eeFeature'] = state_geoms.apply(
            lambda row: ee.Feature(row['eeGeom'], {
                id_col: row[id_col], 'planting_month': int(row['planting_month']), 'gs_year': year}),
            axis=1)
        n_batches = (len(state_geoms) + batch_n - 1) // batch_n
        for bx in range(n_batches):
            submit_batch(state_geoms.iloc[bx * batch_n:(bx + 1) * batch_n], state, bx, year)

print(f"\nAll tasks submitted. Outputs → Google Drive/{folder}/")
print("Next: download shards, run clean_band_histograms.py, then "
      "gb_band_hist_prediction.py to train and add 'Band Hist.' rows to the tables.")
