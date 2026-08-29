"""
One-off asset fetch for fig_methodology_diagram.py panel 1: download a real
AEF embedding chip (all 64 dims, 10 m, 2022) centered on the densest cropland
(ESA WorldCover class 40, as in the extraction pipeline) inside municipality
26002 (Agua Prieta, Sonora — the high-yield example region).

Requires Earth Engine auth (browser flow on first run).

Output: Data/alpha_earth/aef_chip_mun26002_2022.npy   (64 x 128 x 128 float32)
Run:    ~/miniforge3/envs/ml_cuda/bin/python fetch_aef_chip_fig.py
"""
import io
import os
import urllib.request

import ee
import numpy as np

# ── Directories ──────────────────────────────────────────
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir   =  os.path.join(proj_dir, "Data", "alpha_earth")

# ── Outputs ──────────────────────────────────────────────
chip_path =  os.path.join(aef_dir, "aef_chip_mun26002_2022.npy")  # 64 x 128 x 128

YEAR      =  2022
CHIP_PX   =  128            # chip size in 10 m pixels (1.28 km)
SCALE     =  10
# mun 26002 bounding box (WGS84), from MUNICIPIOS.shp
BBOX      =  [-109.7465, 30.7089, -108.6838, 31.3343]

try:
    ee.Initialize(project="avocadoyieldsdeforestation")
except Exception:
    ee.Authenticate()                                   # browser flow, first run only
    ee.Initialize(project="avocadoyieldsdeforestation")

region  =  ee.Geometry.Rectangle(BBOX)
wc      =  ee.ImageCollection("ESA/WorldCover/v200").filterBounds(region)
agland  =  (wc.map(lambda x: x.eq(40).rename("ag")).max().unmask(0)
              .setDefaultProjection(crs="EPSG:4326", scale=10))

# densest 1 km cropland cell in the bbox -> chip center
dens    =  (agland.reduceResolution(ee.Reducer.mean(), maxPixels=65536)
                  .reproject(crs="EPSG:4326", scale=1000)
                  .addBands(ee.Image.pixelLonLat()))
best    =  (dens.sample(region=region, scale=1000, numPixels=5000, seed=1)
                .sort("ag", False).first().getInfo())["properties"]
lon, lat =  best["longitude"], best["latitude"]
print(f"chip center: ({lat:.4f}, {lon:.4f}), cropland density {best['ag']:.2f}")

half    =  CHIP_PX * SCALE / 2.0
proj    =  ee.Projection("EPSG:4326").atScale(SCALE)
chip_re =  ee.Geometry.Point([lon, lat]).buffer(half, 1).bounds()

aef     =  (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
              .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
              .filterBounds(chip_re).mosaic())

url     =  aef.getDownloadURL({
    "region":     chip_re,
    "scale":      SCALE,
    "format":     "NPY",
})
with urllib.request.urlopen(url) as resp:
    arr =  np.load(io.BytesIO(resp.read()))

# structured array (bands as fields) -> (64, H, W) float32
chip    =  np.stack([arr[f"A{d:02d}"] for d in range(64)]).astype(np.float32)
np.save(chip_path, chip)
print(f"Wrote {chip_path}  shape={chip.shape}")
