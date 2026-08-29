"""
Asset fetch for the redesigned Figure 2 methodology diagram (top row + panel 1).
For each of the two example municipalities (one low-, one high-yield) this pulls,
over an identical chip bounding box centered on the densest cropland cell:

  - a true-color RGB chip (Sentinel-2 SR, 2022 growing-season median composite)
  - the 64-dim AlphaEarth (AEF) embedding chip (2022)

Both are 10 m; the shared bbox lets panel C (RGB) and panel 1 (AEF) show the same
ground. Chip bboxes are written to a json so the figure script can label extents.

Municipalities (from problem_statement.png; SIAP maize Spring-Summer 2022 yields):
  20517  Santo Domingo Tepuxtepec, OAX   1.30 t/ha   (low)
  03001  Comondu, BCS                    6.87 t/ha   (high)

Requires Earth Engine auth. Interactive getDownloadURL works in restricted mode.

Outputs (Data/alpha_earth/):
  aef_chip_mun{code}_2022.npy    (64 x N x N float32)
  rgb_chip_mun{code}_2022.npy    (N x N x 3 float32, S2 SR reflectance)
  fig2_mun_chips_meta.json       (per-mun bbox, center, yield)
Run: ~/miniforge3/envs/geo_env/bin/python 14_fetch_mun_chips_fig.py
"""
import io
import os
import json
import urllib.request

os.environ.setdefault("PROJ_LIB", os.path.join(
    os.path.expanduser("~"), "miniforge3", "envs", "geo_env", "share", "proj"))

import ee
import numpy as np
import geopandas as gpd

# -- Directories --------------------------------------------
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir   =  os.path.join(proj_dir, "Data", "alpha_earth")
shp_path  =  os.path.join(proj_dir, "Data", "muncodes", "shp", "MUNICIPIOS.shp")
meta_path =  os.path.join(aef_dir, "fig2_mun_chips_meta.json")

YEAR      =  2022
CHIP_PX   =  200            # chip size in 10 m pixels (2.0 km square)
SCALE     =  10

# muncode -> (label, low/high, SS-2022 yield t/ha)
MUNS = {
    "20517": ("Santo Domingo Tepuxtepec, OAX", "low",  1.30),
    "03001": ("Comondu, BCS",                  "high", 6.87),
}

ee.Initialize(project="avocadoyieldsdeforestation")

shp  =  gpd.read_file(shp_path).to_crs("EPSG:4326")   # source is Lambert (m); EE needs lon/lat
shp["muncode"] =  shp["CVE_ENT"].astype(str).str.zfill(2) + shp["CVE_MUN"].astype(str).str.zfill(3)


def dense_cropland_center(bbox):
    """Center of the densest ~2 km cropland (ESA WorldCover cls 40) cluster in bbox.

    Samples cropland pixels (bounded count) and bins them client-side, avoiding a
    10 m setDefaultProjection over huge municipalities (2^31 pixel cap)."""
    region =  ee.Geometry.Rectangle(bbox)
    ag     =  (ee.ImageCollection("ESA/WorldCover/v200").filterBounds(region)
                 .mosaic().eq(40).selfMask().rename("ag"))
    fc     =  (ag.addBands(ee.Image.pixelLonLat())
                 .sample(region=region, scale=300, numPixels=6000, seed=1,
                         dropNulls=True, geometries=False))
    feats  =  fc.getInfo()["features"]
    pts    =  np.array([[f["properties"]["longitude"], f["properties"]["latitude"]]
                        for f in feats])
    # densest ~2 km (0.02 deg) cell
    cell   =  np.round(pts / 0.02).astype(int)
    keys, inv, cnt =  np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    best_k =  cnt.argmax()
    ctr    =  pts[inv == best_k].mean(axis=0)
    return float(ctr[0]), float(ctr[1]), float(cnt[best_k] / len(pts))


def s2_rgb(chip_re):
    """Growing-season 2022 median true-color composite, clouds masked."""
    def mask(img):
        scl =  img.select("SCL")
        ok  =  scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(ok)
    col =  (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(f"{YEAR}-05-01", f"{YEAR}-11-30")
              .filterBounds(chip_re)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
              .map(mask))
    return col.select(["B4", "B3", "B2"]).median()


def download_npy(img, chip_re):
    url =  img.getDownloadURL({"region": chip_re, "scale": SCALE, "format": "NPY"})
    with urllib.request.urlopen(url) as resp:
        return np.load(io.BytesIO(resp.read()))


meta = {}
for code, (label, grp, yld) in MUNS.items():
    geom  =  shp.loc[shp["muncode"] == code].geometry.iloc[0]
    bbox  =  list(geom.bounds)                                  # minx,miny,maxx,maxy
    lon, lat, dens =  dense_cropland_center(bbox)
    half  =  CHIP_PX * SCALE / 2.0
    chip_re =  ee.Geometry.Point([lon, lat]).buffer(half, 1).bounds()
    ext   =  chip_re.bounds().coordinates().get(0).getInfo()    # ring for extent
    print(f"{code} {label}: center ({lat:.4f},{lon:.4f}) cropland {dens:.2f}")

    # AEF 64-dim chip
    aef  =  (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
               .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
               .filterBounds(chip_re).mosaic())
    arr  =  download_npy(aef, chip_re)
    chip =  np.stack([arr[f"A{d:02d}"] for d in range(64)]).astype(np.float32)
    np.save(os.path.join(aef_dir, f"aef_chip_mun{code}_2022.npy"), chip)
    print(f"   AEF chip {chip.shape}")

    # S2 RGB chip
    rgb_arr =  download_npy(s2_rgb(chip_re), chip_re)
    rgb  =  np.dstack([rgb_arr["B4"], rgb_arr["B3"], rgb_arr["B2"]]).astype(np.float32)
    np.save(os.path.join(aef_dir, f"rgb_chip_mun{code}_2022.npy"), rgb)
    print(f"   RGB chip {rgb.shape}")

    xs =  [p[0] for p in ext]; ys =  [p[1] for p in ext]
    meta[code] = {"label": label, "group": grp, "yield": yld,
                  "center": [lon, lat], "cropland_density": dens,
                  "extent": [min(xs), max(xs), min(ys), max(ys)],
                  "mun_bounds": bbox}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote", meta_path)
