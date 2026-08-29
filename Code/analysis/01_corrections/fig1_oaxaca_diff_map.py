"""
fig1_oaxaca_diff_map.py — panel (f) of Figure 1: Oaxaca inset of the DIFFERENCE
between the ADC census yield (panel d) and the municipality census average
(panel c), i.e. within-municipality yield deviations.

Data: CA2007 maize yields at the 2007 ADC level (rendimiento_agr_adc.dta),
summed across seasons (the notebook's season_comp == "combined"), drawn on the
CA2007 ADC polygons (Data/Shapefiles/adc_shapefile.shp). The municipal average
subtracted is sum(Q)/sum(sup_sem) over the same census ADC records. Rendered in
the MUNICIPIOS.shp Lambert Conformal Conic CRS on the identical Oaxaca window
as panels (c)/(d) (x 3050000-3100000, y 710000-745000).

Output (to Maize_prediction/plots/):
  maizeyield_adc_mun_diff.png   ADC yield minus municipal census average, Oaxaca inset

Run: ~/miniforge3/envs/geo_env/bin/python fig1_oaxaca_diff_map.py
"""
import os, warnings
import numpy  as np
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")            # PNG map, matches the other Figure 1 panels (not pgf)
import matplotlib.pyplot as plt
import geopandas as gpd

# ── Directories ──────────────────────────────────────────
home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
misal_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Crop_misallocation")
data_dir    =  os.path.join(proj_dir, "Data")
plot_dir    =  os.path.join(proj_dir, "plots")
ca07_dir    =  os.path.join(misal_dir, "Data", "INEGI", "Microdata_lab_outputs",
                            "CA07_ADC_prod_data_2023-05-09")

# ── Inputs ───────────────────────────────────────────────
adc_yield_path =  os.path.join(ca07_dir, "rendimiento_agr_adc.dta")                # CA2007 maize yields at the 2007 ADC level
adc_shp_path   =  os.path.join(data_dir, "Shapefiles", "adc_shapefile.shp")        # CA2007 ADC polygons (WGS84)
mun_shp_path   =  os.path.join(misal_dir, "Data", "Municipality_shp", "MUNICIPIOS.shp")  # municipality polygons (LCC CRS of panels c/d)
state_shp_path =  os.path.join(misal_dir, "Data", "Municipality_shp", "STATES.shp")      # state boundaries

# ── Outputs ──────────────────────────────────────────────
out_png     =  os.path.join(plot_dir, "maizeyield_adc_mun_diff.png")               # Figure 1 panel (f)

# Oaxaca inset window in the MUNICIPIOS.shp projected CRS — identical to panels (c)/(d)
XLIM        =  (3050000, 3100000)
YLIM        =  (710000, 745000)
DPI         =  600


def main():
    # ── CA2007 maize yields on 2007 ADCs, summed across seasons ──
    adc =  pd.read_stata(adc_yield_path)
    adc =  adc[adc["name"] == "Maize"]
    adc =  (adc.groupby(["adc", "muncode"])[["sup_sem", "Q"]].sum().reset_index()
               .rename(columns={"adc": "adcid"}))
    adc["yield"] =  adc["Q"] / adc["sup_sem"]

    # municipal census average — the quantity mapped in panel (c)
    mun =  adc.groupby("muncode").agg(Q=("Q", "sum"), s=("sup_sem", "sum")).reset_index()
    mun["mun_yield"] =  mun["Q"] / mun["s"]
    adc =  adc.merge(mun[["muncode", "mun_yield"]], on="muncode", how="left")
    adc["yield_diff"] =  adc["yield"] - adc["mun_yield"]     # panel (d) minus panel (c)
    adc["zero_yield"] =  adc["yield"] == 0                   # rendered lightgrey in panel (d); mirror that

    # ── Geometry ─────────────────────────────────────────
    mun_shp =  gpd.read_file(mun_shp_path)
    st_line =  gpd.read_file(state_shp_path)
    st_line["geometry"] =  st_line["geometry"].boundary

    print("reading CA2007 ADC polygons (Oaxaca window)")
    # window y 710-745km in this LCC (lat origin 12) sits at lat ~18.1-18.8, lon ~-96.9..-96.1
    shp =  gpd.read_file(adc_shp_path, bbox=(-97.4, 17.7, -95.6, 19.2))
    shp =  shp[["adcid", "geometry"]]
    shp =  shp.to_crs(mun_shp.crs)
    shp =  shp.merge(adc[["adcid", "yield_diff", "zero_yield"]], on="adcid", how="left")
    print(f"    {len(shp):,} polygons, {shp['yield_diff'].notna().sum():,} with a yield difference")

    # ── Map, composed like panel (d) ─────────────────────
    fig, ax =  plt.subplots(1, 1, figsize=(14, 7))
    mun_shp.plot(ax=ax, edgecolor="white", linewidth=3)
    shp[shp["yield_diff"].isnull()].plot(color="silver", ax=ax)
    zero =  shp[shp["zero_yield"] == True]
    if len(zero): zero.plot(color="lightgrey", ax=ax)
    nn =  shp[shp["yield_diff"].notnull() & (shp["zero_yield"] != True)]
    nn.plot("yield_diff", ax=ax, legend=True, vmin=-2, vmax=2, cmap="RdBu_r")
    st_line.plot(ax=ax, edgecolor="dimgrey", linewidth=1)

    ax.axis("off"); ax.set_axis_off()
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_title("Difference between área de control and municipality maize yield in Oaxaca state")
    plt.savefig(out_png, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
