"""
plot_yields_ca22_maps.py — CA2022 versions of the ADC/municipality yield maps.

Replaces the maps produced by 1_plot_yields_ADC_mun.ipynb, which plotted the
**CA2007** census (that notebook builds adc_rc_df twice; cell 14 overwrites the
CA22 version from cell 3 with CA07 data, so every map cell downstream is CA07)
on the 2016 AMCA polygons. The CA2007 polygon shapefile it referenced
(SCIAGA/CA2007_adcloc_poly.shp) is also no longer present locally.

Geometry here is `adc_geometries_for_ee.csv` — the ADC polygons the entire Earth
Engine extraction ran on. Its `adcid` ("01001106-8001") is the CA22 `adc` id with
the dash removed, so census yields and every model prediction join one-to-one,
and the maps depict exactly the units the paper's results are computed on.

Prediction maps use the paper's best specification: **AEF Hist Ens. with
within-municipality shrinkage** (combined-season R2 0.598, within-R2 0.225).
Shrinkage lambda is estimated exactly as in accuracy_main_2022.py (GroupKFold
over municipalities), so the mapped surface matches the Shrink table rows.

Outputs (to Maize_prediction/plots/) — filenames preserved from the CA07 versions
so they are drop-in replacements for the Overleaf figures, EXCEPT the municipal
SIAP/pred pair, which moves 2018 -> 2022 (the census year, and the only year our
ADC predictions cover). Those two need a one-line \\includegraphics update each.

  (--census-maps only; the paper keeps the CA07 versions of every CENSUS-yield map:)
  maizeyield_allmx_adc.png                 CA22 ADC maize yield, national
  maizeyield_allmx_adc_with_legend.png       "  with colourbar
  maizeyield_adc.png                       CA22 ADC yield, Oaxaca inset
  maizeyield_adc_preds.png                 AEF Hist Ens. Shrink predictions
  maizeyield_adc_pred_error_prederror_ls_noleg.png      NDVI error
  maizeyield_adc_pred_error_prederror_rcpred_noleg.png  AEF Shrink error  <- best
  maizeyield_adc_pred_error_prederror_munyield.png      SIAP muni-average error
  maizeyield_mun_allmx.png                 CA22 municipal yield, national   (--census-maps only)
  maizeyield_mun.png                       CA22 municipal yield, Oaxaca inset (--census-maps only)
  maizeyield_mun_siap_allmx_nolegend_2022.png   SIAP 2022 municipal yield
  maizeyield_mun_pred_allmx_nolegend_2022.png   predicted 2022 municipal yield

NOT regenerated: maize_monthly_harvesting.png (built by
build/02_clean_siap_monthly/1_CleanSIAPMonthlydata.ipynb; unchanged by design).

Run: ~/miniforge3/envs/geo_env/bin/python plot_yields_ca22_maps.py
"""
import os, sys, json, warnings
import numpy  as np
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")            # PNG maps, not pgf — these are images, not typeset plots
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry       import shape
from sklearn.model_selection import GroupKFold

# ── Directories ──────────────────────────────────────────
home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir    =  os.path.join(proj_dir, "Data")
pred_dir    =  os.path.join(data_dir, "predictions")
plot_dir    =  os.path.join(proj_dir, "plots")
inegi_dir   =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
ca2022_dir  =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")

# ── Inputs ───────────────────────────────────────────────
geom_path   =  os.path.join(data_dir, "adc_geometries_for_ee.csv")             # ADC polygons (EE extraction geometry)
ca_use_path =  os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta")         # CA22 ADC maize yields
ens_path    =  os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet")         # AEF Hist Ens. preds + GT
ndvi_path   =  os.path.join(pred_dir, "adc_aefn2_masked_preds.parquet")        # masked NDVI baseline preds
mun_shp     =  os.path.join(data_dir, "muncodes", "shp", "MUNICIPIOS.shp")     # municipality polygons
siap_path   =  os.path.join(home_dir, "Dropbox", "Projects",
                            "Maize_prediction", "Data", "SIAP",
                            "Cleaned", "siap_ag_prod_estimation_by_season.dta") # SIAP municipal yields

CENSUS_MAPS =  "--census-maps" in sys.argv   # regenerate the CA07-retained census maps
EVAL_YEAR   =  2022
DPI         =  600
OAXACA_BBOX =  (-97.2, 16.3, -95.8, 17.6)      # lon/lat window for the inset panels
os.makedirs(plot_dir, exist_ok=True)


# ── Shrinkage (identical to accuracy_main_2022.py) ───────
def cv_lambda(df, pcol, ycol, gc="muncode"):
    s =  df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    cnt =  s.groupby(gc)[ycol].transform("size"); s = s[cnt >= 2].reset_index(drop=True)
    if s[gc].nunique() < 5: return np.nan
    lams =  []
    for tr, _ in GroupKFold(5).split(s, groups=s[gc]):
        d =  s.iloc[tr]
        a =  (d[ycol] - d.groupby(gc)[ycol].transform("mean")).values
        b =  (d[pcol] - d.groupby(gc)[pcol].transform("mean")).values
        den =  np.sqrt(np.sum(a*a) * np.sum(b*b))
        if den > 0:
            rho =  np.sum(a*b)/den; rr = np.sqrt(np.sum(b*b)/np.sum(a*a))
            if rr > 0: lams.append(rho/rr)
    return float(np.clip(np.mean(lams), 0, 1)) if lams else np.nan


def shrink(df, pcol, lam, gc="muncode"):
    g =  df.groupby(gc)[pcol]
    return g.transform("mean") + lam * (df[pcol] - g.transform("mean"))


# ── Data ─────────────────────────────────────────────────
def load_adc_geometries():
    print("[1] reading ADC polygons (large CSV, ~1 min)")
    g =  pd.read_csv(geom_path, usecols=["adcid", "muncode", "coords"])
    g["adc"] =  g["adcid"].astype(str).str.replace("-", "", regex=False)
    g["geometry"] =  g["coords"].apply(lambda s: shape(json.loads(s)))
    gdf =  gpd.GeoDataFrame(g[["adc", "adcid", "muncode", "geometry"]],
                            geometry="geometry", crs="EPSG:4326")
    print(f"    {len(gdf):,} ADC polygons")
    return gdf


def load_values():
    ca =  pd.read_stata(ca_use_path)
    gt =  ca[ca["name"] == "Maize"][["adc", "muncode", "yield", "land_input"]].copy()
    gt["adc"] =  gt["adc"].astype(str)
    gt["muncode"] =  gt["muncode"].astype(str).str.zfill(5)

    ens =  pd.read_parquet(ens_path)[["adc", "muncode", "yield", "pred"]].copy()
    ens["adc"] =  ens["adc"].astype(str)
    ens["muncode"] =  ens["muncode"].astype(str).str.zfill(5)
    lam =  cv_lambda(ens, "pred", "yield")
    ens["pred_shrink"] =  shrink(ens, "pred", lam)
    print(f"[2] AEF Hist Ens. shrinkage lambda = {lam:.3f}")

    ndvi =  pd.read_parquet(ndvi_path)
    ndvi["adc"] =  ndvi["adcid"].astype(str).str.replace("-", "", regex=False)
    ndvi =  ndvi[["adc", "pred"]].rename(columns={"pred": "pred_ndvi"})

    siap =  pd.read_stata(siap_path)
    siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
    s22 =  siap[(siap["name"] == "Maize") & (siap["year"] == EVAL_YEAR)]
    s22 =  s22[~s22["muncode"].str.endswith("000")]
    siap_mun =  s22.groupby("muncode").agg(q=("q", "sum"), ha=("ha_planted", "sum")).reset_index()
    siap_mun["siap_yield"] =  siap_mun["q"] / siap_mun["ha"]

    df =  (gt.merge(ens[["adc", "pred", "pred_shrink"]], on="adc", how="left")
             .merge(ndvi, on="adc", how="left")
             .merge(siap_mun[["muncode", "siap_yield"]], on="muncode", how="left"))
    df["err_shrink"] =  df["pred_shrink"]  - df["yield"]
    df["err_ndvi"]   =  df["pred_ndvi"]    - df["yield"]
    df["err_siap"]   =  df["siap_yield"]   - df["yield"]
    return df, siap_mun


# ── Plot helpers ─────────────────────────────────────────
def base_map(gdf, col, fname, vmin, vmax, cmap="viridis", legend=False,
             bbox=None, overlay=None, title=None, figsize=(14, 7)):
    fig, ax =  plt.subplots(1, 1, figsize=figsize)
    if overlay is not None:
        overlay.plot(ax=ax, color="white", edgecolor="white", linewidth=0.3)
    null =  gdf[gdf[col].isnull()]
    if len(null): null.plot(color="silver", ax=ax)
    nn =  gdf[gdf[col].notnull()]
    if len(nn[nn[col] == 0]): nn[nn[col] == 0].plot(color="lightgrey", ax=ax)
    nn[nn[col] != 0].plot(col, ax=ax, legend=legend, vmin=vmin, vmax=vmax, cmap=cmap)
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    if title: ax.set_title(title)
    ax.axis("off"); ax.set_axis_off()
    ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    out =  os.path.join(plot_dir, fname)
    plt.savefig(out, bbox_inches="tight", dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"    wrote {fname}")


def main():
    gdf =  load_adc_geometries()
    df, siap_mun =  load_values()

    adc =  gdf.merge(df, on="adc", how="left")
    print(f"[3] ADC map frame: {len(adc):,} polygons, "
          f"{adc['yield'].notna().sum():,} with CA22 yield, "
          f"{adc['pred_shrink'].notna().sum():,} with AEF shrink prediction")

    # municipality layer: dissolve ADC polygons (keeps geometry consistent with the ADC maps)
    print("[4] dissolving municipalities")
    # muncode arrives from the geometry CSV as int64; the census/SIAP frames use a
    # zero-padded 5-char string. Cast before dissolving so the merges below line up.
    mun =  adc[["muncode_x", "geometry"]].rename(columns={"muncode_x": "muncode"}).copy()
    mun["muncode"] =  mun["muncode"].astype(str).str.zfill(5)
    mun =  mun.dissolve(by="muncode").reset_index()
    munv =  (df.assign(vol=df["yield"] * df["land_input"])
               .groupby("muncode").agg(vol=("vol", "sum"), la=("land_input", "sum")).reset_index())
    munv["yield_ca22"] =  munv["vol"] / munv["la"]
    predv =  (df.assign(pv=df["pred_shrink"] * df["land_input"])
                .groupby("muncode").agg(pv=("pv", "sum"), la=("land_input", "sum")).reset_index())
    predv["pred_mun"] =  predv["pv"] / predv["la"]
    mun =  (mun.merge(munv[["muncode", "yield_ca22"]], on="muncode", how="left")
               .merge(predv[["muncode", "pred_mun"]], on="muncode", how="left")
               .merge(siap_mun[["muncode", "siap_yield"]], on="muncode", how="left"))

    # ── ADC-level census yield — OFF by default ──────────
    # DECISION 2026-08-16: the CA22 renderings of the three census-yield maps look
    # WORSE than the existing CA07 ones and the paper keeps the CA07 versions. Cause
    # is coverage: only 94,037 of 293,138 ADC polygons carry a CA22 maize yield, so
    # the national map is mostly grey, where the CA07 maps were drawn on the coarser
    # 2016 AMCA polygons and read as denser. Pass --census-maps to regenerate them
    # anyway; without it these three files are left untouched.
    if CENSUS_MAPS:
        print("[5] ADC census-yield maps (--census-maps given; overwrites the CA07 versions)")
        base_map(adc, "yield", "maizeyield_allmx_adc.png",            0, 12)
        base_map(adc, "yield", "maizeyield_allmx_adc_with_legend.png", 0, 12, legend=True)
        base_map(adc, "yield", "maizeyield_adc.png",                   0,  5, legend=True,
                 bbox=OAXACA_BBOX, overlay=mun,
                 title="Maize yield (tons/hectare harvested) at área de control level in Oaxaca state")
    else:
        print("[5] ADC census-yield maps SKIPPED — paper keeps the CA07 versions "
              "(pass --census-maps to override)")

    # ── ADC-level predictions (best model) ───────────────
    base_map(adc, "pred_shrink", "maizeyield_adc_preds.png", 0, 12, legend=True)

    # ── Matched-sample truth vs prediction pair (2026-08-28) ──
    # Both panels restricted to ADCs carrying BOTH a CA22 census maize yield and
    # an ensemble prediction, so the pair is directly comparable; every other
    # polygon renders silver in both panels. Replaces the CA07 truth map in the
    # paper's truth-vs-prediction figure.
    _m = adc["yield"].notna() & adc["pred_shrink"].notna()
    adc["yield_matched"] = np.where(_m, adc["yield"], np.nan)
    adc["pred_matched"]  = np.where(_m, adc["pred_shrink"], np.nan)
    print(f"    matched truth/pred pair: {int(_m.sum()):,} ADCs")
    base_map(adc, "yield_matched", "maizeyield_adc_truth_2022_matched.png", 0, 12, legend=True)
    base_map(adc, "pred_matched",  "maizeyield_adc_preds_2022_matched.png", 0, 12, legend=True)

    # ── Prediction-error maps ────────────────────────────
    print("[6] error maps")
    ERR =  dict(vmin=-4, vmax=4, cmap="RdBu_r")
    base_map(adc, "err_ndvi",   "maizeyield_adc_pred_error_prederror_ls_noleg.png",     **ERR)
    base_map(adc, "err_shrink", "maizeyield_adc_pred_error_prederror_rcpred_noleg.png", **ERR)
    base_map(adc, "err_siap",   "maizeyield_adc_pred_error_prederror_munyield.png",     legend=True, **ERR)

    # ── Municipality maps ────────────────────────────────
    # The two CENSUS-yield municipal maps are gated with the ADC census maps: same
    # DECISION 2026-08-16, the paper keeps the CA07 renderings of every census-yield
    # map. The SIAP and prediction maps below are NOT census-derived and stay CA22.
    print("[7] municipality maps")
    if CENSUS_MAPS:
        base_map(mun, "yield_ca22",  "maizeyield_mun_allmx.png", 0, 12, legend=True)
        base_map(mun, "yield_ca22",  "maizeyield_mun.png",       0,  5, legend=True,
                 bbox=OAXACA_BBOX,
                 title="Maize yield (tons/hectare harvested) at municipality level in Oaxaca state")
    else:
        print("    census-yield municipal maps SKIPPED — paper keeps the CA07 versions")
    base_map(mun, "siap_yield", f"maizeyield_mun_siap_allmx_nolegend_{EVAL_YEAR}.png", 0, 12)
    base_map(mun, "pred_mun",   f"maizeyield_mun_pred_allmx_nolegend_{EVAL_YEAR}.png", 0, 12)
    # legend (colorbar) variants used in the paper since 2026-08-27
    base_map(mun, "siap_yield", f"maizeyield_mun_siap_allmx_{EVAL_YEAR}.png", 0, 12, legend=True)
    base_map(mun, "pred_mun",   f"maizeyield_mun_pred_allmx_{EVAL_YEAR}.png", 0, 12, legend=True)

    print("\ndone — 11 figures written to plots/")
    print("NOTE: the municipal SIAP/pred pair is now _2022 (was _2018); update the two")
    print("      \\includegraphics lines in new_body.tex (L276, L279).")


if __name__ == "__main__":
    main()
