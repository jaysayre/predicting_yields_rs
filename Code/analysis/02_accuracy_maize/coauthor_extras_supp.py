"""coauthor_extras_supp.py

Supplementary diagnostics for AEF Hist Ensemble (paper-aligned),
to accompany `coauthor_extras_paper.py`:

  6. ADC-level prediction-error choropleth (raw + corrected)
  7. Variance-compression: SD(pred) vs SD(obs) per municipality
  8. Exemplar municipality panels (3 high + 3 low within-R²)
  9. State-level accuracy table (CSV + LaTeX)
 10. 2D heatmaps of pooled within-R² (irrig × N_adc, irrig × maize share)

Reads:
  Data/predictions/adc_aef_hist_ens_preds.parquet
  Data/predictions/adc_aef_hist_ens_eval.parquet

Writes (under `Maize_prediction/plots/coauthor_extras_paper/`):
  fig6_err_map_{raw,corr}.png
  fig7_variance_compression.png
  fig8_exemplar_muns.png
  fig9a_within_r2_irrig_x_nadc_heatmap.png
  fig9b_within_r2_irrig_x_maizeshare_heatmap.png
  state_accuracy.csv
  state_accuracy.tex
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams.update({"font.family": "serif",
                      "axes.titlesize": 12, "axes.labelsize": 11,
                      "xtick.labelsize": 9, "ytick.labelsize": 9,
                      "legend.fontsize": 9, "figure.dpi": 130})

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir   =  os.path.join(proj_dir, "Data")
preds_dir  =  os.path.join(data_dir, "predictions")
out_dir    =  os.path.join(proj_dir, "plots", "coauthor_extras_paper")
sciaga_dir =  os.path.join(home_dir, "Dropbox", "Projects",
                            "Crop_misallocation", "Data")
siap_pkg   =  os.path.join(home_dir, "Dropbox", "Projects",
                            "Maize_prediction", "Data", "SIAP", "Cleaned")

adc_shp     =  os.path.join(sciaga_dir, "SCIAGA", "CA2007_adcloc_poly.shp")
state_shp   =  os.path.join(sciaga_dir, "Municipality_shp", "STATES.shp")
agland_csv  =  os.path.join(data_dir, "SIAP_agland", "Output",
                             "2007_adcs_agland_area.csv")
siap_ca07   =  os.path.join(siap_pkg, "siap_ag_prod_estimation_ca2007.dta")

MIN_ADCS_FOR_R2  =  2
R2_CLIP_LO       =  -2.0

# ────────────────────────────────────────────────────────
# Eval helpers (match paper)
# ────────────────────────────────────────────────────────
def _r2(y, yh):
    m  =  np.isfinite(y) & np.isfinite(yh)
    y, yh  =  np.array(y[m]), np.array(yh[m])
    if len(y) < 2: return np.nan
    st  =  np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yh) ** 2) / st if st > 0 else np.nan

def within_r2(d, y, p, gc="muncode"):
    s  =  d[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    c  =  s.groupby(gc).size()
    s  =  s[s[gc].isin(c[c >= MIN_ADCS_FOR_R2].index)]
    if len(s) == 0: return np.nan
    gm  =  s.groupby(gc)[[y, p]].transform("mean")
    return _r2(s[y] - gm[y], s[p] - gm[p])

def between_r2(d, y, p, gc="muncode"):
    s  =  d[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g  =  s.groupby(gc)[[y, p]].mean()
    return _r2(g[y], g[p])

# ────────────────────────────────────────────────────────
# Load
# ────────────────────────────────────────────────────────
print("Loading paper-aligned eval frame ...")
df  =  pd.read_parquet(os.path.join(preds_dir, "adc_aef_hist_ens_eval.parquet"))
pred_adc  =  pd.read_parquet(os.path.join(preds_dir, "adc_aef_hist_ens_preds.parquet"))
pred_adc["adc"]  =  pred_adc["adcid"].str.replace("-", "", regex=False)
print(f"  eval rows: {len(df):,}  unique adcs: {df.adc.nunique():,}")
print(f"  preds rows: {len(pred_adc):,}")

df["state"]  =  df["muncode"].str[:2]
df["err_raw"]   =  df["pred"]      - df["yield"]
df["err_corr"]  =  df["pred_corr"] - df["yield"]

# Aggregate INEGI to one row per adc (planted-area-weighted mean of yield)
def adc_collapse(d, ycol):
    s  =  d.dropna(subset=[ycol, "land_input"]).copy()
    s["wQ"]  =  s[ycol] * s["land_input"]
    s["wA"]  =  s["land_input"]
    g  =  (s.groupby("adc")
             .agg(wQ=("wQ", "sum"), wA=("wA", "sum"))
             .reset_index())
    g[ycol]  =  g["wQ"] / g["wA"]
    return g[["adc", ycol]]

inegi_adc  =  adc_collapse(df, "yield")

# Merge ADC-level: predictions + ADC-collapsed obs
adc  =  (pred_adc[["adcid", "adc", "muncode", "pred"]]
           .merge(inegi_adc, on="adc", how="left"))
# Also merge corrected pred (shift recomputed earlier in df at UP-level — apply at adcid level via the per-mun pred_corr−pred shift)
shift  =  (df.dropna(subset=["pred", "pred_corr"])
              .groupby("muncode")
              .apply(lambda g: pd.Series({"shift_per_mun": (g["pred_corr"] - g["pred"]).mean()}),
                      include_groups=False)
              .reset_index())
adc  =  adc.merge(shift, on="muncode", how="left")
adc["pred_corr"]  =  (adc["pred"] + adc["shift_per_mun"]).clip(lower=0)
adc.loc[adc["shift_per_mun"].isna(), "pred_corr"]  =  np.nan
adc["err_raw"]    =  adc["pred"]      - adc["yield"]
adc["err_corr"]   =  adc["pred_corr"] - adc["yield"]

states_g  =  gpd.read_file(state_shp).to_crs(epsg=4326)
adc_geo   =  gpd.read_file(adc_shp)[["adcid", "geometry"]].to_crs(epsg=4326)

# ────────────────────────────────────────────────────────
# 6. ADC error choropleth (raw + corrected)
# ────────────────────────────────────────────────────────
print("\n=== 6. Error choropleths ===")
for col, fname, ttl in [
    ("err_raw",  "fig6_err_map_raw.png",  "Raw error: AEF Hist Ens. − INEGI 2022"),
    ("err_corr", "fig6_err_map_corr.png", "Corrected error: AEF Hist Ens. Corr. − INEGI 2022"),
]:
    g  =  adc_geo.merge(adc[["adcid", col]], on="adcid", how="left")
    g[col + "_plot"]  =  g[col].clip(lower=-4, upper=4)
    fig, ax  =  plt.subplots(figsize=(14, 11))
    norm  =  TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    g.plot(column=col + "_plot", ax=ax, cmap="RdBu_r", norm=norm, linewidth=0,
            legend=True, missing_kwds={"color": "#e8e8e8"},
            legend_kwds={"label": "Prediction error (t/ha) — clipped at ±4",
                         "shrink": 0.6})
    states_g.boundary.plot(ax=ax, color="#222", linewidth=0.45)
    ax.set_axis_off()
    matched  =  g[col].notna().sum()
    mean_err  =  g[col].mean()
    median_err  =  g[col].median()
    rmse  =  np.sqrt((g[col] ** 2).mean())
    ax.set_title(f"{ttl}\n"
                  f"{matched:,}/{len(adc_geo):,} ADCs   "
                  f"mean={mean_err:+.2f}  median={median_err:+.2f}  "
                  f"RMSE={rmse:.2f} t/ha")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight", dpi=170)
    plt.close(fig)

# ────────────────────────────────────────────────────────
# 7. Variance compression: SD(pred) vs SD(obs) per mun
# ────────────────────────────────────────────────────────
print("\n=== 7. Variance compression ===")
mun_sd  =  (df.dropna(subset=["pred", "yield"])
              .groupby("muncode")
              .apply(lambda g: pd.Series({
                  "sd_obs":   g["yield"].std(),
                  "sd_pred":  g["pred"].std(),
                  "sd_pred_corr": g["pred_corr"].std(),
                  "n_rows":   len(g),
              }), include_groups=False)
              .reset_index())
mun_sd  =  mun_sd[mun_sd["n_rows"] >= MIN_ADCS_FOR_R2]
mun_sd  =  mun_sd.replace([np.inf, -np.inf], np.nan).dropna(subset=["sd_obs", "sd_pred"])

# Slopes via robust quantile fit (median through origin)
def slope_through_origin(x, y):
    m  =  np.isfinite(x) & np.isfinite(y) & (x > 0)
    return np.sum(x[m] * y[m]) / np.sum(x[m] ** 2)

slope_raw   =  slope_through_origin(mun_sd["sd_obs"].values,
                                      mun_sd["sd_pred"].values)
slope_corr  =  slope_through_origin(mun_sd["sd_obs"].values,
                                      mun_sd["sd_pred_corr"].values)

fig, ax  =  plt.subplots(1, 2, figsize=(13, 5.5))
maxv  =  np.nanpercentile(mun_sd[["sd_obs", "sd_pred", "sd_pred_corr"]].values, 99)
for ax_i, (col, lab, slope) in zip(
    ax,
    [("sd_pred",      "Raw",        slope_raw),
     ("sd_pred_corr", "Corrected",  slope_corr)]):
    ax_i.scatter(mun_sd["sd_obs"], mun_sd[col], s=np.sqrt(mun_sd["n_rows"]) * 1.4,
                  alpha=0.35, color="#3a78b0", edgecolor="none", rasterized=True)
    lims  =  [0, maxv]
    ax_i.plot(lims, lims, "--", color="#888", lw=0.8, label="1:1 line")
    ax_i.plot(lims, [0, slope * maxv], "-", color="#c44e52", lw=1.0,
               label=f"slope through origin = {slope:.2f}")
    ax_i.set_xlim(lims); ax_i.set_ylim(lims)
    ax_i.set_xlabel("SD of INEGI obs yield within mun (t/ha)")
    ax_i.set_ylabel(f"SD of {lab.lower()} pred within mun (t/ha)")
    ax_i.set_title(f"{lab} predictions — within-mun spread")
    ax_i.legend(loc="upper left")
fig.suptitle(f"Variance compression: predictions are systematically "
              f"under-dispersed within municipalities  (N = {len(mun_sd):,} muns)",
              y=1.005)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig7_variance_compression.png"),
             bbox_inches="tight", dpi=160)
plt.close(fig)
print(f"  slope SD(pred)/SD(obs)      = {slope_raw:.3f}")
print(f"  slope SD(pred_corr)/SD(obs) = {slope_corr:.3f}")

# ────────────────────────────────────────────────────────
# 8. Exemplar municipality panels (3 high + 3 low within-R²)
# ────────────────────────────────────────────────────────
print("\n=== 8. Exemplar mun panels ===")
mun_r2  =  []
for mc, g in df.dropna(subset=["pred", "yield"]).groupby("muncode"):
    if len(g) < MIN_ADCS_FOR_R2: continue
    ym, yhm  =  g["yield"].mean(), g["pred"].mean()
    ss_tot  =  ((g["yield"] - ym) ** 2).sum()
    ss_res  =  (((g["yield"] - ym) - (g["pred"] - yhm)) ** 2).sum()
    r2v  =  1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mun_r2.append({"muncode": mc, "within_r2": r2v,
                   "n": len(g),
                   "sd_obs": g["yield"].std(),
                   "mean_obs": ym})
mun_r2  =  pd.DataFrame(mun_r2)
# Filter: require enough sample + nontrivial obs variance
cand  =  mun_r2[(mun_r2["n"] >= 20) & (mun_r2["sd_obs"] >= 0.5)
                  & mun_r2["within_r2"].notna()].copy()

# State names for context
siap  =  pd.read_stata(siap_ca07)
mun_names  =  (siap.dropna(subset=["Estado", "Municipio"])
                    .groupby("muncode")
                    .agg(state_name=("Estado", "first"),
                         mun_name=("Municipio", "first"))
                    .reset_index())
cand  =  cand.merge(mun_names, on="muncode", how="left")

top  =  cand.sort_values("within_r2", ascending=False).head(3)
bot  =  cand.sort_values("within_r2", ascending=True).head(3)
exemplars  =  pd.concat([top, bot], ignore_index=True)
print(exemplars[["muncode", "state_name", "mun_name", "n",
                  "sd_obs", "mean_obs", "within_r2"]].to_string(index=False))

fig, axes  =  plt.subplots(2, 3, figsize=(15, 8.5))
for ax_i, (_, row) in zip(axes.flatten(), exemplars.iterrows()):
    g  =  df[(df["muncode"] == row["muncode"]) &
              df["pred"].notna() & df["yield"].notna()].copy()
    g  =  g.sort_values("yield").reset_index(drop=True)
    g["rank"]  =  range(1, len(g) + 1)
    ax_i.scatter(g["rank"], g["yield"], color="#222", s=18,
                  label="INEGI obs", zorder=3)
    ax_i.scatter(g["rank"], g["pred"], color="#c44e52", s=18, marker="s",
                  label="Pred (raw)", alpha=0.85, zorder=2)
    ax_i.scatter(g["rank"], g["pred_corr"], color="#3a78b0", s=18, marker="^",
                  label="Pred (corr.)", alpha=0.85, zorder=2)
    ax_i.set_xlabel("ADC rank (within mun, by obs yield)")
    ax_i.set_ylabel("Maize yield (t/ha)")
    sn  =  (row["state_name"] or "")[:18]
    mn  =  (row["mun_name"] or "")[:24]
    ax_i.set_title(f"{sn} — {mn}  ({row['muncode']})\n"
                    f"N = {row['n']}  within-R² = {row['within_r2']:+.2f}",
                    fontsize=10)
    if ax_i is axes.flatten()[0]:
        ax_i.legend(loc="upper left", fontsize=8)
fig.suptitle("Exemplar muns: 3 highest within-R² (top row) vs "
              "3 lowest (bottom row)", y=1.005)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig8_exemplar_muns.png"),
             bbox_inches="tight", dpi=170)
plt.close(fig)

# ────────────────────────────────────────────────────────
# 9. State-level accuracy table (raw + corrected, combined + SS)
# ────────────────────────────────────────────────────────
print("\n=== 9. State accuracy table ===")
state_names  =  (siap.dropna(subset=["Estado"])
                       .groupby("CVE_ENT")
                       .agg(state_name=("Estado", "first"))
                       .reset_index()
                       .rename(columns={"CVE_ENT": "state"}))
state_names["state"]  =  state_names["state"].astype(str).str.zfill(2)

rows  =  []
for st in sorted(df["state"].dropna().unique()):
    sub  =  df[df["state"] == st]
    for ycol, sn in [("yield", "Combined"), ("yield_pv", "P-V")]:
        d  =  sub[sub[ycol].notna()].copy() if ycol == "yield_pv" else sub
        for pcol, plab in [("pred", "Raw"), ("pred_corr", "Corr.")]:
            s  =  d[[ycol, pcol, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 5: continue
            rows.append({
                "state":      st,
                "season":     sn,
                "model":      plab,
                "N":          len(s),
                "n_mun":      s.muncode.nunique(),
                "R2":         _r2(s[ycol], s[pcol]),
                "Between_R2": between_r2(s, ycol, pcol),
                "Within_R2":  within_r2(s, ycol, pcol),
                "RMSE":       np.sqrt(((s[ycol] - s[pcol]) ** 2).mean()),
            })
state_tbl  =  pd.DataFrame(rows).merge(state_names, on="state", how="left")
state_tbl.to_csv(os.path.join(out_dir, "state_accuracy.csv"), index=False)
print(f"  rows: {len(state_tbl)}")

# Wide LaTeX (one row per state, Raw vs Corr in subcolumns, Combined only)
wide  =  state_tbl[state_tbl["season"] == "Combined"].pivot_table(
    index=["state", "state_name"], columns="model",
    values=["N", "R2", "Between_R2", "Within_R2", "RMSE"])
wide  =  wide.sort_values(("Within_R2", "Raw"), ascending=False)
with open(os.path.join(out_dir, "state_accuracy.tex"), "w") as f:
    f.write("% State-level accuracy: AEF Hist Ens. Raw vs Corr., combined season, INEGI 2022 maize\n")
    f.write("\\begin{tabular}{ll rrrrr | rrrrr}\n")
    f.write("\\hline\n")
    f.write("State & Code & \\multicolumn{5}{c}{Raw} & \\multicolumn{5}{c}{Corrected} \\\\\n")
    f.write("\\cline{3-7} \\cline{8-12}\n")
    f.write(" & & $N$ & $R^2$ & Btw & Wtn & RMSE & $N$ & $R^2$ & Btw & Wtn & RMSE \\\\\n")
    f.write("\\hline\n")
    for (st, name), row in wide.iterrows():
        def fmtN(v):
            try: return f"{int(v):,}"
            except Exception: return "---"
        def fmtX(v, d=3):
            try: return f"{v:.{d}f}" if np.isfinite(v) else "---"
            except Exception: return "---"
        nm  =  (str(name)[:22] if pd.notna(name) else "")
        f.write(f"{nm} & {st} & "
                 f"{fmtN(row[('N','Raw')])} & "
                 f"{fmtX(row[('R2','Raw')])} & "
                 f"{fmtX(row[('Between_R2','Raw')])} & "
                 f"{fmtX(row[('Within_R2','Raw')])} & "
                 f"{fmtX(row[('RMSE','Raw')])} & "
                 f"{fmtN(row[('N','Corr.')])} & "
                 f"{fmtX(row[('R2','Corr.')])} & "
                 f"{fmtX(row[('Between_R2','Corr.')])} & "
                 f"{fmtX(row[('Within_R2','Corr.')])} & "
                 f"{fmtX(row[('RMSE','Corr.')])} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(f"  CSV: state_accuracy.csv   LaTeX: state_accuracy.tex (not copied to Overleaf)")

# ────────────────────────────────────────────────────────
# 10. 2D heatmaps of pooled within-R² (irrig × N_adc, irrig × maize share)
# ────────────────────────────────────────────────────────
print("\n=== 10. 2D within-R² heatmaps ===")
agland  =  pd.read_csv(agland_csv)
agland["muncode"]  =  agland["adc07"].str[:5]
mun_ag  =  (agland.groupby("muncode", as_index=False)
                   .agg(siap_irrig_area=("siap_irrig_area", "sum"),
                        siap_agland_area=("siap_agland_area", "sum")))
mun_ag["irrig_share"]  =  (mun_ag["siap_irrig_area"]
                            / mun_ag["siap_agland_area"].replace(0, np.nan))

siap_22  =  siap[siap["year"] == 2022].copy()
siap_22["is_maize"]  =  siap_22["name"].astype(str).str.lower().eq("maize")
siap_22["maize_ha"]  =  np.where(siap_22["is_maize"], siap_22["ha_planted"], 0.0)
mun_share  =  (siap_22.groupby("muncode", as_index=False)
                       .agg(maize_ha=("maize_ha", "sum"),
                            total_ha=("ha_planted", "sum")))
mun_share["maize_share"]  =  mun_share["maize_ha"] / mun_share["total_ha"]

mun_x  =  (df.dropna(subset=["pred", "yield"])
              .groupby("muncode")
              .size().rename("n_rows").reset_index())
mun_x  =  mun_x.merge(mun_ag[["muncode", "irrig_share"]], on="muncode", how="left")
mun_x  =  mun_x.merge(mun_share[["muncode", "maize_share"]],
                        on="muncode", how="left")

# Pooled within-R² inside a (mun-grouped) bin of muns:
def pooled_within_r2_in_muns(muns):
    s  =  df[df["muncode"].isin(muns)].dropna(subset=["pred", "yield"])
    c  =  s.groupby("muncode").size()
    s  =  s[s["muncode"].isin(c[c >= MIN_ADCS_FOR_R2].index)]
    if len(s) == 0: return np.nan, 0, 0
    gm  =  s.groupby("muncode")[["yield", "pred"]].transform("mean")
    yd  =  s["yield"] - gm["yield"]
    yhd =  s["pred"]  - gm["pred"]
    ss_tot  =  (yd ** 2).sum()
    ss_res  =  ((yd - yhd) ** 2).sum()
    return (1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            len(s), s.muncode.nunique())

def heatmap_2d(xcol, xbins, xlab, ycol, ybins, ylab, fname,
                xticklabels=None, yticklabels=None):
    """Plot a 2D heatmap using uniform cell sizes (ignore bin widths) so
    irregular bins read cleanly."""
    d  =  mun_x[[xcol, ycol, "muncode"]].dropna()
    d["xbin"]  =  pd.cut(d[xcol], xbins, include_lowest=True)
    d["ybin"]  =  pd.cut(d[ycol], ybins, include_lowest=True)
    nx, ny  =  len(xbins) - 1, len(ybins) - 1
    grid_r2  =  np.full((nx, ny), np.nan)
    grid_nm  =  np.zeros((nx, ny), dtype=int)
    grid_n   =  np.zeros((nx, ny), dtype=int)
    for i, xb in enumerate(d["xbin"].cat.categories):
        for j, yb in enumerate(d["ybin"].cat.categories):
            muns  =  d[(d["xbin"] == xb) & (d["ybin"] == yb)]["muncode"]
            if len(muns) < 3: continue
            r2v, n, nm  =  pooled_within_r2_in_muns(muns)
            grid_r2[i, j]  =  r2v
            grid_n[i, j]   =  n
            grid_nm[i, j]  =  nm

    fig, ax  =  plt.subplots(figsize=(8.5, 6.5))
    norm  =  TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.5)
    # Use index-coords (uniform cells)
    im  =  ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), grid_r2.T,
                           cmap="RdBu_r", norm=norm, edgecolors="white",
                           linewidth=0.4)
    for i in range(nx):
        for j in range(ny):
            v  =  grid_r2[i, j]
            if not np.isnan(v):
                ax.text(i + 0.5, j + 0.5,
                         f"{v:+.2f}\n(n={grid_nm[i, j]})",
                         ha="center", va="center", fontsize=8,
                         color=("white" if abs(v) > 0.3 else "black"))
    # Tick labels: bin ranges
    def _lab(edges, custom):
        if custom is not None: return custom
        return [f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
                if edges[-1] > 1.01 or i == len(edges) - 2
                else f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
                for i in range(len(edges) - 1)]
    ax.set_xticks(np.arange(nx) + 0.5)
    ax.set_xticklabels(_lab(xbins, xticklabels), rotation=25, ha="right")
    ax.set_yticks(np.arange(ny) + 0.5)
    ax.set_yticklabels(_lab(ybins, yticklabels))
    cbar  =  fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Pooled within-mun R² in bin")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"Pooled within-R² by {xlab} × {ylab}\n"
                  f"(cell label: R²  (n muns); cells with <3 muns blank)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight", dpi=160)
    plt.close(fig)

ibins  =  np.array([0, 0.05, 0.15, 0.30, 0.50, 0.80, 1.001])
ilabs  =  ["0–5%", "5–15%", "15–30%", "30–50%", "50–80%", "80–100%"]
nbins  =  np.array([1, 4, 10, 25, 60, 200, 1500])
nlabs  =  ["1–3", "4–9", "10–24", "25–59", "60–199", "200+"]
sbins  =  np.array([0, 0.10, 0.25, 0.45, 0.65, 0.85, 1.001])
slabs  =  ["0–10%", "10–25%", "25–45%", "45–65%", "65–85%", "85–100%"]
heatmap_2d("irrig_share",  ibins, "Irrigation share (SIAP)",
            "n_rows",      nbins, "INEGI rows per mun",
            "fig9a_within_r2_irrig_x_nadc_heatmap.png",
            xticklabels=ilabs, yticklabels=nlabs)
heatmap_2d("irrig_share",  ibins, "Irrigation share (SIAP)",
            "maize_share", sbins, "Maize share (SIAP)",
            "fig9b_within_r2_irrig_x_maizeshare_heatmap.png",
            xticklabels=ilabs, yticklabels=slabs)

print(f"\nAll outputs → {out_dir}")
