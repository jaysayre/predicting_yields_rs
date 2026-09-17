"""
Figure: end-to-end pipeline diagram (companion to the Figure 2 methodology diagram).

Five sections, no outlined boxes (soft fills only):
  1  Input data              - AEF annual embeddings, Landsat NDVI, cropland mask, spatial units
  2  Model training          - AEF Hist / AEF Hist Ensemble / AEF Mean / NDVI Hist
  3  Prediction & transfer   - same features at every scale: Guanajuato municipalities ->
                               Penjamo ADCs -> CIMMYT plots (real geometries)
  4  Ex-post correction      - (a) aggregate-consistency correction, (b) shrinkage; each shown
                               as before/after maps of Penjamo's ADC predictions (real values)
  5  Accuracy metrics        - overall R2 and within-municipality R2, on the real ADC evaluation

The canvas is 12 in wide and is placed at \textwidth (6.5 in) in the paper, so text set at
12 pt here prints at ~6.5 pt; keep lines short (the column widths are hand-wrapped).

Inputs:  Data/predictions/adc_aef_hist_ens_eval.parquet   (adc, muncode, yield, pred, pred_corr)
         Data/Shapefiles/adc_shapefile.shp                 (CA2007 ADC polygons, WGS84)
         Data/muncodes/shp/MUNICIPIOS.shp                  (municipality polygons, LCC)
         Intermediates/CIMMYT/cimmyt_plot_locations.shp    (CIMMYT plot points)
         Data/alpha_earth/aef_chip_mun03001_2022.npy       (AEF chip for the input-data panel)
Output:  plots/fig_pipeline_diagram.pdf (+ .png)
Run:     ~/miniforge3/envs/geo_env/bin/python 16_fig_pipeline_diagram.py
"""
import os
import subprocess

os.environ.setdefault("PROJ_LIB", os.path.join(
    os.path.expanduser("~"), "miniforge3", "envs", "geo_env", "share", "proj"))

import numpy as np
import pandas as pd
import geopandas as gpd

from   matplotlib.patches import Rectangle, FancyArrowPatch
from   matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib.pyplot as plt

base_font_size = 12  # match \documentclass[12pt]{article}
from cycler import cycler
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.unicode_minus": False,
    "font.size": base_font_size,
    "axes.titlesize": base_font_size + 3,
    "axes.labelsize": base_font_size + 2,
    "xtick.labelsize": base_font_size,
    "ytick.labelsize": base_font_size,
    "legend.fontsize": base_font_size,
    "figure.titlesize": base_font_size + 3,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#404040",
    "axes.labelcolor": "#404040",
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "grid.color": "#D0D0D0",
    "grid.linestyle": (0, (1, 3)),
    "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#4A4A4A"]),
    "pgf.preamble": r"""
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath}
""",
})

# -- Directories --------------------------------------------
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
pred_dir  =  os.path.join(proj_dir, "Data", "predictions")
aef_dir   =  os.path.join(proj_dir, "Data", "alpha_earth")
plot_dir  =  os.path.join(proj_dir, "plots")

# -- Inputs ---------------------------------------------------
eval_path =  os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet")           # ADC census yield + raw/corrected ensemble preds
adc_shp   =  os.path.join(proj_dir, "Data", "Shapefiles", "adc_shapefile.shp")  # CA2007 ADC polygons (WGS84)
mun_shp   =  os.path.join(proj_dir, "Data", "muncodes", "shp", "MUNICIPIOS.shp")  # municipality polygons (LCC)
cim_shp   =  os.path.join(proj_dir, "Intermediates", "CIMMYT", "cimmyt_plot_locations.shp")  # CIMMYT plot points
chip_path =  os.path.join(aef_dir, "aef_chip_mun03001_2022.npy")               # AEF chip (Comondu window, as in Fig. 2)

# -- Outputs --------------------------------------------------
out_pdf   =  os.path.join(plot_dir, "fig_pipeline_diagram.pdf")
out_png   =  os.path.join(plot_dir, "fig_pipeline_diagram.png")

MUN       =  "11023"          # Penjamo, Guanajuato: 557 ADCs with predictions and CIMMYT plots
STATE     =  "11"
LAM       =  0.72             # deployed shrinkage factor (within-municipality shrinkage section)
CHIP_DIM  =  11               # embedding dimension shown in Fig. 2

GRAY      =  "#4A4A4A"
MUTED     =  "#7A7A7A"
FILL      =  "white"          # section background (thin black outline)
ACCENT    =  "#0072B2"
ACCENT2   =  "#D55E00"
yld_cmap  =  mpl.colormaps["viridis"]
dev_cmap  =  mpl.colormaps["RdBu_r"]

# -- Data -----------------------------------------------------
ev  =  pd.read_parquet(eval_path)[["adc", "muncode", "yield", "pred", "pred_corr"]]
ev  =  ev.dropna(subset=["yield", "pred"])
pen =  ev[ev["muncode"] == MUN].copy()
gm  =  pen["pred"].mean()
pen["pred_shrink"] =  gm + LAM * (pen["pred"] - gm)
e_m =  float((pen["pred"] - pen["pred_corr"]).median())        # additive municipal error removed by the correction
print(f"Penjamo: {len(pen)} ADCs, e_m = {e_m:+.2f} t/ha, mean raw pred {gm:.2f}")

muns   =  gpd.read_file(mun_shp)
muns["muncode"] =  muns["CVE_ENT"].astype(str).str.zfill(2) + muns["CVE_MUN"].astype(str).str.zfill(3)
gto    =  muns[muns["CVE_ENT"].astype(str).str.zfill(2) == STATE].copy()
gto["geometry"] =  gto["geometry"].simplify(200)
pen_g  =  gto[gto["muncode"] == MUN]
bb     =  pen_g.to_crs(4326).total_bounds
adcs   =  gpd.read_file(adc_shp, bbox=tuple(bb))
adcs["adc"] =  adcs["adcid"].str.replace("-", "", regex=False)
adcs   =  adcs[adcs["adc"].str[:5] == MUN][["adc", "geometry"]].to_crs(muns.crs)
adcs   =  adcs.merge(pen[["adc", "pred", "pred_corr", "pred_shrink"]], on="adc", how="left")
adcs["dev_raw"]    =  adcs["pred"] - gm
adcs["dev_shrink"] =  adcs["pred_shrink"] - gm
print(f"  {len(adcs)} ADC polygons, {adcs['pred'].notna().sum()} with predictions")
cim    =  gpd.read_file(cim_shp)
cim    =  cim[(cim["id_state"] == 11) & (cim["id_mun"] == 23)].to_crs(muns.crs)
print(f"  {len(cim)} CIMMYT plots in Penjamo")

chip   =  np.load(chip_path)[CHIP_DIM]
lo, hi =  np.nanpercentile(chip, [2, 98]); chip = np.clip((chip - lo) / (hi - lo + 1e-9), 0, 1)

vmax   =  float(np.nanquantile(pd.concat([adcs["pred"], adcs["pred_corr"], adcs["pred_shrink"]]), 0.98))
vnorm  =  Normalize(vmin=0, vmax=vmax)
dmax   =  float(np.nanquantile(adcs["dev_raw"].abs(), 0.98))
dnorm  =  Normalize(vmin=-dmax, vmax=dmax)

# ============================================================
fig = plt.figure(figsize=(12.0, 15.0))
fs  = base_font_size              # body text (prints at ~6.5 pt in the paper)
fss = base_font_size - 1.5        # small text
fh  = base_font_size + 3          # section headers


def section(x0, y0, x1, y1, title):
    """White section background with a thin black outline and a bold title at its top-left."""
    fig.patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                 facecolor=FILL, edgecolor="black", linewidth=0.9,
                                 zorder=-10))   # behind the axes (zorder 0)
    fig.text(x0 + 0.012, y1 - 0.011, title, ha="left", va="top", fontsize=fh,
             fontweight="bold", color=GRAY)


def ftext(x, y, s, **kw):
    kw.setdefault("fontsize", fs); kw.setdefault("color", GRAY); kw.setdefault("ha", "left"); kw.setdefault("va", "top")
    kw.setdefault("linespacing", 1.25)
    return fig.text(x, y, s, **kw)


def farrow(x0, y0, x1, y1, color=MUTED, lw=1.3):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, zorder=3))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("none")


def map_axes(rect, gdf, column=None, cmap=None, norm=None, **kw):
    ax = fig.add_axes(rect)
    if column is None:
        gdf.plot(ax=ax, facecolor="#D0D0D0", edgecolor="white", linewidth=0.4, rasterized=True, **kw)
    else:
        gdf.plot(ax=ax, column=column, cmap=cmap or yld_cmap, norm=norm or vnorm, edgecolor="white", linewidth=0.15,
                 missing_kwds={"color": "#E8E8E8"}, rasterized=True, **kw)
    ax.set_aspect("equal"); clean(ax)
    return ax


# Row / column geometry (figure fractions)
R1 = (0.615, 0.985)     # row 1: input data | model training
R2 = (0.290, 0.600)     # row 2: scale transfer | ex-post correction
R3 = (0.015, 0.275)     # row 3: accuracy metrics
L  = (0.015, 0.360)     # left column
R  = (0.375, 0.985)     # right column

# ------------------------------------------------------------
# 1  Input data
# ------------------------------------------------------------
section(L[0], R1[0], L[1], R1[1], r"1 $\cdot$ Input data")
y = R1[1] - 0.040
ftext(L[0] + 0.012, y, r"\textbf{Satellite data}")
ax = fig.add_axes([L[0] + 0.014, y - 0.082, 0.072, 0.062])
ax.imshow(chip, cmap="viridis", interpolation="nearest"); clean(ax)
ftext(L[0] + 0.014, y - 0.084, f"AEF, dim A{CHIP_DIM}", fontsize=fss, color=MUTED)
t    = np.linspace(0, 1, 200)
ndvi = 0.25 + 0.45 * np.exp(-((t - 0.62) / 0.16) ** 2)
ax = fig.add_axes([L[0] + 0.014, y - 0.150, 0.072, 0.045])
ax.plot(t, ndvi, color=GRAY, lw=1.2)
for tt in [0.45, 0.62, 0.79]:
    ax.scatter([tt], [0.25 + 0.45 * np.exp(-((tt - 0.62) / 0.16) ** 2)], s=16, color=ACCENT2, zorder=3)
ax.set_ylim(0.1, 0.8); clean(ax)
ftext(L[0] + 0.014, y - 0.152, "NDVI, harmonic fit", fontsize=fss, color=MUTED)
ftext(L[0] + 0.104, y - 0.018,
      "AlphaEarth Foundations (AEF)\nannual embeddings: 64 bands,\n10 m, 2017--2024\n\n"
      "Landsat 7/8 NDVI: per-pixel\n3rd-order harmonic fit, read\n1, 3 and 5 months after\nlocal planting")

y2 = y - 0.170
ftext(L[0] + 0.012, y2, r"\textbf{Cropland mask}")
ax = fig.add_axes([L[0] + 0.014, y2 - 0.068, 0.05, 0.05])
rng = np.random.default_rng(3); m = rng.random((6, 6)) < 0.55
ax.imshow(np.where(m, 1.0, 0.0), cmap=mpl.colors.ListedColormap(["#E0E0E0", "#7FB069"]), interpolation="nearest")
clean(ax)
ftext(L[0] + 0.076, y2 - 0.018,
      "ESA WorldCover v200 (2021) cropland\nclass; only cropland pixels enter the\nfeatures. The mask does not identify\nmaize fields.")

y3 = y2 - 0.084
ftext(L[0] + 0.012, y3, r"\textbf{Spatial units}")
ftext(L[0] + 0.012, y3 - 0.018,
      r"Municipalities (admin-2): training labels" "\n"
      r"\'Areas de control (admin-4): validation" "\n"
      r"CIMMYT farmer plots: validation")

# ------------------------------------------------------------
# 2  Model training
# ------------------------------------------------------------
section(R[0], R1[0], R[1], R1[1], r"2 $\cdot$ Model training")
cards = [
    ("a", r"\textbf{AEF Hist}",
     "8 fixed-width bins per dimension\n($64 \\times 8 = 512$ features), trained\n"
     "on $K=5$ draws of $N=2$ pixels\nper dim (ADC-like inputs)\nHistGradientBoosting"),
    ("b", r"\textbf{AEF Hist Ensemble}",
     "Percentile model: per dimension\nmean, p10, p25, p50, p75, p90\nand SD ($64 \\times 7 = 448$ features)\n"
     r"$0.5 \times$ AEF Hist $+\, 0.5 \times$ percentile"),
    ("c", r"\textbf{AEF Mean}",
     "64 per-band means only\n(Ma et al., 2026)\nRandom forest"),
    ("d", r"\textbf{NDVI Hist}",
     "3 growth-stage NDVI values\n+ 7 harmonic coefficients\n= 10 features, summarized\nwith the same distributional\nrecipe (150 inputs)\nGradient boosting"),
]
cx = [R[0] + 0.012, R[0] + 0.315]
cy = [R1[1] - 0.042, R1[1] - 0.190]
for i, (k, ttl, body) in enumerate(cards):
    x0 = cx[i % 2]; y0 = cy[i // 2]
    ftext(x0, y0, f"({k}) " + ttl)
    ftext(x0, y0 - 0.022, body)
    gx = x0 + 0.205; gw = 0.068; gh = 0.055
    ax = fig.add_axes([gx, y0 - 0.086, gw, gh])
    if k == "b":     # density with percentile ticks
        xs = np.linspace(-0.4, 0.4, 200); d = np.exp(-((xs - 0.03) / 0.11) ** 2)
        ax.fill_between(xs, 0, d, color="#C9D9EA"); ax.plot(xs, d, color=ACCENT, lw=1)
        for q in [-0.11, -0.05, 0.03, 0.11, 0.17]:
            ax.axvline(q, color=ACCENT, lw=0.6, ls=(0, (2, 2)))
        ax.set_ylim(0, 1.15); clean(ax)
        ax.set_title("percentiles, per dim", fontsize=fss, color=MUTED, pad=2)
    elif k == "a":   # smooth mun histogram vs spiky subsampled draw
        ctr = np.arange(8)
        ax.bar(ctr - 0.2, [0, 0.02, 0.2, 0.45, 0.25, 0.06, 0.02, 0], width=0.38, color="#B8B8B8")
        ax.bar(ctr + 0.2, [0, 0, 0, 0.5, 0.5, 0, 0, 0], width=0.38, color=ACCENT)
        ax.set_ylim(0, 0.6); clean(ax)
        ax.set_title("mun.\\ vs.\\ $N{=}2$ draw", fontsize=fss, color=MUTED, pad=2)
    elif k == "c":   # 1 x 64 strip of means
        ax.imshow(np.random.default_rng(1).normal(0, 0.12, (1, 64)), cmap="viridis", aspect="auto",
                  interpolation="nearest"); clean(ax)
        ax.set_title(r"$1 \times 64$ mean vector", fontsize=fss, color=MUTED, pad=2)
    else:            # NDVI curve, three stages
        ax.plot(t, ndvi, color=GRAY, lw=1.2)
        for tt in [0.45, 0.62, 0.79]:
            ax.scatter([tt], [0.25 + 0.45 * np.exp(-((tt - 0.62) / 0.16) ** 2)], s=16, color=ACCENT2, zorder=3)
        ax.set_ylim(0.1, 0.8); clean(ax)
        ax.set_title("3 growth stages", fontsize=fss, color=MUTED, pad=2)
ftext(R[0] + 0.012, R1[0] + 0.030,
      "All models are trained on DGSIAP municipal maize yields (spring--summer, 2017--2024).",
      fontsize=fss, color=MUTED)

# ------------------------------------------------------------
# 3  Prediction and scale transfer
# ------------------------------------------------------------
section(L[0], R2[0], L[1], R2[1], r"3 $\cdot$ Prediction and scale transfer")
ftext(L[0] + 0.012, R2[1] - 0.042,
      "Every feature is a distributional summary of\na unit's cropland pixels, so its length is the\n"
      "same at every scale: train once at the\nmunicipality scale and apply the same model,\nwithout modification, at any finer scale.")
mw = 0.100; my0 = R2[0] + 0.085; mh = 0.150
xs3 = [L[0] + 0.006 + k * (mw + 0.014) for k in range(3)]
ax = map_axes([xs3[0], my0, mw, mh], gto)
pen_g.plot(ax=ax, facecolor=yld_cmap(0.55), edgecolor="white", linewidth=0.4, rasterized=True)
ax = map_axes([xs3[1], my0, mw, mh], adcs)
ax = map_axes([xs3[2], my0, mw, mh], pen_g)
cim.plot(ax=ax, color=ACCENT2, markersize=1.2, alpha=0.7, rasterized=True)
for xk, lab in zip(xs3, ["Municipalities\n(Guanajuato; train)", f"ADCs of P\\'{{e}}njamo\n({len(adcs):,}; validate)",
                         f"CIMMYT plots\n({len(cim):,}; validate)"]):
    ftext(xk + mw / 2, my0 - 0.006, lab, ha="center", fontsize=fss, color=MUTED)
for k in range(2):
    farrow(xs3[k] + mw + 0.001, my0 + mh / 2, xs3[k + 1] - 0.001, my0 + mh / 2)

# ------------------------------------------------------------
# 4  Ex-post correction
# ------------------------------------------------------------
section(R[0], R2[0], R[1], R2[1], r"4 $\cdot$ Ex-post correction")
sub = [
    ("a", r"\textbf{Scale (aggregate-consistency) correction}",
     "Aggregate the ADC predictions\nwith ex-ante agricultural-land\nweights; compare with the\n"
     "DGSIAP municipal yield;\nsubtract the error $e_m$ from\nevery ADC in the municipality:\n"
     r"$\hat z_i^{\,corr} = \hat z_i - e_m$" f"   (P\\'{{e}}njamo: $e_m = {e_m:+.2f}$ t/ha)",
     "pred", "pred_corr", "raw", "corrected", None, None, vnorm, yld_cmap, "predicted yield (t/ha)"),
    ("b", r"\textbf{Within-municipality shrinkage}",
     "Keep each municipality's mean\nprediction; compress the ADC\ndeviations around it:\n"
     r"$\hat z_i^{\,shrink} = \bar{\hat z}_m + \lambda\,(\hat z_i - \bar{\hat z}_m)$" "\n"
     r"$\lambda = 0.72$ from public data (irrigation" "\n"
     "projection); no retraining and\nbetween-municipality accuracy unchanged",
     "dev_raw", "dev_shrink", "raw deviation", "shrunk deviation", dev_cmap, dnorm, dnorm, dev_cmap,
     "deviation from municipal mean (t/ha)"),
]
sw = 0.303
for i, (k, ttl, body, c0, c1, l0, l1, cm_, nm_, cbn, cbc, cbl) in enumerate(sub):
    x0 = R[0] + 0.012 + i * sw
    ftext(x0, R2[1] - 0.042, f"({k}) " + ttl)
    ftext(x0, R2[1] - 0.064, body)
    mw2 = 0.112; mh2 = 0.104; ym = R2[0] + 0.062
    a0 = map_axes([x0 + 0.010, ym, mw2, mh2], adcs, column=c0, cmap=cm_, norm=nm_)
    a1 = map_axes([x0 + mw2 + 0.030, ym, mw2, mh2], adcs, column=c1, cmap=cm_, norm=nm_)
    farrow(x0 + mw2 + 0.012, ym + mh2 / 2, x0 + mw2 + 0.028, ym + mh2 / 2)
    ftext(x0 + 0.010 + mw2 / 2, ym - 0.005, l0, ha="center", fontsize=fss, color=MUTED)
    ftext(x0 + mw2 + 0.030 + mw2 / 2, ym - 0.005, l1, ha="center", fontsize=fss, color=MUTED)
    cax = fig.add_axes([x0 + 0.045, R2[0] + 0.030, 0.17, 0.007])
    cb  = fig.colorbar(mpl.cm.ScalarMappable(norm=cbn, cmap=cbc), cax=cax, orientation="horizontal")
    cb.set_label(cbl, fontsize=fss, color=MUTED, labelpad=2)
    cb.ax.tick_params(labelsize=fss - 1, length=2); cb.outline.set_edgecolor("#B0B0B0")

    # schematic of what the correction does, top-right of the sub-block (below the title)
    gax = fig.add_axes([x0 + 0.212, 0.483, 0.078, 0.055]); clean(gax)
    gax.set_xlim(0, 1); gax.set_ylim(0, 1)
    if k == "a":                       # level shift: every ADC moves onto the DGSIAP line
        gd  = np.array([0.07, -0.10, 0.12, -0.05, 0.09, -0.13])
        gx1 = np.linspace(0.14, 0.40, 6); gx2 = gx1 + 0.48
        gax.plot([0.02, 0.98], [0.76, 0.76], color=ACCENT2, lw=1.0, zorder=2)
        gax.text(0.02, 0.83, "DGSIAP", ha="left", va="bottom", fontsize=fss - 1, color=ACCENT2)
        gax.plot([0.06, 0.48], [0.30, 0.30], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
        gax.scatter(gx1, 0.30 + gd, s=7, color=MUTED, zorder=3)
        gax.scatter(gx2, 0.76 + gd, s=7, color=ACCENT, zorder=3)
        gax.annotate("", xy=(0.53, 0.72), xytext=(0.53, 0.34),
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.9, mutation_scale=8))
        gax.text(0.50, 0.53, r"$e_m$", ha="right", va="center", fontsize=fss, color=GRAY)
    else:                              # compression: deviations pulled toward an unchanged mean
        gd  = np.array([0.32, -0.24, 0.15, -0.34, 0.26])
        gx1 = np.linspace(0.08, 0.30, 5); gx2 = gx1 + 0.36
        gax.plot([0.02, 0.98], [0.50, 0.50], color=ACCENT2, lw=1.0, zorder=2)
        gax.text(0.98, 0.47, "mean", ha="right", va="top", fontsize=fss - 2, color=ACCENT2)
        gax.scatter(gx1, 0.50 + gd, s=7, color=MUTED, zorder=3)
        gax.scatter(gx2, 0.50 + 0.42 * gd, s=7, color=ACCENT, zorder=3)
        for y0_, y1_ in [(0.82, 0.58), (0.18, 0.42)]:
            gax.annotate("", xy=(0.37, y1_), xytext=(0.37, y0_),
                         arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8, mutation_scale=7))
        gax.text(0.37, 1.00, r"$\times\,\lambda$", ha="center", va="top", fontsize=fss, color=GRAY)

# ------------------------------------------------------------
# 5  Accuracy metrics
# ------------------------------------------------------------
section(L[0], R3[0], R[1], R3[1], r"5 $\cdot$ Accuracy metrics")
smp = ev.sample(min(len(ev), 40000), random_state=0)
gmn = smp.groupby("muncode")[["yield", "pred"]].transform("mean")
dy, dp = smp["yield"] - gmn["yield"], smp["pred"] - gmn["pred"]
r2_all = 1 - np.sum((smp["yield"] - smp["pred"]) ** 2) / np.sum((smp["yield"] - smp["yield"].mean()) ** 2)
r2_w   = 1 - np.sum((dy - dp) ** 2) / np.sum(dy ** 2)
metrics = [
    ("a", r"\textbf{Overall $R^2$}",
     r"$R^2 = 1 - \dfrac{\sum_i (Y_i - \hat Y_i)^2}{\sum_i (Y_i - \bar Y)^2}$" "\n\n"
     "Fit against the ADC yields themselves;\na model that assigns every ADC its\nmunicipal average already scores\nwell here.",
     smp["yield"], smp["pred"], "census yield (t/ha)", "predicted (t/ha)", (0, 12), f"$R^2$ = {r2_all:.2f}"),
    ("b", r"\textbf{Within-municipality $R^2$}",
     r"$1 - \dfrac{\sum_m \sum_{i \in m} [(\hat Y_i - \bar{\hat Y}_m) - (Y_i - \bar Y_m)]^2}{\sum_m \sum_{i \in m} (Y_i - \bar Y_m)^2}$" "\n\n"
     "Every term is demeaned within its\nmunicipality: the share of sub-municipal\nvariation the model explains beyond\nwhat municipal statistics give.",
     dy, dp, "census yield, demeaned (t/ha)", "predicted, demeaned (t/ha)", (-5, 5), f"within-$R^2$ = {r2_w:.2f}"),
]
for i, (k, ttl, body, xv, yv, xl, yl, lim, lab) in enumerate(metrics):
    x0 = L[0] + 0.012 + i * 0.485
    ftext(x0, R3[1] - 0.042, f"({k}) " + ttl)
    ftext(x0, R3[1] - 0.072, body)
    ax = fig.add_axes([x0 + 0.315, R3[0] + 0.048, 0.135, 0.175])
    ax.hexbin(xv, yv, gridsize=35, bins="log", cmap="viridis", extent=(lim[0], lim[1], lim[0], lim[1]),
              linewidths=0, rasterized=True)
    ax.plot(lim, lim, "--", color="#B0B0B0", lw=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xl, fontsize=fss); ax.set_ylabel(yl, fontsize=fss)
    ax.tick_params(labelsize=fss - 1, length=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_facecolor("white")
    ax.set_title(lab, fontsize=fss, color=GRAY, pad=3)

# flow arrows between sections
farrow(L[1] + 0.001, (R1[0] + R1[1]) / 2, R[0] - 0.001, (R1[0] + R1[1]) / 2)
farrow((R[0] + R[1]) / 2, R1[0] - 0.001, (R[0] + R[1]) / 2, R2[1] + 0.001)
farrow(L[1] + 0.001, (R2[0] + R2[1]) / 2, R[0] - 0.001, (R2[0] + R2[1]) / 2)
farrow((R[0] + R[1]) / 2, R2[0] - 0.001, (R[0] + R[1]) / 2, R3[1] + 0.001)

os.makedirs(plot_dir, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
print("Wrote", out_pdf)
subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", out_pdf, out_png[:-4]], check=True)
print("Wrote", out_png)
