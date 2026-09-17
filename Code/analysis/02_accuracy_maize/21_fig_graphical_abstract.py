"""
Graphical abstract for the journal submission system (NOT a paper figure).

Condenses the featurization diagram (step 15) and the pipeline diagram (step 16)
into one landscape strip, left to right:
  Inputs -> Featurization -> Scale transfer -> Corrections -> Validation
The last stage reproduces three panels of the combined-season accuracy figure
(step 4): AEF Hist Ens. Shrink, DGSIAP Mun. Avg., NDVI Shrink. The data loading,
the deployed shrinkage and the DGSIAP municipal anchor are copied from
4_accuracy_main_2022.py, so the R2 / within-R2 printed here are the numbers of
Table \ref{tab:accuracy_combined} (nothing is typed in by hand).

Elsevier requires 531 x 1328 px (h x w) or proportionally larger, legible at
5 x 13 cm. The canvas is therefore exactly 2 x (13 x 5 cm) = 10.24 x 3.94 in, no
text is set below 14 pt (>= 7 pt when printed at 13 x 5 cm), and the PNG is
rendered at 300 dpi -> 3073 x 1182 px.

Inputs:  Data/predictions/adc_aef_hist_ens_eval.parquet    (ADC census yield + ensemble preds)
         Data/predictions/adc_aefn2_masked_preds.parquet   (NDVI masked baseline preds)
         Data/SIAP/Cleaned/siap_ag_prod_estimation_by_season.dta  (DGSIAP municipal yields)
         Data/muncodes/shp/MUNICIPIOS.shp                  (municipality polygons, LCC)
         Data/Shapefiles/adc_shapefile.shp                 (CA2007 ADC polygons, WGS84)
         Data/alpha_earth/aef_chip_mun03001_2022.npy       (AEF chip, Comondu window)
         Data/alpha_earth/alpha_earth_mex_mun_binned_hist.parquet  (64 x 8 bin shares)
Output:  plots/graphical_abstract.pdf (+ .png, 3073 x 1182 px)
Run:     ~/miniforge3/envs/geo_env/bin/python 21_fig_graphical_abstract.py
"""
import os
import subprocess
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROJ_LIB", os.path.join(
    os.path.expanduser("~"), "miniforge3", "envs", "geo_env", "share", "proj"))

import numpy as np
import pandas as pd
import geopandas as gpd

from   matplotlib.patches import Rectangle, FancyArrowPatch
from   matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib.pyplot as plt

base_font_size = 14  # floor for the abstract: 14 pt here -> 7 pt at 13 x 5 cm
from cycler import cycler
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.unicode_minus": False,
    "font.size": base_font_size,
    "axes.titlesize": base_font_size,
    "axes.labelsize": base_font_size,
    "xtick.labelsize": base_font_size,
    "ytick.labelsize": base_font_size,
    "legend.fontsize": base_font_size,
    "figure.titlesize": base_font_size,
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

# -- Directories ----------------------------------------------
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
pred_dir  =  os.path.join(proj_dir, "Data", "predictions")
aef_dir   =  os.path.join(proj_dir, "Data", "alpha_earth")
plot_dir  =  os.path.join(proj_dir, "plots")

# -- Inputs ---------------------------------------------------
eval_path =  os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet")            # ADC census yield + ensemble preds
ndvi_path =  os.path.join(pred_dir, "adc_aefn2_masked_preds.parquet")           # NDVI (masked) baseline preds
siap_path =  os.path.join(proj_dir, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")  # DGSIAP municipal yields
mun_shp   =  os.path.join(proj_dir, "Data", "muncodes", "shp", "MUNICIPIOS.shp")   # municipality polygons (LCC)
adc_shp   =  os.path.join(proj_dir, "Data", "Shapefiles", "adc_shapefile.shp")     # CA2007 ADC polygons (WGS84)
chip_path =  os.path.join(aef_dir, "aef_chip_mun03001_2022.npy")                # AEF chip (Comondu window)
hist_path =  os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet")    # 64 x 8 bin shares per municipality

# -- Outputs --------------------------------------------------
out_pdf   =  os.path.join(plot_dir, "graphical_abstract.pdf")
out_png   =  os.path.join(plot_dir, "graphical_abstract.png")

YEAR      =  2022
MUN       =  "11023"        # Penjamo, Guanajuato (scale-transfer illustration)
STATE     =  "11"           # Guanajuato
CHIP_MUN  =  "03001"        # Comondu, BCS (chip + fingerprint shown in stages 1-2)
LAM_DEPLOY =  0.72          # deployed shrinkage factor (Sec. 3.6), as in step 4
N_DIMS, N_BINS =  64, 8

GRAY   =  "#4A4A4A"
MUTED  =  "#7A7A7A"
ACCENT =  "#0072B2"
ACCENT2 = "#D55E00"
yld_cmap =  mpl.colormaps["viridis"]

FS  =  base_font_size          # body text (never smaller)
FH  =  base_font_size + 1      # stage headers

# ============================================================
# Data: accuracy panels (logic copied from 4_accuracy_main_2022.py)
# ============================================================
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    return 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2) if m.sum() > 1 else np.nan


def within_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(gc).size(); s = s[s[gc].isin(c[c >= 2].index)]
    if len(s) == 0: return np.nan
    gm = s.groupby(gc)[[y, p]].transform("mean"); return r2(s[y] - gm[y], s[p] - gm[p])


def shrink(df, pcol, lam, gc="muncode"):
    g = df.groupby(gc)[pcol]; return g.transform("mean") + lam * (df[pcol] - g.transform("mean"))


def fmt(v):
    if v is None or not np.isfinite(v): return "---"
    s = f"{v:.2f}"
    return s.replace("-", "$-$") if v < 0 else s


ev =  pd.read_parquet(eval_path)[["adc", "muncode", "yield", "pred"]].rename(
        columns={"pred": "AEF Hist Ens."})
ev =  ev[ev["AEF Hist Ens."].notna()].copy()        # ensemble-eligible sample, as in step 4
nd =  pd.read_parquet(ndvi_path)
nd =  nd[nd["year"] == YEAR].copy()
nd["adc"] =  nd["adcid"].astype(str).str.replace("-", "", regex=False)
ev =  ev.merge(nd[["adc", "pred"]].rename(columns={"pred": "NDVI"}).dropna().drop_duplicates("adc"),
               on="adc", how="left")

siap =  pd.read_stata(siap_path)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
s22  =  siap[(siap["name"] == "Maize") & (siap["year"] == YEAR)]
s22  =  s22[~s22["muncode"].str.endswith("000")]
anch =  s22.groupby("muncode").agg(q=("q", "sum"), ha=("ha_planted", "sum")).reset_index()
anch["siap"] =  anch["q"] / anch["ha"]
anchor =  anch.set_index("muncode")["siap"]           # combined-season DGSIAP anchor
ev["siap"] =  ev["muncode"].map(anchor)

panels = []
for nm, lab in [("AEF Hist Ens.", "AEF Hist\nEns.\\ Shrink"), ("DGSIAP", "DGSIAP\nmun.\\ avg."),
                ("NDVI", "NDVI\nShrink")]:
    if nm == "DGSIAP":
        c = "siap"
    else:
        c = f"_{nm}_shrink"; ev[c] = shrink(ev, nm, LAM_DEPLOY)
    ov =  r2(ev["yield"].values, ev[c].values)
    wt =  within_r2(ev, "yield", c)
    if np.isfinite(wt) and abs(wt) < 5e-4: wt = 0.0
    panels.append((c, lab, ov, wt))
    print(f"  {lab.splitlines()[0]:16s} R2={ov:.3f}  within-R2={wt:.3f}")
print(f"  {len(ev):,} ADCs, {ev['muncode'].nunique():,} municipalities")

# ============================================================
# Data: maps and glyph assets (from 15/16_fig_*_diagram.py)
# ============================================================
muns =  gpd.read_file(mun_shp)
muns["muncode"] =  muns["CVE_ENT"].astype(str).str.zfill(2) + muns["CVE_MUN"].astype(str).str.zfill(3)
mex  =  muns.copy()
mex["geometry"] =  mex["geometry"].simplify(1500)     # national view: coarse is enough
mex =  mex.merge(anchor.rename("yield").reset_index(), on="muncode", how="left")
ynorm =  Normalize(vmin=float(mex["yield"].quantile(0.02)), vmax=float(mex["yield"].quantile(0.98)))

gto  =  muns[muns["CVE_ENT"].astype(str).str.zfill(2) == STATE].copy()
gto["geometry"] =  gto["geometry"].simplify(200)
pen_g =  gto[gto["muncode"] == MUN]
bb   =  pen_g.to_crs(4326).total_bounds
adcs =  gpd.read_file(adc_shp, bbox=tuple(bb))
adcs["adc"] =  adcs["adcid"].str.replace("-", "", regex=False)
adcs =  adcs[adcs["adc"].str[:5] == MUN][["adc", "geometry"]].to_crs(muns.crs)
adcs =  adcs.merge(ev[["adc", panels[0][0]]].rename(columns={panels[0][0]: "pred"}), on="adc", how="left")
pnorm =  Normalize(vmin=0, vmax=float(np.nanquantile(adcs["pred"], 0.98)))
print(f"  Penjamo: {len(adcs)} ADC polygons, {adcs['pred'].notna().sum()} with predictions")

bin_cols =  [f"A{d:02d}_b{b}" for d in range(N_DIMS) for b in range(N_BINS)]
bh   =  pd.read_parquet(hist_path)
bh   =  bh[bh["year"] == YEAR].set_index("muncode")
fp   =  bh.loc[CHIP_MUN, bin_cols].values.astype(float).reshape(N_DIMS, N_BINS)
# display dimension: the one whose bin shares are most spread out (a one-bar
# histogram would read as a mistake at this size)
DIM  =  int((-(fp * np.log(fp + 1e-12))).sum(axis=1).argmax())
print(f"  display dim A{DIM:02d}")

chip =  np.load(chip_path)[DIM]
_lo, _hi =  np.nanpercentile(chip, [2, 98])
chip =  np.clip((chip - _lo) / (_hi - _lo + 1e-9), 0, 1)

# ============================================================
# Canvas: exactly 2 x (13 x 5 cm); all geometry specified in inches
# ============================================================
FW, FH_IN =  10.24, 3.94
fig =  plt.figure(figsize=(FW, FH_IN))


def rect(x, y, w, h):
    """inches (origin bottom-left) -> figure-fraction rect"""
    return [x / FW, y / FH_IN, w / FW, h / FH_IN]


def box(x, y, w, h, title):
    fig.patches.append(Rectangle((x / FW, y / FH_IN), w / FW, h / FH_IN,
                                 transform=fig.transFigure, facecolor="white",
                                 edgecolor="black", linewidth=0.9, zorder=-10))
    fig.text((x + 0.10) / FW, (y + h - 0.09) / FH_IN, title, ha="left", va="top",
             fontsize=FH, fontweight="bold", color="black")


def txt(x, y, s, **kw):
    kw.setdefault("fontsize", FS); kw.setdefault("color", GRAY)
    kw.setdefault("ha", "left"); kw.setdefault("va", "top"); kw.setdefault("linespacing", 1.30)
    return fig.text(x / FW, y / FH_IN, s, **kw)


def arrow(x0, y0, x1, y1, color=GRAY, lw=1.4, ms=15):
    fig.patches.append(FancyArrowPatch((x0 / FW, y0 / FH_IN), (x1 / FW, y1 / FH_IN),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=ms, color=color, lw=lw, zorder=5))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("none")


# stage geometry ------------------------------------------------
M, GAP, WR =  0.06, 0.19, 3.45
W  =  (FW - 2 * M - 4 * GAP - WR) / 4          # 1.478 in per narrow stage
X  =  [M + i * (W + GAP) for i in range(4)] + [M + 4 * (W + GAP)]
BY, BH_ =  0.08, 3.78                          # box bottom, box height
TOP =  BY + BH_                                # 3.86
CTOP =  TOP - 0.44                             # first content line

for i in range(4):
    arrow(X[i] + W + 0.02, BY + BH_ / 2, X[i + 1] - 0.02, BY + BH_ / 2)

# ------------------------------------------------------------
# 1  Inputs
# ------------------------------------------------------------
box(X[0], BY, W, BH_, "Inputs")
txt(X[0] + 0.10, CTOP, "Public\nmunicipal maize\nyields")
axm =  fig.add_axes(rect(X[0] + 0.07, 1.98, W - 0.14, 0.92))
mex.plot(ax=axm, column="yield", cmap=yld_cmap, norm=ynorm, edgecolor="none",
         linewidth=0, missing_kwds={"color": "#E8E8E8"}, rasterized=True)
axm.set_aspect("equal"); clean(axm)
txt(X[0] + 0.10, 1.88, "AlphaEarth\nembeddings,\n10 m pixels")
axc =  fig.add_axes(rect(X[0] + (W - 0.86) / 2, 0.32, 0.86, 0.86))
axc.imshow(chip, cmap="viridis", interpolation="nearest"); clean(axc)
for s in axc.spines.values():
    s.set_visible(True); s.set_color("#B0B0B0"); s.set_linewidth(0.6)

# ------------------------------------------------------------
# 2  Featurization
# ------------------------------------------------------------
box(X[1], BY, W, BH_, "Featurization")
txt(X[1] + 0.10, CTOP, "Weighted\nhistogram and\nquantile\naggregation")
axb =  fig.add_axes(rect(X[1] + 0.22, 1.98, W - 0.40, 0.62))
axb.bar(np.arange(N_BINS), fp[DIM], width=0.72, color=ACCENT)
axb.set_ylim(0, max(fp[DIM]) * 1.15); clean(axb)
axh =  fig.add_axes(rect(X[1] + 0.22, 1.18, W - 0.40, 0.48))
axh.imshow(fp.T, cmap="viridis", aspect="auto", interpolation="nearest", origin="lower",
           rasterized=True)
clean(axh)
for s in axh.spines.values():
    s.set_visible(True); s.set_color("#B0B0B0"); s.set_linewidth(0.6)
arrow(X[1] + W / 2, 1.94, X[1] + W / 2, 1.70, color=MUTED, lw=1.1, ms=11)
txt(X[1] + 0.10, 1.02, "Gradient\nboosting on\nstandardized\nfeatures")

# ------------------------------------------------------------
# 3  Scale transfer
# ------------------------------------------------------------
box(X[2], BY, W, BH_, "Scale transfer")
txt(X[2] + 0.10, CTOP, "Same model at\nevery scale:\nadmin-2,\nadmin-4,\nplot level")
mw, mh, my =  0.60, 1.05, 0.95
mx =  [X[2] + 0.06, X[2] + W - 0.06 - mw]
ax1 =  fig.add_axes(rect(mx[0], my, mw, mh))
gto.plot(ax=ax1, facecolor="#D8D8D8", edgecolor="white", linewidth=0.3, rasterized=True)
pen_g.plot(ax=ax1, facecolor=ACCENT2, edgecolor="white", linewidth=0.3, rasterized=True)
ax1.set_aspect("equal"); clean(ax1)
ax2 =  fig.add_axes(rect(mx[1], my, mw, mh))
adcs.plot(ax=ax2, column="pred", cmap=yld_cmap, norm=pnorm, edgecolor="white",
          linewidth=0.12, missing_kwds={"color": "#E8E8E8"}, rasterized=True)
ax2.set_aspect("equal"); clean(ax2)
arrow(mx[0] + mw + 0.02, my + mh / 2, mx[1] - 0.02, my + mh / 2, color=MUTED, lw=1.1, ms=11)

# ------------------------------------------------------------
# 4  Ex-post corrections
# ------------------------------------------------------------
box(X[3], BY, W, BH_, "Corrections")
txt(X[3] + 0.10, CTOP, r"Match municipal" "\n" r"aggregate ($e_m$)")
txt(X[3] + 0.10, 1.76, "Shrink\ndeviations by\n" r"$\lambda = 0.72$")

gax =  fig.add_axes(rect(X[3] + 0.16, 2.06, W - 0.32, 0.74)); clean(gax)
gax.set_xlim(0, 1); gax.set_ylim(0, 1)
gd  =  np.array([0.07, -0.10, 0.12, -0.05, 0.09, -0.13])
gx1 =  np.linspace(0.10, 0.36, 6); gx2 = gx1 + 0.50
gax.plot([0.02, 0.98], [0.74, 0.74], color=ACCENT2, lw=1.2, zorder=2)
gax.plot([0.04, 0.44], [0.28, 0.28], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
gax.scatter(gx1, 0.28 + gd, s=11, color=MUTED, zorder=3)
gax.scatter(gx2, 0.74 + gd, s=11, color=ACCENT, zorder=3)
gax.annotate("", xy=(0.49, 0.70), xytext=(0.49, 0.32),
             arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0, mutation_scale=9))
gax.text(0.46, 0.51, r"$e_m$", ha="right", va="center", fontsize=FS, color=GRAY)

gax2 =  fig.add_axes(rect(X[3] + 0.16, 0.24, W - 0.32, 0.70)); clean(gax2)
gax2.set_xlim(0, 1); gax2.set_ylim(0, 1)
gd2 =  np.array([0.34, -0.26, 0.16, -0.36, 0.28])
gx1 =  np.linspace(0.06, 0.28, 5); gx2 = gx1 + 0.42
gax2.plot([0.02, 0.98], [0.50, 0.50], color=ACCENT2, lw=1.2, zorder=2)
gax2.scatter(gx1, 0.50 + gd2, s=11, color=MUTED, zorder=3)
gax2.scatter(gx2, 0.50 + LAM_DEPLOY * gd2, s=11, color=ACCENT, zorder=3)
for y0_, y1_ in [(0.86, 0.62), (0.14, 0.38)]:
    gax2.annotate("", xy=(0.36, y1_), xytext=(0.36, y0_),
                  arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.9, mutation_scale=8))
gax2.text(0.36, 1.02, r"$\times\,\lambda$", ha="center", va="top", fontsize=FS, color=GRAY)

# ------------------------------------------------------------
# 5  Validation (three panels of the combined-season accuracy figure)
# ------------------------------------------------------------
box(X[4], BY, WR, BH_, "Validation")
PW, PG, PX0, PY =  0.70, 0.36, X[4] + 0.45, 1.52
for k, (c, lab, ov, wt) in enumerate(panels):
    px =  PX0 + k * (PW + PG)
    ax =  fig.add_axes(rect(px, PY, PW, PW))
    y, yh =  ev["yield"].values, ev[c].values
    m =  np.isfinite(y) & np.isfinite(yh)
    ax.hexbin(y[m], yh[m], gridsize=30, bins="log", cmap="viridis",
              extent=(0, 12, 0, 12), linewidths=0, rasterized=True)
    ax.plot([0, 12], [0, 12], "--", color="#B0B0B0", lw=0.9)
    ax.set_xlim(0, 12); ax.set_ylim(0, 12)
    ax.set_xticks([0, 6, 12]); ax.set_yticks([0, 6, 12])
    ax.tick_params(labelsize=FS, length=0, pad=2)
    if k:
        ax.set_yticklabels([])
    for s in ax.spines.values():
        s.set_color("#909090"); s.set_linewidth(0.6)
    txt(px + PW / 2, PY + PW + 0.94,
        f"{lab}\n$R^2$ = {fmt(ov)}\nwithin {fmt(wt)}", ha="center", color=GRAY)
txt(X[4] + 0.10, PY + PW / 2, "predicted (t/ha)", rotation=90, ha="center", va="center",
    color=MUTED)
txt(PX0 + (3 * PW + 2 * PG) / 2, PY - 0.44, "reported yield (t/ha)", ha="center", color=MUTED)
txt(X[4] + 0.18, PY - 0.80, "Agricultural census microdata\n(admin-4) and plot-level data",
    color=GRAY)
# role labels above the panel titles: the headline model vs the two reference rows
LY =  PY + PW + 0.94 + 0.10
txt(PX0 + PW / 2, LY, "Headline", ha="center", va="bottom", color=GRAY, fontweight="bold")
txt(PX0 + 1.5 * (PW + PG) + PW / 2, LY, "Reference", ha="center", va="bottom", color=GRAY, fontweight="bold")

# ------------------------------------------------------------
os.makedirs(plot_dir, exist_ok=True)
fig.savefig(out_pdf)          # no bbox_inches: the canvas size is the deliverable
print("Wrote", out_pdf)
subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", out_pdf, out_png[:-4]], check=True)
print("Wrote", out_png)
