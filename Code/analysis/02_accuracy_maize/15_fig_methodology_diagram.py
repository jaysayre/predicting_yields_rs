"""
Figure 2 methodology diagram (two rows).

Step 15 of the accuracy chain; prerequisite chips fetched by
14_fetch_mun_chips_fig.py. Supersedes dep/fig_methodology_diagram.py (v1).

Top row (inputs / the problem):
  A  Maize yields across Mexico (SIAP mun choropleth), two example muns outlined
  B  The two example municipalities at true relative scale, filled by yield
  C  True-color (Sentinel-2 2022) imagery for each, a 2 km cropland window

Bottom row (the method; the original 4-panel pipeline, re-targeted to the two muns):
  1  Extract  - AEF embedding chip for the SAME window as C, per municipality
  2  Histogram+quantize - per-dim bin shares, low- vs high-yield municipality
  3  2D representation - high- minus low-yield mun mass across 64x8 bins,
                          dims in natural label order (not sorted by contrast)
  4  Relate to yields - LOYO scatter, arrows to the two municipalities' points

Two example municipalities (problem_statement.png; SIAP maize Spring-Summer 2022):
  20517  Santo Domingo Tepuxtepec, OAX   1.30 t/ha   114 km2    (low)
  03001  Comondu, BCS                    6.87 t/ha  18156 km2    (high)

Assets from 14_fetch_mun_chips_fig.py (rgb_/aef_chip_mun*_2022.npy + meta json).

Output: plots/fig_methodology_diagram.pdf (+ .png)   [replaces the v1 figure]
Run:    ~/miniforge3/envs/geo_env/bin/python 15_fig_methodology_diagram.py
"""
import os
import json
import subprocess

os.environ.setdefault("PROJ_LIB", os.path.join(
    os.path.expanduser("~"), "miniforge3", "envs", "geo_env", "share", "proj"))

import numpy as np
import pandas as pd
import geopandas as gpd

from   matplotlib.patches import FancyArrowPatch, Rectangle, ConnectionPatch
from   matplotlib.colors import LinearSegmentedColormap, Normalize
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
""",
})

# -- Directories --------------------------------------------
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir   =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir  =  os.path.join(proj_dir, "Data", "predictions")
plot_dir  =  os.path.join(proj_dir, "plots")
shp_path  =  os.path.join(proj_dir, "Data", "muncodes", "shp", "MUNICIPIOS.shp")
siap_path =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")

hist_path =  os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet")
loyo_path =  os.path.join(pred_dir, "mun_aef_hist_gb_loyo_preds.parquet")
meta_path =  os.path.join(aef_dir, "fig2_mun_chips_meta.json")

out_pdf   =  os.path.join(plot_dir, "fig_methodology_diagram.pdf")
out_png   =  os.path.join(plot_dir, "fig_methodology_diagram.png")

YEAR      =  2022
N_DIMS    =  64
N_BINS    =  8
BIN_MIN, BIN_MAX =  -0.8, 0.8
HI_COL    =  "#0072B2"   # high-yield municipality identity (CVD-safe pair)
LO_COL    =  "#D55E00"   # low-yield municipality identity
GRAY      =  "#4A4A4A"
MUTED     =  "#7A7A7A"
bin_cols  =  [f"A{d:02d}_b{b}" for d in range(N_DIMS) for b in range(N_BINS)]
div_cmap  =  LinearSegmentedColormap.from_list("hilo", [LO_COL, "#FFFFFF", HI_COL])
# sequential yield ramp: pale -> deep green (magnitude, single hue)
yld_cmap  =  LinearSegmentedColormap.from_list(
    "yield", ["#F1F4E8", "#A6C36F", "#4C8C2B", "#1C5A1C"])

LO, HI    =  "20517", "03001"                       # the two example muns

# -- Load meta + the two muns' yields -----------------------
meta =  json.load(open(meta_path))
y_lo =  meta[LO]["yield"]
y_hi =  meta[HI]["yield"]

# -- National maize yields (SIAP SS 2022) -------------------
siap =  pd.read_stata(siap_path)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
siap["yield"]   =  siap["q"] / siap["ha_planted"]
siap =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer")
             & (siap["year"] == YEAR) & siap["yield"].notna() & (siap["yield"] > 0)
             & ~siap["muncode"].str.endswith("000")][["muncode", "yield"]]
yv    =  siap["yield"]
ynorm =  Normalize(vmin=float(yv.quantile(0.02)), vmax=float(yv.quantile(0.98)))

shp  =  gpd.read_file(shp_path)                     # Lambert Conformal (metres)
shp["muncode"] =  shp["CVE_ENT"].astype(str).str.zfill(2) + shp["CVE_MUN"].astype(str).str.zfill(3)
shp["geometry"] =  shp["geometry"].simplify(400)    # speed; 400 m tol
shp  =  shp.merge(siap, on="muncode", how="left")

# -- The two muns' 64x8 fingerprints ------------------------
bh   =  pd.read_parquet(hist_path)
bh   =  bh[bh["year"] == YEAR].set_index("muncode")
fp_lo =  bh.loc[LO, bin_cols].values.astype(float).reshape(N_DIMS, N_BINS)
fp_hi =  bh.loc[HI, bin_cols].values.astype(float).reshape(N_DIMS, N_BINS)

# display dim: largest low/high mun contrast among dims with visible spread
spread =  (fp_hi.max(axis=1) < 0.85) & (fp_lo.max(axis=1) < 0.85)
gaps   =  np.where(spread, np.abs(fp_hi - fp_lo).sum(axis=1), -1)
dim    =  int(gaps.argmax())
print(f"display dim A{dim:02d}")

# -- LOYO predictions ---------------------------------------
loyo =  pd.read_parquet(loyo_path)
loyo =  loyo[loyo["year"] == YEAR].dropna(subset=["yield", "yield_pred"])


def load_rgb(code):
    rgb =  np.load(os.path.join(aef_dir, f"rgb_chip_mun{code}_2022.npy")).astype(float)
    out =  np.zeros_like(rgb)
    for k in range(3):                              # per-channel 2-98 stretch + gamma
        lo, hi =  np.nanpercentile(rgb[..., k], [2, 98])
        out[..., k] =  np.clip((rgb[..., k] - lo) / (hi - lo + 1e-9), 0, 1) ** 0.85
    return np.nan_to_num(out)


def load_aef(code):
    chip =  np.load(os.path.join(aef_dir, f"aef_chip_mun{code}_2022.npy"))[dim]
    lo, hi =  np.nanpercentile(chip, [2, 98])
    return np.clip((chip - lo) / (hi - lo + 1e-9), 0, 1)


# ============================================================
fig =  plt.figure(figsize=(13.8, 9.6))
gs  =  fig.add_gridspec(2, 12, height_ratios=[1.02, 1.0],
                        left=0.045, right=0.985, top=0.90, bottom=0.075,
                        hspace=0.42, wspace=0.55)


def chip_pair(subspec, loader, border=True, dimlabel=None):
    """Two stacked chips (low on top, high below) inside a cell."""
    sg =  subspec.subgridspec(2, 1, hspace=0.22)
    axes =  []
    for i, (code, col, name) in enumerate(
            [(LO, LO_COL, "Sto Domingo Tepuxtepec"), (HI, HI_COL, "Comondu")]):
        ax =  fig.add_subplot(sg[i])
        ax.imshow(loader(code), interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(1.8 if border else 0.6)
        ax.set_title(f"{name}", fontsize=base_font_size - 3, color=col, pad=2)
        axes.append(ax)
    if dimlabel:
        axes[-1].annotate(dimlabel, (0.5, -0.14), xycoords="axes fraction",
                          ha="center", va="top", fontsize=base_font_size - 2, color=MUTED)
    return axes


# --- A: national yields + two muns outlined ----------------
axA =  fig.add_subplot(gs[0, 0:4])
shp.plot(ax=axA, column="yield", cmap=yld_cmap, norm=ynorm,
         edgecolor="none", linewidth=0, missing_kwds={"color": "#ECECEC"},
         rasterized=True)   # ~2.4k polygons -> raster, else pgf blows TeX memory
shp[shp["muncode"] == LO].plot(ax=axA, facecolor="none", edgecolor=LO_COL, linewidth=1.1, rasterized=True)
shp[shp["muncode"] == HI].plot(ax=axA, facecolor="none", edgecolor=HI_COL, linewidth=1.1, rasterized=True)
axA.set_axis_off(); axA.set_aspect("equal")
# leader labels to the two muns
for code, col, lab, off in [(LO, LO_COL, "low: 1.3 t/ha", (0.30, 0.12)),
                            (HI, HI_COL, "high: 6.9 t/ha", (0.62, 0.86))]:
    c =  shp[shp["muncode"] == code].geometry.iloc[0].centroid
    axA.annotate(lab, xy=(c.x, c.y), xycoords="data",
                 xytext=off, textcoords="axes fraction",
                 fontsize=base_font_size - 2, color=col, ha="center",
                 arrowprops=dict(arrowstyle="-", color=col, lw=1.0,
                                 connectionstyle="arc3,rad=0.2"))
# colorbar
caxA =  axA.inset_axes([0.06, 0.02, 0.34, 0.035])
cb =  fig.colorbar(mpl.cm.ScalarMappable(norm=ynorm, cmap=yld_cmap),
                   cax=caxA, orientation="horizontal")
cb.set_label("maize yield (t/ha)", fontsize=base_font_size - 3, color=MUTED)
cb.ax.tick_params(labelsize=base_font_size - 4, length=2)
cb.outline.set_edgecolor("#B0B0B0")

# --- B: two muns at true relative scale --------------------
axB =  fig.add_subplot(gs[0, 4:8])
from shapely.affinity import translate
g_hi =  shp[shp["muncode"] == HI].geometry.iloc[0]
g_lo =  shp[shp["muncode"] == LO].geometry.iloc[0]
hx0, hy0, hx1, hy1 =  g_hi.bounds
lx0, ly0, lx1, ly1 =  g_lo.bounds
# place Comondu centred at origin; small mun to its left, vertically centred, true scale
g_hiT =  translate(g_hi, -(hx0 + hx1) / 2, -(hy0 + hy1) / 2)
gap   =  (hx1 - hx0) * 0.10
# small mun: centre it, then shift left of Comondu with a gap (true scale kept)
g_loT =  translate(g_lo, -(lx0 + lx1) / 2 - ((hx1 - hx0) / 2 + gap + (lx1 - lx0) / 2),
                   -(ly0 + ly1) / 2)
for g, col, y in [(g_hiT, HI_COL, y_hi), (g_loT, LO_COL, y_lo)]:
    gpd.GeoSeries([g]).plot(ax=axB, color=yld_cmap(ynorm(y)), edgecolor=col, linewidth=1.8, rasterized=True)
axB.set_aspect("equal"); axB.set_axis_off()
# labels
_wbox =  dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.2")
axB.annotate("Comondu, BCS\n6.9 t/ha  $\\cdot$  18{,}156 km$^2$", (0.70, 0.05),
             xycoords="axes fraction", ha="center", va="bottom",
             fontsize=base_font_size - 2, color=HI_COL, linespacing=1.3, bbox=_wbox)
loc =  g_loT.centroid
axB.annotate("Sto Domingo\nTepuxtepec, OAX\n1.3 t/ha $\\cdot$ 114 km$^2$",
             xy=(loc.x, loc.y), xycoords="data", xytext=(0.15, 0.70),
             textcoords="axes fraction", ha="center", va="center",
             fontsize=base_font_size - 3, color=LO_COL, linespacing=1.3, bbox=_wbox,
             arrowprops=dict(arrowstyle="-|>", color=LO_COL, lw=1.0,
                             connectionstyle="arc3,rad=-0.25"))
axB.annotate(r"same unit, $160\times$ the area", (0.5, 1.0), xycoords="axes fraction",
             ha="center", va="top", fontsize=base_font_size - 2, color=GRAY, bbox=_wbox)

# --- C: true-color imagery ---------------------------------
axC_list =  chip_pair(gs[0, 8:12], load_rgb, dimlabel="2 km window, 10 m pixels")

# --- 1: AEF embedding chips --------------------------------
ax1_list =  chip_pair(gs[1, 0:3], load_aef, dimlabel=f"dim A{dim:02d} " r"($\times$64)")

# --- 2: per-dim histogram, two muns ------------------------
ax2   =  fig.add_subplot(gs[1, 3:6])
edges =  np.linspace(BIN_MIN, BIN_MAX, N_BINS + 1)
ctr   =  (edges[:-1] + edges[1:]) / 2
bw    =  0.085
for e in edges:
    ax2.axvline(e, color="#D0D0D0", lw=0.6, ls=(0, (1, 3)), zorder=0)
ax2.bar(ctr - bw / 2 - 0.004, fp_hi[dim], width=bw, color=HI_COL, zorder=3)
ax2.bar(ctr + bw / 2 + 0.004, fp_lo[dim], width=bw, color=LO_COL, zorder=3)
ax2.set_xlim(-0.44, 0.44)
ax2.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
ax2.set_xlabel(f"embedding value, dim A{dim:02d}", fontsize=base_font_size)
ax2.set_ylabel("share of pixels", fontsize=base_font_size)
ax2.tick_params(length=0, labelsize=base_font_size - 1)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)
ax2.annotate(r"8 fixed bins $\times$ 64 dims = 512 features", (0.5, -0.30),
             xycoords="axes fraction", ha="center", va="top",
             fontsize=base_font_size - 1, color=MUTED)
for y, col, lab in [(0.93, HI_COL, "Comondu (high)"), (0.83, LO_COL, "Tepuxtepec (low)")]:
    ax2.add_patch(Rectangle((0.045, y), 0.05, 0.055, transform=ax2.transAxes,
                            clip_on=False, facecolor=col, edgecolor="none"))
    ax2.annotate(lab, (0.11, y + 0.028), xycoords="axes fraction", ha="left",
                 va="center", fontsize=base_font_size - 2, color=GRAY)

# --- 3: 64x8 mun contrast, natural dim order ---------------
ax3  =  fig.add_subplot(gs[1, 6:9])
diff =  fp_hi - fp_lo
vmax =  np.abs(diff).max()
im3  =  ax3.imshow(diff.T, cmap=div_cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest", origin="lower")
ax3.set_xticks([0, 15, 31, 47, 63]); ax3.set_xticklabels(["1", "16", "32", "48", "64"])
ax3.set_yticks([0, 7]); ax3.set_yticklabels(["1", "8"])
ax3.tick_params(length=0, labelsize=base_font_size - 2)
for s in ax3.spines.values():
    s.set_color("#B0B0B0")
ax3.set_xlabel("embedding dimension (1-64)", fontsize=base_font_size)
ax3.set_ylabel("bin", fontsize=base_font_size - 1)
cax3 =  ax3.inset_axes([0.3, -0.26, 0.4, 0.06])
cb3  =  fig.colorbar(im3, cax=cax3, orientation="horizontal"); cb3.set_ticks([])
cb3.outline.set_edgecolor("#B0B0B0")
cax3.annotate("more mass, low", (-0.04, 0.5), xycoords="axes fraction",
              ha="right", va="center", fontsize=base_font_size - 3, color=MUTED)
cax3.annotate("more mass, high", (1.04, 0.5), xycoords="axes fraction",
              ha="left", va="center", fontsize=base_font_size - 3, color=MUTED)

# --- 4: scatter + arrows to the two muns -------------------
ax4  =  fig.add_subplot(gs[1, 9:12])
ax4.scatter(loyo["yield"], loyo["yield_pred"], s=6, color="#9A9A9A", alpha=0.30,
            linewidths=0, zorder=2)
lim  =  float(np.nanquantile(loyo[["yield", "yield_pred"]].values, 0.999))
ax4.plot([0, lim], [0, lim], ls="--", lw=1.0, color="#B0B0B0", zorder=1)
ss   =  loyo[["yield", "yield_pred"]].values
r2   =  1 - np.sum((ss[:, 0] - ss[:, 1])**2) / np.sum((ss[:, 0] - ss[:, 0].mean())**2)
ax4.annotate(f"$R^2 = {r2:.2f}$", (0.06, 0.90), xycoords="axes fraction",
             fontsize=base_font_size, color=GRAY)
for code, col, lab, xytext in [
        (LO, LO_COL, "Tepuxtepec", (0.42, 0.16)),
        (HI, HI_COL, "Comondu",    (0.55, 0.92))]:
    r =  loyo[loyo["muncode"] == code]
    if len(r):
        x, yv2 =  float(r["yield"].iloc[0]), float(r["yield_pred"].iloc[0])
        ax4.scatter([x], [yv2], s=46, color=col, edgecolor="white", linewidth=0.8, zorder=5)
        ax4.annotate(lab, xy=(x, yv2), xycoords="data", xytext=xytext,
                     textcoords="axes fraction", fontsize=base_font_size - 2, color=col,
                     ha="center", arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                                  connectionstyle="arc3,rad=0.2"))
ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
ax4.set_xlabel("observed yield (t/ha)", fontsize=base_font_size)
ax4.set_ylabel("predicted yield (t/ha)", fontsize=base_font_size)
ax4.tick_params(length=0, labelsize=base_font_size - 1)
for s in ["top", "right"]:
    ax4.spines[s].set_visible(False)
ax4.annotate(f"gradient boosting, LOYO municipalities, {YEAR}", (0.5, -0.30),
             xycoords="axes fraction", ha="center", va="top",
             fontsize=base_font_size - 1, color=MUTED)

# --- panel titles ------------------------------------------
top_titles =  [(axA, r"\textbf{Maize yields across Mexico}" "\nSIAP municipal yields, 2022"),
               (axB, r"\textbf{Two example municipalities}" "\ntrue relative scale, filled by yield"),
               (axC_list[0], r"\textbf{True-color imagery}" "\nSentinel-2, 2022")]
for ax, t in top_titles:
    pos =  ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, 0.965, t, ha="center", va="top",
             fontsize=base_font_size, color=GRAY, linespacing=1.35)

bot_titles =  [(ax1_list[0], r"\textbf{1 $\cdot$ Extract}" "\nAEF embeddings (GEE),\n64 dims / 10 m cropland pixel"),
               (ax2, r"\textbf{2 $\cdot$ Histogram + quantize}" "\neach dim binned into\n8 fixed-width bins"),
               (ax3, r"\textbf{3 $\cdot$ 2D representation}" "\nhigh- minus low-yield mun mass\nacross all 64 $\\times$ 8 bins"),
               (ax4, r"\textbf{4 $\cdot$ Relate to yields}" "\ntrained on SIAP\nmunicipal yields")]
for ax, t in bot_titles:
    pos =  ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, 0.475, t, ha="center", va="top",
             fontsize=base_font_size, color=GRAY, linespacing=1.35)

# --- imagery -> embedding note (inter-panel connector arrows removed) ---
fig.text(0.30, 0.515, "imagery $\\rightarrow$ embedding\n(same 2 km window)",
         ha="center", va="center", fontsize=base_font_size - 3, color=MUTED, style="italic")

os.makedirs(plot_dir, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
print("Wrote", out_pdf)
subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", out_pdf, out_png[:-4]], check=True)
print("Wrote", out_png)
