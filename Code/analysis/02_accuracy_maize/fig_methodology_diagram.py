"""
Methodology process diagram: GEE AEF extraction -> per-dim histogram/quantization
-> high- vs low-yield contrast in the 64x8 quantized representation -> GB model
relating fingerprints to SIAP yields (LOYO mun predictions).

Panels compare the top-20% vs bottom-20% yield municipalities (2022):
  - Data/alpha_earth/aef_chip_mun26002_2022.npy                 (real AEF chip; run
    fetch_aef_chip_fig.py once to create it — synthetic fallback otherwise)
  - Data/alpha_earth/alpha_earth_mex_mun_binned_hist.parquet    (64 dims x 8 bins)
  - SIAP season yields (maize, Spring-Summer)                    (group definitions)
  - Data/predictions/mun_aef_hist_gb_loyo_preds.parquet          (obs vs pred panel)

Output: plots/fig_methodology_diagram.pdf (+ .png preview via pdftoppm)
Run:    ~/miniforge3/envs/geo_env/bin/python fig_methodology_diagram.py
"""
import os
import subprocess
import numpy as np
import pandas as pd

from   matplotlib.ticker import FuncFormatter
from   matplotlib.patches import FancyArrowPatch, Rectangle
from   matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
import matplotlib.pyplot as plt

base_font_size = 12  # match \documentclass[12pt]{article}

from cycler import cycler
mpl.use("pgf")  # typeset via LaTeX/pgf
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,            # don't let mpl override fonts
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.unicode_minus": False,
    "font.size": base_font_size,            # default text
    "axes.titlesize": base_font_size + 3,   # plot titles
    "axes.labelsize": base_font_size + 2,   # x/y labels
    "xtick.labelsize": base_font_size,      # tick labels
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
    "grid.linestyle": (0, (1, 3)),   # fine dotted
    "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#4A4A4A"]),  # default series color
    "pgf.preamble": r"""
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
""",
})

# ── Directories ──────────────────────────────────────────
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir   =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir  =  os.path.join(proj_dir, "Data", "predictions")
plot_dir  =  os.path.join(proj_dir, "plots")
siap_path =  os.path.join(home_dir, "Dropbox", "Projects",
                          "The Promise of Crop Substitution",
                          "data", "SIAP", "Cleaned",
                          "siap_ag_prod_estimation_by_season.dta")

# ── Inputs ───────────────────────────────────────────────
chip_path =  os.path.join(aef_dir, "aef_chip_mun26002_2022.npy")               # real AEF chip (64,H,W)
hist_path =  os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet")  # mun 64x8 bin shares
loyo_path =  os.path.join(pred_dir, "mun_aef_hist_gb_loyo_preds.parquet")      # LOYO mun obs/pred

# ── Outputs ──────────────────────────────────────────────
out_pdf   =  os.path.join(plot_dir, "fig_methodology_diagram.pdf")
out_png   =  os.path.join(plot_dir, "fig_methodology_diagram.png")

YEAR      =  2022
Q_GRP     =  0.20        # top/bottom yield quantile defining the two groups
N_DIMS    =  64
N_BINS    =  8
BIN_MIN, BIN_MAX =  -0.8, 0.8
HI_COL    =  "#0072B2"   # high-yield accent (CVD-safe pair)
LO_COL    =  "#D55E00"   # low-yield accent
GRAY      =  "#4A4A4A"
MUTED     =  "#7A7A7A"
bin_cols  =  [f"A{d:02d}_b{b}" for d in range(N_DIMS) for b in range(N_BINS)]
div_cmap  =  LinearSegmentedColormap.from_list("hilo", [LO_COL, "#FFFFFF", HI_COL])


# ── Load fingerprints + yields, define high/low-yield groups ──
bh   =  pd.read_parquet(hist_path)
bh   =  bh[bh["year"] == YEAR].copy()

siap =  pd.read_stata(siap_path)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
siap["yield"]   =  siap["q"] / siap["ha_planted"]
siap =  siap[(siap["name"] == "Maize")
             & (siap["growing_season"] == "Spring-Summer")
             & (siap["year"] == YEAR)
             & siap["yield"].notna() & (siap["yield"] > 0)
             & ~siap["muncode"].str.endswith("000")][["muncode", "yield"]]

cand =  bh.merge(siap, on="muncode", how="inner")
cand =  cand[cand[bin_cols].notna().all(axis=1)]            # complete fingerprints only
cand =  cand.sort_values("yield").reset_index(drop=True)
n_g  =  int(Q_GRP * len(cand))
lo_muns  =  set(cand.iloc[:n_g]["muncode"])
hi_muns  =  set(cand.iloc[-n_g:]["muncode"])
fp_lo    =  cand.iloc[:n_g][bin_cols].values.reshape(-1, N_DIMS, N_BINS).mean(axis=0)
fp_hi    =  cand.iloc[-n_g:][bin_cols].values.reshape(-1, N_DIMS, N_BINS).mean(axis=0)
print(f"groups: {n_g} muns each; high-yield mean {cand.iloc[-n_g:]['yield'].mean():.1f} t/ha, "
      f"low-yield mean {cand.iloc[:n_g]['yield'].mean():.1f} t/ha")

# display dim for panels 1-2: largest group gap among dims with visible spread
spread  =  (fp_hi.max(axis=1) < 0.75) & (fp_lo.max(axis=1) < 0.75)
gaps    =  np.where(spread, np.abs(fp_hi - fp_lo).sum(axis=1), -1)
dim     =  int(gaps.argmax())
print(f"display dim A{dim:02d}")

loyo  =  pd.read_parquet(loyo_path)
loyo  =  loyo[loyo["year"] == YEAR].dropna(subset=["yield", "yield_pred"])


# ── Panel 1 field: real AEF chip if fetched, else synthetic ──
if os.path.exists(chip_path):
    chip  =  np.load(chip_path)
    field =  np.clip(chip[dim], BIN_MIN, BIN_MAX)
    side  =  min(field.shape)                               # center-crop to square
    r0    =  (field.shape[0] - side) // 2
    c0    =  (field.shape[1] - side) // 2
    field =  field[r0:r0 + side, c0:c0 + side]
    v1, v2   =  np.percentile(field, [2, 98])               # display stretch
    chip_lab =  f"dim A{dim:02d}, 10 m pixels (Sonora)"
    print(f"panel 1: real AEF chip dim A{dim:02d}, {field.shape}")
else:
    print("panel 1: synthetic fallback (run fetch_aef_chip_fig.py for the real chip)")
    rng   =  np.random.default_rng(3)
    f     =  rng.normal(size=(26, 26))
    k     =  np.ones(3) / 3.0
    for _ in range(3):
        f =  np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, f)
        f =  np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, f)
    field =  np.clip((f - f.mean()) / (f.std() + 1e-9) * 0.35, BIN_MIN, BIN_MAX)
    v1, v2   =  BIN_MIN, BIN_MAX
    chip_lab =  f"dim A{dim:02d} " + r"$\in [-0.8, 0.8]$ shown"


# ── Figure ───────────────────────────────────────────────
fig  =  plt.figure(figsize=(13.6, 5.0))
gs   =  fig.add_gridspec(1, 4, width_ratios=[0.95, 0.95, 1.45, 1.0],
                         left=0.045, right=0.985, top=0.76, bottom=0.17, wspace=0.5)

# — Stage 1: AEF extraction —
ax1  =  fig.add_subplot(gs[0])
for off in [0.10, 0.05]:                                    # layer-stack cue: x64 dims
    ax1.add_patch(Rectangle((off, off), 1, 1, transform=ax1.transAxes, clip_on=False,
                            facecolor="#F2F2F0", edgecolor="#B0B0B0", lw=0.7, zorder=0))
ax1.imshow(field, cmap="Greys", vmin=v1, vmax=v2, interpolation="nearest",
           extent=(0, 1, 0, 1), zorder=2)
ax1.set_xticks([]); ax1.set_yticks([])
for s in ax1.spines.values():
    s.set_color("#B0B0B0")
ax1.annotate(r"$\times$64 dims", (1.11, 1.06), xycoords="axes fraction",
             ha="right", va="bottom", fontsize=base_font_size - 1, color=MUTED)
ax1.annotate(chip_lab, (0.5, -0.09),
             xycoords="axes fraction", ha="center", va="top",
             fontsize=base_font_size - 1, color=MUTED)

# — Stage 2: histogram + quantize (one dim, group averages) —
ax2    =  fig.add_subplot(gs[1])
edges  =  np.linspace(BIN_MIN, BIN_MAX, N_BINS + 1)
ctr    =  (edges[:-1] + edges[1:]) / 2
bw     =  0.085
for e in edges:
    ax2.axvline(e, color="#D0D0D0", lw=0.6, ls=(0, (1, 3)), zorder=0)
ax2.bar(ctr - bw / 2 - 0.004, fp_hi[dim], width=bw, color=HI_COL, zorder=3)
ax2.bar(ctr + bw / 2 + 0.004, fp_lo[dim], width=bw, color=LO_COL, zorder=3)
# zoom to where cropland mass lives ([-0.4, 0.4] holds >99% of it); the outer
# fixed bins are near-empty, so showing the full [-0.8, 0.8] range looks sparse.
ax2.set_xlim(-0.44, 0.44)
ax2.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
ax2.set_xlabel(f"embedding value, dim A{dim:02d}", fontsize=base_font_size)
ax2.set_ylabel("share of pixels", fontsize=base_font_size)
ax2.tick_params(length=0, labelsize=base_font_size - 1)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)
ax2.annotate(r"8 fixed bins $\times$ 64 dims = 512 features", (0.5, -0.28),
             xycoords="axes fraction", ha="center", va="top",
             fontsize=base_font_size - 1, color=MUTED)
for y, col, lab in [(0.93, HI_COL, r"high-yield muns (top 20\%)"),
                    (0.83, LO_COL, r"low-yield muns (bottom 20\%)")]:
    ax2.add_patch(Rectangle((0.035, y), 0.045, 0.055, transform=ax2.transAxes,
                            clip_on=False, facecolor=col, edgecolor="none"))
    ax2.annotate(lab, (0.10, y + 0.028), xycoords="axes fraction", ha="left",
                 va="center", fontsize=base_font_size - 2, color=GRAY)

# — Stage 3: high vs low contrast across the full 64x8 representation —
ax3   =  fig.add_subplot(gs[2])
diff  =  fp_hi - fp_lo
order =  np.argsort(-np.abs(diff).sum(axis=1))              # sort dims by contrast
vmax  =  np.abs(diff).max()
im3   =  ax3.imshow(diff[order].T, cmap=div_cmap, vmin=-vmax, vmax=vmax,
                    aspect="auto", interpolation="nearest", origin="lower")
ax3.set_xticks([0, 16, 32, 48, 63]); ax3.set_xticklabels(["1", "16", "32", "48", "64"])
ax3.set_yticks([0, 7]); ax3.set_yticklabels(["1", "8"])
ax3.tick_params(length=0, labelsize=base_font_size - 2)
for s in ax3.spines.values():
    s.set_color("#B0B0B0")
ax3.set_xlabel("embedding dimension (sorted by contrast)", fontsize=base_font_size)
ax3.set_ylabel("bin", fontsize=base_font_size - 1)
cax  =  ax3.inset_axes([0.3, -0.40, 0.4, 0.06])
cb   =  fig.colorbar(im3, cax=cax, orientation="horizontal")
cb.set_ticks([])
cb.outline.set_edgecolor("#B0B0B0")
cax.annotate("more mass, low-yield", (-0.04, 0.5), xycoords="axes fraction",
             ha="right", va="center", fontsize=base_font_size - 2, color=MUTED)
cax.annotate("more mass, high-yield", (1.04, 0.5), xycoords="axes fraction",
             ha="left", va="center", fontsize=base_font_size - 2, color=MUTED)

# — Stage 4: relate to yields —
ax4   =  fig.add_subplot(gs[3])
grp   =  np.where(loyo["muncode"].isin(hi_muns), "hi",
          np.where(loyo["muncode"].isin(lo_muns), "lo", "mid"))
for g, col, al, zo in [("mid", "#9A9A9A", 0.30, 2), ("lo", LO_COL, 0.5, 3),
                       ("hi", HI_COL, 0.5, 3)]:
    sub =  loyo[grp == g]
    ax4.scatter(sub["yield"], sub["yield_pred"], s=6, color=col, alpha=al,
                linewidths=0, zorder=zo)
lim   =  float(np.nanquantile(loyo[["yield", "yield_pred"]].values, 0.999))
ax4.plot([0, lim], [0, lim], ls="--", lw=1.0, color="#B0B0B0", zorder=1)
ss    =  loyo[["yield", "yield_pred"]].values
r2    =  1 - np.sum((ss[:, 0] - ss[:, 1])**2) / np.sum((ss[:, 0] - ss[:, 0].mean())**2)
ax4.annotate(f"$R^2 = {r2:.2f}$", (0.06, 0.90), xycoords="axes fraction",
             fontsize=base_font_size, color=GRAY)
ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
ax4.set_xlabel("observed yield (t/ha)", fontsize=base_font_size)
ax4.set_ylabel("predicted yield (t/ha)", fontsize=base_font_size)
ax4.tick_params(length=0, labelsize=base_font_size - 1)
for s in ["top", "right"]:
    ax4.spines[s].set_visible(False)
ax4.annotate(f"gradient boosting, LOYO municipalities, {YEAR}", (0.5, -0.28),
             xycoords="axes fraction", ha="center", va="top",
             fontsize=base_font_size - 1, color=MUTED)

# — Stage titles + connecting arrows —
titles =  [r"\textbf{1 $\cdot$ Extract}" "\nAEF embeddings (GEE),\n64 dims per 10 m cropland pixel",
           r"\textbf{2 $\cdot$ Histogram + quantize}" "\neach dim binned into\n8 fixed-width bins",
           r"\textbf{3 $\cdot$ 2D representation}" "\nhigh- minus low-yield mass\nacross all 64 $\\times$ 8 bin shares",
           r"\textbf{4 $\cdot$ Relate to yields}" "\ntrained on SIAP\nmunicipal yields"]
for ax, t in zip([ax1, ax2, ax3, ax4], titles):
    pos =  ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, 0.985, t, ha="center", va="top",
             fontsize=base_font_size, color=GRAY, linespacing=1.35)
for a, b in [(ax1, ax2), (ax2, ax3), (ax3, ax4)]:
    p0, p1 =  a.get_position(), b.get_position()
    y      =  (p0.y0 + p0.y1) / 2
    fig.add_artist(FancyArrowPatch((p0.x1 + 0.016, y), (p1.x0 - 0.042, y),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=16, lw=1.4, color=GRAY))

os.makedirs(plot_dir, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
print("Wrote", out_pdf)

# PNG preview for slides / quick viewing
subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile",
                out_pdf, out_png[:-4]], check=True)
print("Wrote", out_png)
