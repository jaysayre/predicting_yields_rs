"""
2_plot_maize_monthly_harvesting.py — national maize planting/harvesting seasonality.

Reproduces figures/maize_monthly_harvesting.png, the 2023-03-07 figure that lives
only in the Overleaf tree: the plotting code in 1_CleanSIAPMonthlydata.ipynb is
entirely commented out, and the variants it would write are per-cycle and
per-state (plots/monthly_prod/), never this national all-cycle version. That
mismatch is what made `make copy_to_overleaf` abort, since the target expects the
file in plots/.

Logic is lifted from the commented block in cell 11 of that notebook, summed over
crop cycles rather than split by cycle (the published title carries no cycle
qualifier) over 2018-2024, the window the paper text cites.

Input:  Data/SIAP_monthly/Output/mnthly_siap.dta   (cleaned monthly SIAP)
Output: plots/maize_monthly_harvesting.pdf (pgf/Times, the paper's font) + .png (pdftoppm)

Run: ~/miniforge3/envs/geo_env/bin/python 2_plot_maize_monthly_harvesting.py
     [--years 2018-2024] [--out <path>]
"""
import os, sys, warnings, subprocess
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use("pgf")  # typeset via LaTeX/pgf -> Times, matching the paper body
base_font_size = 12  # match \documentclass[12pt]{article}
from cycler import cycler
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
import matplotlib.pyplot as plt

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
base_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
output_dir =  os.path.join(base_dir, "Data", "SIAP_monthly", "Output")
plot_dir   =  os.path.join(base_dir, "plots")

# ── Inputs ───────────────────────────────────────────────
monthly_siap_dta =  os.path.join(output_dir, "mnthly_siap.dta")   # cleaned monthly SIAP panel

# ── Outputs ──────────────────────────────────────────────
out_path =  os.path.join(plot_dir, "maize_monthly_harvesting.pdf")   # + .png alongside

CROP   =  "Maíz grano"
MONTHS =  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

argv =  sys.argv
y0, y1 =  2018, 2024
if "--years" in argv:
    y0, y1 =  (int(x) for x in argv[argv.index("--years") + 1].split("-"))
if "--out" in argv:
    out_path =  argv[argv.index("--out") + 1]


def main():
    df =  pd.read_stata(monthly_siap_dta)
    # Drop January 2018 — the same exclusion the rest of 1_CleanSIAPMonthlydata.ipynb
    # applies before computing max_harv_mnth*: "Remove January 2018, which is the first
    # month in the dataset and thus meaningless". SIAP's `Sup. Sem.` is cycle-to-date
    # cumulative, and the cleaning differences it with an ungrouped .shift(1); Jan-2018
    # has no prior period, and the final .fillna() puts the raw cumulative back in. Left
    # in, it books a full cycle's national planting (6.37M ha of Spring-Summer alone)
    # into one month and ~10x's the January bars.
    m  =  df[(df["Crop"] == CROP) & (df["year"] >= y0) & (df["year"] <= y1)]
    n0 =  len(m)
    m  =  m[~((m["year"] == 2018) & (m["Mes"] == 1))]
    print(f"[0] dropped {n0 - len(m):,} January-2018 rows (cumulative-not-incremental artifact)")
    print(f"[1] {len(m):,} maize month-municipality-cycle rows, {y0}-{y1}")

    # mean across years within municipality x cycle x month, then sum over the country
    g =  m.groupby(["Mes", "muncode", "Cycle"])[["ha_planted", "ha_harv"]].mean().reset_index()
    g =  g.groupby("Mes")[["ha_planted", "ha_harv"]].sum().reset_index().sort_values("Mes")
    g[["ha_planted", "ha_harv"]] =  g[["ha_planted", "ha_harv"]] / 1_000_000
    print(f"    planted {g['ha_planted'].sum():.2f}M ha/yr, harvested {g['ha_harv'].sum():.2f}M ha/yr")

    # 6.5 in wide = the 12pt article text width, so the 12pt figure fonts print
    # at the body-text size after \includegraphics[width=0.95\textwidth]
    fig, ax =  plt.subplots(1, 1, figsize=(6.5, 3.4))
    ax.bar(range(12), g["ha_planted"], width=0.25, label="Hectares planted",   color="#70ad47")
    ax.bar([y + 0.25 for y in range(12)], g["ha_harv"], width=0.25,
           label="Hectares harvested", color="#ffc000")
    ax.set_title(f"Average maize planting and harvesting by month in Mexico, {y0}--{y1}")
    ax.set_xticks(range(12), MONTHS)
    ax.set_ylabel("Millions of hectares")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left", frameon=False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    stem =  os.path.splitext(out_path)[0]
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", out_path, stem], check=True)
    plt.close(fig)
    print(f"[2] wrote {out_path} and {stem}.png")


if __name__ == "__main__":
    main()
