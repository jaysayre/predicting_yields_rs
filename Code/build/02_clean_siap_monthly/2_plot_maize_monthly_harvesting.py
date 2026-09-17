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
qualifier) and restricted to 2018-2022 to match it -- the underlying data now
extends to 2024.

Input:  Data/SIAP_monthly/Output/mnthly_siap.dta   (cleaned monthly SIAP)
Output: plots/maize_monthly_harvesting.png

Run: ~/miniforge3/envs/geo_env/bin/python 2_plot_maize_monthly_harvesting.py
     [--years 2018-2022] [--out <path>]
"""
import os, sys, warnings
import pandas as pd
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
base_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
output_dir =  os.path.join(base_dir, "Data", "SIAP_monthly", "Output")
plot_dir   =  os.path.join(base_dir, "plots")

# ── Inputs ───────────────────────────────────────────────
monthly_siap_dta =  os.path.join(output_dir, "mnthly_siap.dta")   # cleaned monthly SIAP panel

# ── Outputs ──────────────────────────────────────────────
out_path =  os.path.join(plot_dir, "maize_monthly_harvesting.png")

CROP   =  "Maíz grano"
MONTHS =  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

argv =  sys.argv
y0, y1 =  2018, 2022
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

    fig, ax =  plt.subplots(1, 1, figsize=(14, 7))
    ax.bar(range(12), g["ha_planted"], width=0.25, label="Hectares planted",   color="#70ad47")
    ax.bar([y + 0.25 for y in range(12)], g["ha_harv"], width=0.25,
           label="Hectares harvested", color="#ffc000")
    ax.set_title(f"Average maize planting and harvesting by month in Mexico, {y0}-{y1}")
    ax.set_xticks(range(12), MONTHS)
    ax.set_ylabel("Millions of hectares")
    ax.set_xlabel("Month")
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", frameon=False, ncol=7)
    plt.xticks(rotation=0)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"[2] wrote {out_path}")


if __name__ == "__main__":
    main()
