"""
Dumbbell / slope figure: the model ranking inverts between the municipality level
and the farm (ADC, within-municipality) level. The Agg-NN is best at the
municipality level (Table \ref{tab:validation_mun}) but worst at the farm level,
illustrating why aggregate metrics mislead for downscaling.

Municipality-level R^2 from Table 1 (held-out municipalities, vs DGSIAP); farm-level
pooled within-municipality R^2 from exante_trust_across_models.csv.

Output: plots/coauthor_extras_paper/fig_ranking_inversion.pdf (pgf/Times)
Run:    ~/miniforge3/envs/geo_env/bin/python 13_fig_ranking_inversion.py
"""
import os
import numpy as np
import pandas as pd
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

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
out  = os.path.join(proj, "plots", "coauthor_extras_paper")

# Municipality-level R^2 (Table 1, standardized: all models trained with the
# held-out municipalities excluded, scored on the common N=3,571 held-out
# mun-years; from 4_validation_mun_level.py, full 2017-2024 masked panel).
# Farm-level within-R^2 loaded from the CSV.
MUN = {"Agg-NN": 0.790, "AEF mean": 0.520, "AEF Hist Ens.": 0.627,
       "AEF Hist": 0.608, "NDVI": 0.534}
cm = pd.read_csv(os.path.join(out, "exante_trust_across_models.csv"))
FARM = dict(zip(cm["model"], cm["pooled_within_r2"]))
# AEF RF in Table 1 == the AEF standard model in the within table
models = ["Agg-NN", "AEF mean", "AEF Hist Ens.", "AEF Hist", "NDVI"]

mun_rank  = {m: r for r, m in enumerate(sorted(models, key=lambda m: -MUN[m]), 1)}
farm_rank = {m: r for r, m in enumerate(sorted(models, key=lambda m: -FARM[m]), 1)}

fig, ax = plt.subplots(figsize=(7.2, 5.0))
x0, x1 = 0.0, 1.0
for m in models:
    y0, y1 = mun_rank[m], farm_rank[m]
    inverted = (m == "Agg-NN")
    col = "#B22222" if inverted else "#7A7A7A"
    lw  = 2.6 if inverted else 1.4
    ax.plot([x0, x1], [y0, y1], "-", color=col, lw=lw, alpha=0.9, zorder=3 if inverted else 2)
    ax.scatter([x0, x1], [y0, y1], s=70, color=col, zorder=4, edgecolor="white", linewidth=0.8)
    label = ("\\textbf{%s}" % m) if False else m
    ax.annotate(f"{m}  ({MUN[m]:.2f})", (x0, y0), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9.5, color=col)
    ax.annotate(f"({FARM[m]:+.2f})  {m}", (x1, y1), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9.5, color=col)

ax.set_xlim(-0.78, 1.78)
ax.set_ylim(len(models) + 0.6, 0.4)  # rank 1 at top
ax.set_yticks(range(1, len(models) + 1)); ax.set_yticklabels([f"{r}" for r in range(1, len(models) + 1)])
ax.set_ylabel("rank (1 = best)")
ax.set_xticks([x0, x1])
ax.set_xticklabels(["Municipality level\n($R^2$ vs DGSIAP)",
                    "Farm level\n(within-municipality $R^2$)"], fontsize=10.5)
ax.set_title("The model ranking inverts from the municipality to the farm level\n"
             "(the Agg-NN is best at the aggregate, worst at the disaggregate)",
             fontsize=11.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
p = os.path.join(out, "fig_ranking_inversion.pdf")
fig.savefig(p, bbox_inches="tight")
print("Wrote", p)
print("mun ranks:", mun_rank)
print("farm ranks:", farm_rank)
