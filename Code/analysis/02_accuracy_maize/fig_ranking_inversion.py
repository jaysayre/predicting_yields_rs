"""
Dumbbell / slope figure: the model ranking inverts between the municipality level
and the farm (ADC, within-municipality) level. The Agg-NN is best at the
municipality level (Table \ref{tab:validation_mun}) but worst at the farm level,
illustrating why aggregate metrics mislead for downscaling.

Municipality-level R^2 from Table 1 (held-out municipalities, vs SIAP); farm-level
pooled within-municipality R^2 from exante_trust_across_models.csv.

Output: plots/coauthor_extras_paper/fig_ranking_inversion.png
Run:    ~/miniforge3/envs/geo_env/bin/python fig_ranking_inversion.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
out  = os.path.join(proj, "plots", "coauthor_extras_paper")

# Municipality-level R^2 (Table 1, standardized: all models trained with the
# held-out municipalities excluded, scored on the common N=3,571 held-out
# mun-years; from validation_mun_level.py, full 2017-2024 masked panel).
# Farm-level within-R^2 loaded from the CSV.
MUN = {"Agg-NN": 0.812, "AEF mean": 0.579, "AEF Hist Ens.": 0.627,
       "AEF Hist": 0.611, "NDVI": 0.534}
cm = pd.read_csv(os.path.join(out, "exante_trust_across_models.csv"))
FARM = dict(zip(cm["model"], cm["pooled_within_r2"]))
# AEF RF in Table 1 == the AEF standard model in the within table
models = ["Agg-NN", "AEF mean", "AEF Hist Ens.", "AEF Hist", "NDVI"]

mun_rank  = {m: r for r, m in enumerate(sorted(models, key=lambda m: -MUN[m]), 1)}
farm_rank = {m: r for r, m in enumerate(sorted(models, key=lambda m: -FARM[m]), 1)}

plt.rcParams.update({"font.family": "serif", "axes.edgecolor": "#404040",
                     "xtick.color": "#404040", "ytick.color": "#404040"})
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
ax.set_ylim(6.6, 0.4)               # rank 1 at top
ax.set_yticks(range(1, 7)); ax.set_yticklabels([f"{r}" for r in range(1, 7)])
ax.set_ylabel("rank (1 = best)")
ax.set_xticks([x0, x1])
ax.set_xticklabels(["Municipality level\n($R^2$ vs SIAP, Table 1)",
                    "Farm level\n(within-municipality $R^2$)"], fontsize=10.5)
ax.set_title("The model ranking inverts from the municipality to the farm level\n"
             "(the Agg-NN is best at the aggregate, worst at the disaggregate)",
             fontsize=11.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
p = os.path.join(out, "fig_ranking_inversion.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print("Wrote", p)
print("mun ranks:", mun_rank)
print("farm ranks:", farm_rank)
