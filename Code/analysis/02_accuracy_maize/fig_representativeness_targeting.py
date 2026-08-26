"""
Ex-ante targeting figure (Goal 3): where does downscaling resolve WITHIN-area
yield variation, and can we tell IN ADVANCE without ground truth?

Theory: the method's within-area resolution is governed by REPRESENTATIVENESS --
predictability rises with the share of within-area yield variance that is
agro-ecological (terrain, water, soil, crop fraction; visible to AEF) rather
than management-driven (fertilizer, seed, timing; invisible to AEF). Every
ex-ante-computable proxy for agro-ecological heterogeneity (AEF within-mun
spread, irrigation gradient, intermediate maize share, enough ADCs to resolve)
should therefore predict realized within-mun skill.

This figure shows realized within-municipality skill rising monotonically across:
  (a) quintiles of the ex-ante TRUST INDEX (an unsupervised theory-weighted
      composite of representativeness features, built with NO ground truth), and
  (b) quintiles of AEF within-municipality SPREAD (the single most interpretable
      ex-ante driver of agro-ecological heterogeneity).

Inputs (under ~/Dropbox/Projects/Maize_prediction/):
  Data/predictions/adc_aef_hist_ens_eval.parquet           -- ADC yield + pred
  plots/coauthor_extras_paper/exante_trust_index.csv       -- per-mun ex-ante features
Output:
  plots/coauthor_extras_paper/fig_representativeness_targeting.png

Run:  ~/miniforge3/envs/geo_env/bin/python fig_representativeness_targeting.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
pred_dir  =  os.path.join(proj_dir, "Data", "predictions")
plot_dir  =  os.path.join(proj_dir, "plots", "coauthor_extras_paper")

ev    =  pd.read_parquet(os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet"))
trust =  pd.read_csv(os.path.join(plot_dir, "exante_trust_index.csv"),
                     dtype={'muncode': str})

d = ev.dropna(subset=['yield', 'pred']).merge(
    trust[['muncode', 'trust_index', 'aef_spread']], on='muncode', how='inner')
# keep muns with >=5 ADCs (stable within-mun stats)
cnt = d.groupby('muncode')['yield'].transform('size')
d = d[cnt >= 5].copy()


def pooled_within_r2(df):
    a = (df['yield'] - df.groupby('muncode')['yield'].transform('mean')).values
    b = (df['pred']  - df.groupby('muncode')['pred'].transform('mean')).values
    return 1 - np.sum((a - b)**2) / np.sum(a*a)

def median_within_spearman(df):
    rs = []
    for _, g in df.groupby('muncode'):
        if len(g) >= 5 and g['pred'].std() > 0 and g['yield'].std() > 0:
            rs.append(spearmanr(g['yield'], g['pred']).correlation)
    return np.nanmedian(rs)


def gradient(df, score_col, q=5):
    """Pooled within-R2 and median within-Spearman per quintile of score_col."""
    muns = df.groupby('muncode')[score_col].first().dropna()
    edges = muns.quantile(np.linspace(0, 1, q + 1)).values.copy()
    edges[0] -= 1e-9; edges[-1] += 1e-9
    lab = pd.cut(muns, bins=np.unique(edges), labels=False)
    qmap = dict(zip(muns.index, lab))
    df = df.assign(_q=df['muncode'].map(qmap)).dropna(subset=['_q'])
    out = []
    for qi in sorted(df['_q'].unique()):
        sub = df[df['_q'] == qi]
        out.append(dict(q=int(qi),
                        wr2=pooled_within_r2(sub),
                        rho=median_within_spearman(sub),
                        score=muns[lab == qi].median(),
                        n_mun=sub['muncode'].nunique()))
    return pd.DataFrame(out)

g_trust = gradient(d, 'trust_index')
g_spread = gradient(d, 'aef_spread')
print("Trust-index gradient:\n", g_trust.to_string(index=False))
print("\nAEF-spread gradient:\n", g_spread.to_string(index=False))

# ── Plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
GREY = "#4A4A4A"; RED = "#B22222"

for ax, gdf, xlab, title in [
    (axes[0], g_trust,  "ex-ante trust-index quintile (low $\\to$ high)",
     "(a) Realized within-mun skill rises with the\nex-ante trust index (no ground truth used)"),
    (axes[1], g_spread, "AEF within-mun spread quintile (low $\\to$ high)",
     "(b) ...and with AEF within-mun spread\n(agro-ecological heterogeneity proxy)"),
]:
    x = gdf['q'].values + 1
    ax.bar(x - 0.0, gdf['wr2'], width=0.55, color=GREY, label="pooled within-$R^2$")
    ax.plot(x, gdf['rho'], '-o', color=RED, lw=1.8, label="median within-mun Spearman $\\rho$")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel(xlab); ax.set_xticks(x)
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, axis='y', ls=(0, (1, 3)), lw=.6, color="#D0D0D0"); ax.set_axisbelow(True)
axes[0].set_ylabel("within-municipality skill")
axes[0].legend(fontsize=9, loc='upper left', framealpha=0.9)
fig.suptitle("Ex-ante targeting: the downscaling method resolves within-area variation "
             "where heterogeneity is agro-ecological", fontsize=12, y=1.02)
fig.tight_layout()
out = os.path.join(plot_dir, "fig_representativeness_targeting.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nWrote {out}")
