"""
Ex-ante trust index --- UNSUPERVISED construction (no ground truth).

The trust index is a fixed, theory-weighted composite of ex-ante features:
each municipality is scored by the equal-weight average of the SIGNED z-scores
of four representativeness drivers, with signs fixed a priori by theory (more
agro-ecological heterogeneity that the satellite can see, and enough sub-areas
to resolve it, => higher trust):

    trust_index = mean_k  sign_k * z(feature_k),  k in
      { AEF within-mun embedding spread        (+),
        AEF effective dimension                (+),
        within-mun irrigation heterogeneity    (+),
        log number of constituent ADCs         (+) }

NO farm-level ground truth enters the index: every feature is computable from
satellite embeddings and public aggregate survey data alone. Ground truth is
used ONLY to VALIDATE the index --- to check that it orders realized
within-municipality skill --- never to construct or weight it.

Outputs (under plots/coauthor_extras_paper/):
  exante_trust_index.csv          (adds trust_index; refreshes within_r2_mun
                                   from the deployed Shrink predictions)

Run:  ~/miniforge3/envs/geo_env/bin/python 10_exante_trust_composite.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P    = os.path.join(proj, "Data", "predictions")
out  = os.path.join(proj, "plots", "coauthor_extras_paper")

# ── theory-fixed composite (signs a priori, equal weight, NO fitting) ──
SIGNS = {"aef_spread": +1, "aef_eff_dim": +1, "irrig_share_sd": +1, "log_n_adc": +1}

def build_index(D):
    cols = list(SIGNS)
    Z = (D[cols] - D[cols].mean()) / D[cols].std()
    return (Z * pd.Series(SIGNS)).sum(axis=1) / len(cols)

# ── load ex-ante features (these are all label-free) ──
D = pd.read_csv(os.path.join(out, "exante_trust_index.csv"), dtype={"muncode": str})
D["muncode"] = D["muncode"].str.zfill(5)
D["trust_index"] = build_index(D)
D.to_csv(os.path.join(out, "exante_trust_index.csv"), index=False)
print(f"Composite features (signed, equal weight): {SIGNS}")

# ── VALIDATION ONLY: does the unsupervised index order realized skill? ──
# Realized skill is that of the DEPLOYED specification: AEF Hist Ens. Shrink
# (lambda = 0.72, Sec 3.6), matching Figure exante_targeting (2026-08-28).
ev = pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))
ev = ev[["adc", "muncode", "yield", "pred"]].dropna(subset=["yield", "pred"]).copy()
ev["muncode"] = ev["muncode"].astype(str).str.zfill(5)
LAM = 0.72
_g = ev.groupby("muncode")["pred"]
ev["pred"] = _g.transform("mean") + LAM * (ev["pred"] - _g.transform("mean"))

# per-mun within-R2 of the deployed (shrunk) predictions, >=5 maize ADCs;
# refreshes the CSV's raw-based within_r2_mun column
_w = []
for m, g in ev.groupby("muncode"):
    if len(g) < 5: continue
    a = g["yield"] - g["yield"].mean(); b = g["pred"] - g["pred"].mean()
    if (a**2).sum() > 0: _w.append((m, 1 - ((a - b)**2).sum() / (a**2).sum()))
_w = pd.DataFrame(_w, columns=["muncode", "within_r2_mun"])
D = D.drop(columns=["within_r2_mun"]).merge(_w, on="muncode", how="left")
D.to_csv(os.path.join(out, "exante_trust_index.csv"), index=False)

# realized per-mun rank skill (Spearman rho), >=5 maize ADCs
rho = []
for m, g in ev.groupby("muncode"):
    if len(g) >= 5 and g["pred"].std() > 0 and g["yield"].std() > 0:
        rho.append((m, spearmanr(g["yield"], g["pred"]).correlation))
rho = pd.DataFrame(rho, columns=["muncode", "rho_mun"])

V = D[["muncode", "trust_index", "within_r2_mun"]].merge(rho, on="muncode", how="left")
sub = V.dropna(subset=["trust_index", "within_r2_mun"])
y = sub["within_r2_mun"].clip(-2, 1)
pear = np.corrcoef(sub["trust_index"], y)[0, 1]
spear = spearmanr(sub["trust_index"], y).correlation
auc = roc_auc_score((sub["within_r2_mun"] > 0).astype(int), sub["trust_index"])
sr = V.dropna(subset=["trust_index", "rho_mun"])
rho_corr = np.corrcoef(sr["trust_index"], sr["rho_mun"])[0, 1]
print("\n── Validation of the unsupervised index vs realized within-mun skill ──")
print(f"  n municipalities         : {len(sub):,}")
print(f"  Pearson  (within-R2)     : {pear:+.3f}")
print(f"  Spearman (within-R2)     : {spear:+.3f}")
print(f"  corr. with rank skill rho: {rho_corr:+.3f}  (n={len(sr):,})")
print(f"  sign-AUC (within-R2 > 0) : {auc:.3f}")

# bivariate driver correlations (Spearman) vs realized within-mun skill
print("\n── Bivariate drivers (Spearman rho vs realized within-mun R2) ──")
for feat in ["log_n_adc", "aef_spread", "irrig_share_sd", "aef_eff_dim"]:
    dd = D[[feat, "within_r2_mun"]].dropna()
    print(f"  {feat:16s} rho = {spearmanr(dd[feat], dd['within_r2_mun']).correlation:+.3f}")

# Cross-model validation lives in 11_exante_trust_across_models.py, the sole
# producer of exante_trust_across_models.csv (RF-based; the numbers the paper quotes).
