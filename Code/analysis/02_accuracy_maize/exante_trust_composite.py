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
  exante_trust_index.csv          (adds the unsupervised `trust_index` column)
  exante_trust_across_models.csv  (model pooled within-R2 + composite corr.)

Run:  ~/miniforge3/envs/geo_env/bin/python exante_trust_composite.py
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
ev = pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))
ev = ev[["adc", "muncode", "yield", "pred"]].dropna(subset=["yield", "pred"]).copy()
ev["muncode"] = ev["muncode"].astype(str).str.zfill(5)

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

# ── cross-model: same single index vs EACH model's realized within-skill ──
# 2026-08-26: single cropland-masked NDVI baseline (aefn2) replaces the two
# unmasked h3 variants, matching the main accuracy tables.
MODELS = [("NDVI (masked)", "adc_aefn2_masked_preds.parquet", "pred"),
          ("AEF mean", "adc_alpha_earth_preds.csv", "yield_pred"),
          ("Agg-NN", "adc_mlp_yield_preds.csv", "pred_yield"),
          ("AEF Hist", "adc_aef_hist_gb_preds.parquet", "yield_pred"),
          ("AEF Hist Ens.", None, None)]
base = ev.rename(columns={"pred": "AEF Hist Ens."})
for nm, f, c in MODELS:
    if f is None: continue
    d = pd.read_parquet(os.path.join(P, f)) if f.endswith("parquet") else pd.read_csv(os.path.join(P, f))
    if "year" in d.columns: d = d[d["year"] == 2022]
    k = "adc" if "adc" in d.columns else "adcid"
    d["adc"] = d[k].astype(str).str.replace("-", "", regex=False)
    base = base.merge(d[["adc", c]].rename(columns={c: nm}).dropna().drop_duplicates("adc"), on="adc", how="left")

def pooled_within(p):
    s = base[["muncode", "yield", p]].dropna()
    cnt = s.groupby("muncode")["yield"].transform("size"); s = s[cnt >= 2]
    a = (s["yield"] - s.groupby("muncode")["yield"].transform("mean")).values
    b = (s[p] - s.groupby("muncode")[p].transform("mean")).values
    return 1 - np.sum((a - b)**2) / np.sum(a * a)

def per_mun_within(p):
    rows = []
    for m, g in base.dropna(subset=["yield", p]).groupby("muncode"):
        if len(g) < 5: continue
        a = g["yield"] - g["yield"].mean(); b = g[p] - g[p].mean()
        if (a**2).sum() > 0: rows.append((m, 1 - ((a - b)**2).sum() / (a**2).sum()))
    return pd.DataFrame(rows, columns=["muncode", "wr2"])

res = []
for nm, _, _ in MODELS:
    pw = pooled_within(nm)
    pm = per_mun_within(nm).merge(D[["muncode", "trust_index"]], on="muncode", how="inner").dropna()
    corr = np.corrcoef(pm["trust_index"], pm["wr2"].clip(-2, 1))[0, 1]
    res.append({"model": nm, "pooled_within_r2": round(pw, 3),
                "trust_corr": round(corr, 3), "n_mun": len(pm)})
df = pd.DataFrame(res).sort_values("pooled_within_r2", ascending=False)
print("\n── Same ex-ante index vs each model's realized within-skill ──")
print(df.to_string(index=False))
df.to_csv(os.path.join(out, "exante_trust_across_models.csv"), index=False)
print(f"\nWrote {os.path.join(out, 'exante_trust_across_models.csv')}")
