"""
De-risk: does cropland-masking the NDVI features materially change their ADC-level
within-municipality skill? Decides whether finishing the (multi-cycle) cropland-
masked NDVI extraction is worth it.

Same ADC-level oracle as aefn2_parity_oracle.py (HistGB, 5-fold GroupKFold by muni,
trained on ADC census labels), on the SAME common ADCs (2022, states 01-11):
  Unmasked NDVI  h3 quantile 2D-hist   Data/harmonic_features/adc_h3_quantile.parquet   (paper's current recipe)
  Masked   NDVI  per-dim quantile      Data/cropland_features/adc_ndvi_qbin_parity_2022.parquet  (aefn2)
  Masked   AEF   qbin  (ceiling ref)   Data/alpha_earth/alpha_earth_mex_adcs_qbin_hist.parquet

Run: ~/miniforge3/envs/geo_env/bin/python mask_effect_check.py
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
D    = os.path.join(proj, "Data")

def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
def decomp(df, y="yield", p="pred", g="muncode"):
    s = df[[y, p, g]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(g).size(); s = s[s[g].isin(c[c >= 2].index)]
    gm = s.groupby(g)[[y, p]].transform("mean"); gmean = s.groupby(g)[[y, p]].mean()
    return (len(s), r2(s[y], s[p]), r2(gmean[y], gmean[p]), r2(s[y]-gm[y], s[p]-gm[p]))

ev = pd.read_parquet(os.path.join(D, "predictions", "adc_aef_hist_ens_eval.parquet"))
ev = ev[["adc", "muncode", "yield"]].dropna(subset=["yield"]); ev["adc"] = ev["adc"].astype(str)

def strip(s): return s.astype(str).str.replace("-", "", regex=False)

# masked NDVI (aefn2) drives the common-ADC set
nd_m = pd.read_parquet(os.path.join(D, "cropland_features", "adc_ndvi_qbin_parity_2022.parquet"))
nd_m["adc"] = strip(nd_m["adcid"])
common = set(ev["adc"]) & set(nd_m["adc"])

# unmasked NDVI: read only 2022 + the common ADCs
nd_u = pd.read_parquet(os.path.join(D, "harmonic_features", "adc_h3_quantile.parquet"),
                       filters=[("year", "==", 2022)])
nd_u["adc"] = strip(nd_u["adcid"]); nd_u = nd_u[nd_u["adc"].isin(common)]

aefq = pd.read_parquet(os.path.join(D, "alpha_earth", "alpha_earth_mex_adcs_qbin_hist.parquet"))
aefq = aefq[aefq["year"] == 2022]; aefq["adc"] = strip(aefq["adcid"])

common &= set(nd_u["adc"]) & set(aefq["adc"])
print(f"common ADCs: {len(common):,}")

nd_m_feat = [c for c in nd_m.columns if c not in ("adcid", "adc", "gs_year", "muncode", "year")]
nd_u_feat = [c for c in nd_u.columns if c.startswith("h3q_")]
aq_feat   = [c for c in aefq.columns if c.startswith("A") and "_b" in c]

SETS = [
    ("NDVI UNMASKED (h3 qhist, 768f)", nd_u, nd_u_feat),
    ("NDVI MASKED (per-dim qbin, 150f)", nd_m, nd_m_feat),
    ("AEF MASKED (qbin, 512f)  [ceiling]", aefq, aq_feat),
]
print(f"\n{'feature set':38s} {'N':>7} {'R2':>7} {'Btw':>7} {'Wtn':>7}")
rows = []
for name, d, feat in SETS:
    m = d[d["adc"].isin(common)].dropna(subset=feat).drop_duplicates("adc")
    g = m[["adc"] + feat].merge(ev, on="adc", how="inner")
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=31, random_state=0)
    g["pred"] = cross_val_predict(gb, g[feat].values, g["yield"].values,
                                  groups=g["muncode"].values, cv=GroupKFold(5))
    n, ov, bt, wt = decomp(g); rows.append((name, n, ov, bt, wt))
    print(f"{name:38s} {n:7,} {ov:7.3f} {bt:7.3f} {wt:7.3f}")

pd.DataFrame(rows, columns=["set", "N", "R2", "between_R2", "within_R2"]).to_csv(
    os.path.join(proj, "plots", "mask_effect_check.csv"), index=False)
