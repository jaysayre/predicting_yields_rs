"""
Embeddings vs methodology: does applying AEF's distributional recipe (per-dim
mean / percentiles / sd / 8 quantile-bin histogram) to NDVI features close the
gap to AEF embeddings, or is the advantage the embeddings themselves?

We compare feature SOURCES under the SAME recipe by their information content --
an ADC-level oracle: HistGradientBoosting trained directly on ADC census yields,
scored out-of-sample by 5-fold GroupKFold over municipalities (identical to the
"Oracle (ADC-trained)" row of the accuracy tables). All feature sets are evaluated
on the SAME common ADC set so differences are source, not sample.

Feature sources (2022, cropland-masked, quantile-binned):
  NDVI qbin   parity  Data/cropland_features/adc_ndvi_qbin_parity_2022.parquet   (10 NDVI dims)
  AEF  qbin   hist    Data/alpha_earth/alpha_earth_mex_adcs_qbin_hist.parquet    (64 AEF dims)
  AEF  fixed  hist    Data/alpha_earth/alpha_earth_mex_adcs_binned_hist.parquet  (64 AEF dims)

Ground truth: INEGI 2022 census ADC maize yield (adc_aef_hist_ens_eval.parquet).

Run: ~/miniforge3/envs/geo_env/bin/python aefn2_parity_oracle.py
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
    gm = s.groupby(g)[[y, p]].transform("mean")
    gmean = s.groupby(g)[[y, p]].mean()
    return (len(s), r2(s[y], s[p]), r2(gmean[y], gmean[p]),
            r2(s[y] - gm[y], s[p] - gm[p]))

# ── ground truth (census 2022 ADC maize yield) ──────────────
ev = pd.read_parquet(os.path.join(D, "predictions", "adc_aef_hist_ens_eval.parquet"))
ev = ev[["adc", "muncode", "yield"]].dropna(subset=["yield"])
ev["adc"] = ev["adc"].astype(str)                       # census format: dashes stripped

# ── feature sources ─────────────────────────────────────────
def load(path, year_filter=True):
    d = pd.read_parquet(path)
    if year_filter and "year" in d.columns:
        d = d[d["year"] == 2022]
    if "gs_year" in d.columns:
        d = d[d["gs_year"] == 2022]
    # crosswalk to census adc id (gb_aef_hist_ensemble.py: strip the dash)
    d["adc"] = d["adcid"].astype(str).str.replace("-", "", regex=False)
    return d

ndvi = load(os.path.join(D, "cropland_features", "adc_ndvi_qbin_parity_2022.parquet"))
aefq = load(os.path.join(D, "alpha_earth", "alpha_earth_mex_adcs_qbin_hist.parquet"))
aeff = load(os.path.join(D, "alpha_earth", "alpha_earth_mex_adcs_binned_hist.parquet"))

nd_feat = [c for c in ndvi.columns if c not in ("adcid", "adc", "gs_year", "muncode", "year")]
aq_feat = [c for c in aefq.columns if c.startswith(("A",)) and "_b" in c]
af_feat = [c for c in aeff.columns if c.startswith(("A",)) and "_b" in c]
# quantile-bin-only subset of the NDVI parity set (matches AEF's qbin recipe exactly)
nd_qb   = [c for c in nd_feat if "_qb" in c]

# common ADC set = intersection so the comparison is source, not sample
common = set(ev["adc"]) & set(ndvi["adc"]) & set(aefq["adc"]) & set(aeff["adc"])
print(f"common ADCs (states 01-11 census ∩ all sources): {len(common):,}")

SETS = [
    ("NDVI qbin (10 dims, 80 feat)", ndvi, nd_qb),
    ("NDVI full (10 dims, 150 feat)", ndvi, nd_feat),
    ("AEF qbin (64 dims, 512 feat)", aefq, aq_feat),
    ("AEF fixed-hist (64 dims, 512 feat)", aeff, af_feat),
]

print(f"\n{'feature source':34s} {'N':>7} {'R2':>7} {'Btw':>7} {'Wtn':>7}")
rows = []
for name, d, feat in SETS:
    m = d[d["adc"].isin(common)].dropna(subset=feat).drop_duplicates("adc")
    g = m[["adc"] + feat].merge(ev, on="adc", how="inner")   # muncode/yield from ev only
    X, y, grp = g[feat].values, g["yield"].values, g["muncode"].values
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=31, random_state=0)
    g["pred"] = cross_val_predict(gb, X, y, groups=grp, cv=GroupKFold(5))
    n, ov, bt, wt = decomp(g)
    rows.append((name, n, ov, bt, wt))
    print(f"{name:34s} {n:7,} {ov:7.3f} {bt:7.3f} {wt:7.3f}")

pd.DataFrame(rows, columns=["source", "N", "R2", "between_R2", "within_R2"]).to_csv(
    os.path.join(proj, "plots", "aefn2_parity_oracle.csv"), index=False)
print("\nInterpretation: if NDVI-under-AEF-recipe within-R2 stays well below the AEF")
print("sources, the advantage is the EMBEDDINGS, not the distributional methodology.")
