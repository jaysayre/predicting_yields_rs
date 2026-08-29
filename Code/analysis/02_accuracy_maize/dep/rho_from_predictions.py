"""
Census-free estimate of the within-municipality correlation rho -- and hence
lambda* = rho/r -- from the predictions themselves (paper Sec 3.6).

Idea (classical test theory): train the deployed AEF Hist Ensemble twice on
DISJOINT random halves of the training municipalities (same config, seeds 42).
The two prediction sets carry independent estimation noise, so their mutual
agreement measures prediction reliability at any scale. Municipal-scale
validity is publicly observable (held-out municipal predictions vs SIAP), and
if the signal's structural validity is scale-invariant, validity attenuates
with the square root of reliability:

    rho_adc  =  rho_mun * sqrt(rel_adc / rel_mun)

where
  rho_mun = within-state corr(half-model municipal preds, SIAP 2022 P-V yield),
            each half scored only on municipalities OUTSIDE its training half
  rel_mun = within-state corr(pA_mun, pB_mun) across 2022 municipalities
  rel_adc = within-municipality corr(pA_adc, pB_adc) across ensemble-eligible
            2022 ADCs
All inputs are SIAP + AEF embeddings + model outputs: no census, no CIMMYT.
lambda_hat = rho_adc / r_hat with the public dispersion gauge r_hat = 0.655.
The census value rho = 0.48 (lambda* = 2/3) is reported for comparison only.

Run:  ~/miniforge3/envs/ml_cuda/bin/python rho_from_predictions.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
aefd  =  os.path.join(proj, "Data", "alpha_earth")
siapp =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")

mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  [f"A{d:02d}{s}" for d in range(64)
              for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']]
pct_combined =  pct_cols + mean_cols
cfg =  dict(max_iter=1500, max_depth=8, learning_rate=0.03,
            min_samples_leaf=5, random_state=42, early_stopping=False)
W_BIN  =  0.4
R_HAT  =  0.655   # public dispersion gauge (Sec 3.6)

# ── SIAP P-V training frame (public) ─────────────────────
siap =  pd.read_stata(siapp)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
siap["yield"]   =  siap["q"]/siap["ha_planted"]
st =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
           (siap["year"] >= 2017)]
st =  st[st["yield"].notna() & (st["yield"] > 0) &
         ~st["muncode"].str.endswith("000")][["muncode", "year", "yield"]]

# ── municipal + ADC features ─────────────────────────────
mun_bh   =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_mun_binned_hist.parquet"))
bin_cols =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])
mun_pct  =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_mun_hist.parquet"))
for cc, z in [("CVE_ENT", 2), ("CVE_MUN", 3)]:
    mun_pct[cc] =  mun_pct[cc].astype(str).str.zfill(z)
mun_pct["muncode"] =  mun_pct["CVE_ENT"] + mun_pct["CVE_MUN"]
mun_mean =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_muns.parquet"))
for cc, z in [("CVE_ENT", 2), ("CVE_MUN", 3)]:
    mun_mean[cc] =  mun_mean[cc].astype(str).str.zfill(z)
mun_mean["muncode"] =  mun_mean["CVE_ENT"] + mun_mean["CVE_MUN"]
mun_pct_full =  mun_pct.merge(mun_mean[["muncode", "year"] + mean_cols],
                              on=["muncode", "year"], how="inner")

adc_bh  =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs_binned_hist.parquet"))
adc_bh  =  adc_bh[adc_bh["year"] == 2022].copy()
adc_pct =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs_hist.parquet"))
adc_pct =  adc_pct[adc_pct["year"] == 2022].copy()
adc_m   =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs.parquet"))
adc_m   =  adc_m[adc_m["year"] == 2022].copy()
apf   =  adc_pct.merge(adc_m[["adcid", "year"] + mean_cols], on=["adcid", "year"], how="inner")
combo =  adc_bh.merge(apf[["adcid", "year"] + pct_combined], on=["adcid", "year"], how="inner")
_bn =  combo[bin_cols].isna().all(axis=1); _pn =  combo[pct_cols].isna().all(axis=1)
combo =  combo[~(_bn | _pn)].copy()          # ensemble-eligible ADCs only
combo["muncode"] =  combo["adcid"].str[:5]

def subsample_bins(train_df, K=5, N=2, seed=42):
    rng =  np.random.default_rng(seed)
    n   =  len(train_df)
    arr =  train_df[bin_cols].values.reshape(n, 64, 8).astype(np.float64)
    arr =  np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    sm  =  arr.sum(axis=2, keepdims=True); sm[sm < 1e-8] = 1.0; arr = arr/sm
    ys  =  train_df["yield"].values
    ab, ay =  [], []
    for k in range(K):
        o =  np.zeros((n, 64, 8), dtype=np.float32)
        for d in range(64):
            for i in range(n):
                o[i, d, :] =  rng.multinomial(N, arr[i, d, :])/N
        ab.append(o.reshape(n, 512)); ay.append(ys)
    return np.vstack(ab), np.concatenate(ay)

def train_half(train_muns, tag):
    tb =  mun_bh.merge(st[st["muncode"].isin(train_muns)], on=["muncode", "year"], how="inner")
    tp =  mun_pct_full.merge(st[st["muncode"].isin(train_muns)], on=["muncode", "year"], how="inner")
    print(f"  {tag}: bin={len(tb):,} pct={len(tp):,} mun-years")
    mp =  HistGradientBoostingRegressor(**cfg)
    mp.fit(tp[pct_combined].fillna(0).values.astype(np.float32), tp["yield"].values)
    ab, ay =  subsample_bins(tb)
    mb =  HistGradientBoostingRegressor(**cfg)
    mb.fit(ab.astype(np.float32), ay)
    # ADC-level ensemble predictions
    p_adc =  (W_BIN*mb.predict(combo[bin_cols].fillna(0).values.astype(np.float32)).clip(0)
              + (1-W_BIN)*mp.predict(combo[pct_combined].fillna(0).values.astype(np.float32)).clip(0))
    # municipal-level ensemble predictions (2022)
    m22b =  mun_bh[mun_bh["year"] == 2022].copy()
    m22p =  mun_pct_full[mun_pct_full["year"] == 2022].copy()
    mm   =  m22b[["muncode"] + bin_cols].merge(m22p[["muncode"] + pct_combined], on="muncode", how="inner")
    p_mun =  (W_BIN*mb.predict(mm[bin_cols].fillna(0).values.astype(np.float32)).clip(0)
              + (1-W_BIN)*mp.predict(mm[pct_combined].fillna(0).values.astype(np.float32)).clip(0))
    return p_adc, pd.Series(p_mun, index=mm["muncode"].values)

def within_corr(df, a, b, gc):
    d  =  df[[a, b, gc]].dropna()
    cnt =  d.groupby(gc)[a].transform("size"); d = d[cnt >= 2]
    ga =  (d[a]-d.groupby(gc)[a].transform("mean")).values
    gb =  (d[b]-d.groupby(gc)[b].transform("mean")).values
    return float(np.sum(ga*gb)/np.sqrt(np.sum(ga*ga)*np.sum(gb*gb))), len(d)

# ── split municipalities, train the two half-models ──────
muns =  sorted(st["muncode"].unique())
rng  =  np.random.default_rng(42)
half =  rng.permutation(len(muns)) < len(muns)//2
M1   =  set(np.array(muns)[half]); M2 = set(np.array(muns)[~half])
print(f"training munis: {len(M1):,} vs {len(M2):,}")
pA_adc, pA_mun =  train_half(M1, "half A")
pB_adc, pB_mun =  train_half(M2, "half B")

adc =  combo[["adcid", "muncode"]].copy()
adc["pA"], adc["pB"] =  pA_adc, pB_adc

# SIAP 2022 P-V municipal truth (public)
s22 =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
            (siap["year"] == 2022) & (~siap["muncode"].str.endswith("000"))]
s22 =  s22.groupby("muncode").apply(lambda g: g["q"].sum()/g["ha_planted"].sum()).rename("y_siap")
mun =  pd.DataFrame({"pA": pA_mun, "pB": pB_mun}).join(s22, how="inner").reset_index(names="muncode")
mun["state"] =  mun["muncode"].str[:2]

rel_adc, n_adc =  within_corr(adc, "pA", "pB", "muncode")
rel_mun, n_mun =  within_corr(mun, "pA", "pB", "state")
vA, _ =  within_corr(mun[~mun["muncode"].isin(M1)], "pA", "y_siap", "state")   # A on munis it never saw
vB, _ =  within_corr(mun[~mun["muncode"].isin(M2)], "pB", "y_siap", "state")
rho_mun =  (vA + vB)/2
rho_adc =  rho_mun*np.sqrt(rel_adc/rel_mun)
lam     =  rho_adc/R_HAT

out =  {"rel_adc": round(rel_adc, 3), "n_adc": n_adc,
        "rel_mun": round(rel_mun, 3), "n_mun": n_mun,
        "rho_mun_heldout_A": round(vA, 3), "rho_mun_heldout_B": round(vB, 3),
        "rho_mun": round(rho_mun, 3),
        "rho_adc_hat": round(float(rho_adc), 3),
        "lambda_hat": round(float(lam), 3),
        "census_reference": {"rho": 0.48, "lambda": 0.667}}
print(json.dumps(out, indent=1))
sp =  os.environ.get("SCRATCH", "/tmp")
json.dump(out, open(os.path.join(sp, "rho_from_predictions.json"), "w"), indent=1)
