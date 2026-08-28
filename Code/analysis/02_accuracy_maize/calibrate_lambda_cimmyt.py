"""Census-free shrinkage-lambda estimator (paper Sec 3.6, 2026-08-27).

lambda* = rho/r.  r comes from public data (see sample_accounting_and_lambda.py:
predicted within-mun ADC dispersion 1.09 t/ha vs SIAP within-state mun dispersion
1.66 -> r_hat = 0.655).  rho is scale-free, so it is estimated here from the
INDEPENDENT CIMMYT plot network (never the census): trains the AEF Hist Ensemble
exactly as gb_aef_hist_ensemble.py / accuracy_cimmyt_profile.py, predicts at
CIMMYT plots, and computes the within-municipality correlation rho_hat = 0.43
(census value 0.48).  lambda_hat = rho_hat/r_hat = 0.65, coinciding with the
a priori 2/3 and the census-CV optimum 0.669.

Also reports the share of matched plot-years whose AEF features are entirely
null (no WorldCover cropland pixels inside the plot polygon): ~28%.

Run:  ~/miniforge3/envs/ml_cuda/bin/python calibrate_lambda_cimmyt.py  (~2 min)
Writes lambda_cimmyt_v2.json (set $SCRATCH to redirect; defaults to /tmp).
"""
import os, numpy as np, pandas as pd, json, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
aefd  =  os.path.join(proj, "Data", "alpha_earth")
siapp =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
cim   =  os.path.join(proj, "Data", "CIMMYT", "Farmer_plots",
                      "2.-Sowing_harvest_yields_2012-2022_02.xlsx")

mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  [f"A{d:02d}{s}" for d in range(64)
              for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']]
pct_combined =  pct_cols + mean_cols
cfg = dict(max_iter=1500, max_depth=8, learning_rate=0.03,
           min_samples_leaf=5, random_state=42, early_stopping=False)
W_BIN = 0.4

cm =  pd.read_parquet(os.path.join(aefd, "alpha_earth_cimmyt_plot.parquet"))
ch =  pd.read_parquet(os.path.join(aefd, "alpha_earth_cimmyt_plot_hist.parquet"))
cb =  pd.read_parquet(os.path.join(aefd, "alpha_earth_cimmyt_plot_binned_hist.parquet"))
for df in [cm, ch, cb]: df["plot_id"] = df["plot_id"].astype(str)
cpf =  ch.merge(cm[["plot_id", "year"] + mean_cols], on=["plot_id", "year"], how="inner")
combo =  cb.merge(cpf[["plot_id", "year"] + pct_combined], on=["plot_id", "year"], how="inner")
print("combo rows:", len(combo))

mun_bh  =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_mun_binned_hist.parquet"))
bin_cols =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])
mun_pct =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_mun_hist.parquet"))
for cc, z in [("CVE_ENT", 2), ("CVE_MUN", 3)]:
    mun_pct[cc] = mun_pct[cc].astype(str).str.zfill(z)
mun_pct["muncode"] =  mun_pct["CVE_ENT"] + mun_pct["CVE_MUN"]
mun_mean =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_muns.parquet"))
for cc, z in [("CVE_ENT", 2), ("CVE_MUN", 3)]:
    mun_mean[cc] = mun_mean[cc].astype(str).str.zfill(z)
mun_mean["muncode"] =  mun_mean["CVE_ENT"] + mun_mean["CVE_MUN"]
mun_pct_full =  mun_pct.merge(mun_mean[["muncode", "year"] + mean_cols],
                              on=["muncode", "year"], how="inner")

siap =  pd.read_stata(siapp)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
siap["yield"]   =  siap["q"]/siap["ha_planted"]
st =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
           (siap["year"] >= 2017)]
st =  st[st["yield"].notna() & (st["yield"] > 0) &
         ~st["muncode"].str.endswith("000")][["muncode", "year", "yield"]]
train_bin =  mun_bh.merge(st, on=["muncode", "year"], how="inner")
train_pct =  mun_pct_full.merge(st, on=["muncode", "year"], how="inner")

def subsample_bins(train_df, K=5, N=2, seed=42):
    rng = np.random.default_rng(seed)
    n_mun = len(train_df)
    arr = train_df[bin_cols].values.reshape(n_mun, 64, 8).astype(np.float64)
    arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    s = arr.sum(axis=2, keepdims=True); s[s < 1e-8] = 1.0; arr = arr/s
    ys = train_df["yield"].values
    ab, ay = [], []
    for k in range(K):
        o = np.zeros((n_mun, 64, 8), dtype=np.float32)
        for d in range(64):
            for i in range(n_mun):
                o[i, d, :] = rng.multinomial(N, arr[i, d, :])/N
        ab.append(o.reshape(n_mun, 512)); ay.append(ys)
    return np.vstack(ab), np.concatenate(ay)

print("training...")
m_pct = HistGradientBoostingRegressor(**cfg)
m_pct.fit(train_pct[pct_combined].fillna(0).values.astype(np.float32), train_pct["yield"].values)
aug_b, aug_y = subsample_bins(train_bin)
m_bin = HistGradientBoostingRegressor(**cfg)
m_bin.fit(aug_b.astype(np.float32), aug_y)

combo["pred"] =  (W_BIN*m_bin.predict(combo[bin_cols].fillna(0).values.astype(np.float32)).clip(0)
                  + (1-W_BIN)*m_pct.predict(combo[pct_combined].fillna(0).values.astype(np.float32)).clip(0))
combo["allnull"] =  combo[bin_cols].isna().all(axis=1) | combo[pct_cols].isna().all(axis=1)

cx =  pd.read_excel(cim)
cy =  cx[(cx["CROP"] == "MAIZE") & (cx["PRODUCT.OBTAINED"] == "GRAIN")
         & (cx["YEAR"] >= 2017) & (cx["YEAR"] <= 2022)
         & cx["ACTUAL.YIELD.(UNIT/HA)"].notna()
         & (cx["ACTUAL.YIELD.(UNIT/HA)"] >= 0.3)
         & (cx["ACTUAL.YIELD.(UNIT/HA)"] <= 20)].copy()
cy =  cy.rename(columns={"PLOT.ID": "plot_id", "YEAR": "year",
                         "ACTUAL.YIELD.(UNIT/HA)": "y"})
cy["plot_id"] =  cy["plot_id"].astype(str)
df =  cy[["plot_id", "year", "y"]].merge(
        combo[["plot_id", "year", "pred", "muncode", "allnull"]],
        on=["plot_id", "year"], how="inner")
df["muncode"] =  df["muncode"].astype(str).str.split(".").str[0].str.zfill(5)
out =  {"matched_clean": int(len(df)),
        "allnull_share": round(float(df["allnull"].mean()), 4)}

def rho_r(d):
    cnt =  d.groupby("muncode")["y"].transform("size")
    d  =  d[cnt >= 2]
    gm =  d.groupby("muncode")[["y", "pred"]].transform("mean")
    a  =  (d["y"]-gm["y"]).values; b = (d["pred"]-gm["pred"]).values
    rho =  float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
    r   =  float(np.sqrt(np.sum(b*b)/np.sum(a*a)))
    return {"rho": round(rho, 3), "r": round(r, 3),
            "lam_own": round(rho/r, 3), "n": int(len(d))}

out["full_sample"]      =  rho_r(df)
out["feat_nonnull"]     =  rho_r(df[~df["allnull"]])
# rho with SIAP ratio bands (representative plots)
siap_pv =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
                (siap["year"].between(2017, 2022)) & siap["yield"].notna()][
                ["muncode", "year", "yield"]].rename(columns={"yield": "ys"})
df =  df.merge(siap_pv, on=["muncode", "year"], how="left")
df["ratio"] =  df["y"]/df["ys"]
out["ratio_lt_1.3"] =  rho_r(df[df["ratio"] < 1.3])
out["ratio_lt_2.0"] =  rho_r(df[df["ratio"] < 2.0])
R_HAT_PUBLIC = 0.655
for k in ["full_sample", "feat_nonnull", "ratio_lt_1.3", "ratio_lt_2.0"]:
    out[k]["lam_via_public_r"] = round(out[k]["rho"]/R_HAT_PUBLIC, 3)
print(json.dumps(out, indent=1))
sp = os.environ.get("SCRATCH", "/tmp")
json.dump(out, open(os.path.join(sp, "lambda_cimmyt_v2.json"), "w"), indent=1)
