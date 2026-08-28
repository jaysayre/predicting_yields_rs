"""Fall-winter-trained AEF Hist Ensemble (paper Sec 5.5, 2026-08-28).

Trains the ensemble on FALL-WINTER SIAP yields (10,305 mun-years, 2017+) and
evaluates against the census O-I ADC yields. Result cited in the seasonal-
heterogeneity discussion: overall R2 rises 0.606 -> 0.729 vs the P-V-trained
model, but between-mun R2 is -0.58 (vs SIAP benchmark's own -0.64) and within
stays negative -- the retrained model inherits SIAP's O-I municipal structure,
so fall-winter accuracy is limited by SIAP-census divergence, not training
domain.

Run:  ~/miniforge3/envs/ml_cuda/bin/python gb_aef_hist_ensemble_oi_trained.py
Writes oi_trained_ensemble.json (set $SCRATCH to redirect; defaults to /tmp).
"""
import os, numpy as np, pandas as pd, json, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
aefd  =  os.path.join(proj, "Data", "alpha_earth")
INEGI =  os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                      "LM2304-CA22-2025-09-29-superficie_ENTREGA")
agland_path =  os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siapp =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")

mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  [f"A{d:02d}{s}" for d in range(64)
              for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']]
pct_combined =  pct_cols + mean_cols
cfg = dict(max_iter=1500, max_depth=8, learning_rate=0.03,
           min_samples_leaf=5, random_state=42, early_stopping=False)
W_BIN = 0.4

siap =  pd.read_stata(siapp)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
siap["yield"]   =  siap["q"]/siap["ha_planted"]
print("seasons:", siap["growing_season"].unique().tolist())
st =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Fall-Winter") &
           (siap["year"] >= 2017)]
st =  st[st["yield"].notna() & (st["yield"] > 0) &
         ~st["muncode"].str.endswith("000")][["muncode", "year", "yield"]]
print(f"O-I training mun-years: {len(st):,} across {st['muncode'].nunique():,} munis")

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
train_bin =  mun_bh.merge(st, on=["muncode", "year"], how="inner")
train_pct =  mun_pct_full.merge(st, on=["muncode", "year"], how="inner")
print(f"train rows: bin={len(train_bin):,} pct={len(train_pct):,}")

def subsample_bins(train_df, K=5, N=2, seed=42):
    rng = np.random.default_rng(seed)
    n_mun = len(train_df)
    arr = train_df[bin_cols].values.reshape(n_mun, 64, 8).astype(np.float64)
    arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    sm = arr.sum(axis=2, keepdims=True); sm[sm < 1e-8] = 1.0; arr = arr/sm
    ys = train_df["yield"].values
    ab, ay = [], []
    for k in range(K):
        o = np.zeros((n_mun, 64, 8), dtype=np.float32)
        for d in range(64):
            for i in range(n_mun):
                o[i, d, :] = rng.multinomial(N, arr[i, d, :])/N
        ab.append(o.reshape(n_mun, 512)); ay.append(ys)
    return np.vstack(ab), np.concatenate(ay)

print("training O-I ensemble...")
m_pct = HistGradientBoostingRegressor(**cfg)
m_pct.fit(train_pct[pct_combined].fillna(0).values.astype(np.float32), train_pct["yield"].values)
aug_b, aug_y = subsample_bins(train_bin)
m_bin = HistGradientBoostingRegressor(**cfg)
m_bin.fit(aug_b.astype(np.float32), aug_y)

# ADC features 2022
adc_bh =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs_binned_hist.parquet"))
adc_bh =  adc_bh[adc_bh["year"] == 2022].copy()
adc_pct =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs_hist.parquet"))
adc_pct =  adc_pct[adc_pct["year"] == 2022].copy()
adc_m =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs.parquet"))
adc_m =  adc_m[adc_m["year"] == 2022].copy()
apf =  adc_pct.merge(adc_m[["adcid", "year"] + mean_cols], on=["adcid", "year"], how="inner")
combo =  adc_bh.merge(apf[["adcid", "year"] + pct_combined], on=["adcid", "year"], how="inner")
_bn =  combo[bin_cols].isna().all(axis=1); _pn = combo[pct_cols].isna().all(axis=1)
combo =  combo[~(_bn | _pn)].copy()
combo["pred"] =  (W_BIN*m_bin.predict(combo[bin_cols].fillna(0).values.astype(np.float32)).clip(0)
                  + (1-W_BIN)*m_pct.predict(combo[pct_combined].fillna(0).values.astype(np.float32)).clip(0))
combo["adc"] =  combo["adcid"].str.replace("-", "", regex=False)

# census O-I ground truth
ca_szn =  pd.read_stata(os.path.join(INEGI, "adc_land_szn_ca22_adc07.dta"))
print("szn types:", ca_szn["type"].unique().tolist())
gt =  ca_szn[(ca_szn["name"] == "Maize") & (ca_szn["type"] == "o-i")][
        ["adc", "muncode", "yield"]].rename(columns={"yield": "y"})
df =  gt.merge(combo[["adc", "pred"]], on="adc", how="left").dropna(subset=["y", "pred"])
df["muncode"] =  df["muncode"].astype(str).str.zfill(5)

# O-I SIAP anchor + agland weights for correction
ag =  pd.read_csv(agland_path); ag["adc"] =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
df =  df.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
df["w"] =  np.where(df["siap_agland_area"] > 0, df["siap_agland_area"], 1.0)
sm22 =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Fall-Winter") &
             (siap["year"] == 2022) & (~siap["muncode"].str.endswith("000"))]
sm22 =  sm22.groupby("muncode").agg(q=("q", "sum"), ha=("ha_planted", "sum"))
sm22["siap_oi"] =  sm22["q"]/sm22["ha"]
df =  df.merge(sm22[["siap_oi"]], on="muncode", how="left")
agg =  df.groupby("muncode").apply(lambda g: np.average(g["pred"], weights=g["w"]))
df["pred_agg"] =  df["muncode"].map(agg)
df["pred_corr"] =  df["pred"] + (df["siap_oi"] - df["pred_agg"])

def metrics(d, p):
    s =  d[["y", p, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    gm =  s.groupby("muncode")[["y", p]].transform("mean")
    ov =  1 - ((s["y"]-s[p])**2).sum()/((s["y"]-s["y"].mean())**2).sum()
    wt =  1 - (((s[p]-gm[p])-(s["y"]-gm["y"]))**2).sum()/((s["y"]-gm["y"])**2).sum()
    g  =  s.groupby("muncode")[["y", p]].mean()
    bt =  1 - ((g["y"]-g[p])**2).sum()/((g["y"]-g["y"].mean())**2).sum()
    rm =  float(np.sqrt(((s["y"]-s[p])**2).mean()))
    return dict(N=len(s), R2=round(ov, 3), Btw=round(bt, 3), Wtn=round(wt, 3), RMSE=round(rm, 3))

gmean =  df.groupby("muncode")["pred"].transform("mean")
df["pred_sh"] =  gmean + (2/3)*(df["pred"] - gmean)
out =  {"train_munyears": int(len(st)),
        "OI-trained Raw":   metrics(df, "pred"),
        "OI-trained Corr":  metrics(df, "pred_corr"),
        "OI-trained Shrink(2/3)": metrics(df, "pred_sh"),
        "reference (P-V-trained, from paper table)": {
            "Raw": {"N": 7686, "R2": 0.606, "Btw": 0.655, "Wtn": -0.317},
            "Corr": {"N": 6647, "R2": 0.737, "Btw": -0.643, "Wtn": -0.399},
            "Shrink": {"N": 7686, "R2": 0.581, "Btw": 0.594, "Wtn": 0.052}}}
print(json.dumps(out, indent=1))
sp = os.environ.get("SCRATCH", "/tmp")
json.dump(out, open(os.path.join(sp, "oi_trained_ensemble.json"), "w"), indent=1)
