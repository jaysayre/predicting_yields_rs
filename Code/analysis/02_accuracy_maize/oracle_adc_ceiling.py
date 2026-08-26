"""
Oracle ADC-trained ceiling (NOT deployable) for the benchmark section of the
main accuracy tables.

Trains HistGradientBoosting directly on ADC-level AEF features (the 448 mean +
distributional features) using ADC-level census labels, scored out-of-fold by
5-fold GroupKFold over municipalities. This requires the very ADC-level ground
truth our method avoids, so it is an upper bound on how much yield signal the
embeddings carry at the ADC level -- reported as the "Oracle (ADC-trained)" row
under the benchmark sections of Tables \ref{tab:accuracy_combined} /
\ref{tab:accuracy_spring_summer}.

Output:  Data/predictions/oracle_ceiling_2022.csv   (season,N,R2,Btw,Wtn,RMSE)
Run:     ~/miniforge3/envs/geo_env/bin/python oracle_adc_ceiling.py
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
aef  = os.path.join(proj, "Data", "alpha_earth")
pred = os.path.join(proj, "Data", "predictions")
ca22 = os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                    "LM2304-CA22-2025-09-29-superficie_ENTREGA")

mean_cols = [f"A{d:02d}" for d in range(64)]
pct_cols  = [f"A{d:02d}{s}" for d in range(64) for s in ["_p10","_p25","_p50","_p75","_p90","_stdDev"]]
FEAT = pct_cols + mean_cols                       # 448
cfg  = dict(max_iter=1500, max_depth=8, learning_rate=0.03,
            min_samples_leaf=5, random_state=42, early_stopping=False)

def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    return 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2) if m.sum() > 1 else np.nan
def within_r2(df, y, p, gc="muncode"):
    s = df[[y,p,gc]].replace([np.inf,-np.inf],np.nan).dropna()
    c = s.groupby(gc).size(); s = s[s[gc].isin(c[c>=2].index)]
    gm = s.groupby(gc)[[y,p]].transform("mean"); return r2(s[y]-gm[y], s[p]-gm[p])
def between_r2(df, y, p, gc="muncode"):
    s = df[[y,p,gc]].replace([np.inf,-np.inf],np.nan).dropna()
    g = s.groupby(gc)[[y,p]].mean(); return r2(g[y], g[p])

# ── ADC 448-feature frame (2022) ──
print("Loading ADC features...")
pct = pd.read_parquet(os.path.join(aef, "alpha_earth_mex_adcs_hist.parquet"))
pct = pct[pct["year"] == 2022].copy()
mn  = pd.read_parquet(os.path.join(aef, "alpha_earth_mex_adcs.parquet"))
mn  = mn[mn["year"] == 2022].copy()
X = pct.merge(mn[["adcid","year"]+mean_cols], on=["adcid","year"], how="inner")
X["adc"]     = X["adcid"].str.replace("-","",regex=False)
X["muncode"] = X["adcid"].str[:5]

# ── ADC census labels (combined + P-V) ──
ca = pd.read_stata(os.path.join(ca22, "adc_land_use_ca22_adc07.dta"))
gt = ca[ca["name"]=="Maize"][["adc","yield"]].rename(columns={"yield":"yield_comb"})
szn = pd.read_stata(os.path.join(ca22, "adc_land_szn_ca22_adc07.dta"))
pv = szn[(szn["name"]=="Maize")&(szn["type"]=="p-v")][["adc","yield"]].rename(columns={"yield":"yield_pv"})
gt = gt.merge(pv, on="adc", how="left")
d  = X.merge(gt, on="adc", how="inner")

def oracle(df, ycol, label):
    s = df[df[ycol].notna()].copy()
    cnt = s.groupby("muncode")[ycol].transform("size"); s = s[cnt>=2].reset_index(drop=True)
    Xm = s[FEAT].fillna(0).values.astype(np.float32); y = s[ycol].values.astype(np.float32)
    oof = np.zeros(len(s))
    for k,(tr,te) in enumerate(GroupKFold(5).split(Xm, groups=s["muncode"])):
        m = HistGradientBoostingRegressor(**cfg); m.fit(Xm[tr], y[tr]); oof[te] = m.predict(Xm[te])
        print(f"  {label} fold {k+1}/5")
    s["oof"] = oof.clip(0)
    rmse = np.sqrt(np.mean((s[ycol].values - s["oof"].values)**2))
    return dict(season=label, N=len(s), R2=round(r2(s[ycol],s["oof"]),3),
                Btw=round(between_r2(s,ycol,"oof"),3), Wtn=round(within_r2(s,ycol,"oof"),3),
                RMSE=round(rmse,3))

rows = [oracle(d, "yield_comb", "combined"), oracle(d, "yield_pv", "spring_summer")]
res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
out = os.path.join(pred, "oracle_ceiling_2022.csv")
res.to_csv(out, index=False)
print(f"\nWrote {out}")
