"""
Cross-model ex-ante trust index.

Shows (a) that the realized farm-level (within-municipality) skill ranking of the
six models is very different from---and indeed inverts---the municipality-level
ranking of Table \ref{tab:validation_mun} (the Agg-NN is first at the municipality
level but last at the farm level), and (b) that the same ex-ante features that
predict the AEF Hist Ensemble's within-municipality skill also predict EVERY
model's within-municipality skill out-of-sample. The trust index is therefore a
tool for ex-ante model selection, not just targeting of a single model.

Output: plots/coauthor_extras_paper/exante_trust_across_models.csv
Run:    ~/miniforge3/envs/geo_env/bin/python exante_trust_across_models.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P    = os.path.join(proj, "Data", "predictions")
out  = os.path.join(proj, "plots", "coauthor_extras_paper")

ev = pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))[["adc","muncode","yield","pred"]].rename(columns={"pred":"AEF Hist Ens."})
MODELS = [("NDVI Hist.","adc_harmonic_h3_fixed_preds.parquet","pred"),
          ("NDVI Q-Hist.","adc_harmonic_h3_quantile_preds.parquet","pred"),
          ("AEF mean","adc_alpha_earth_preds.csv","yield_pred"),
          ("Agg-NN","adc_mlp_yield_preds.csv","pred_yield"),
          ("AEF Hist","adc_aef_hist_gb_preds.parquet","yield_pred"),
          ("AEF Hist Ens.", None, None)]
for nm,f,c in MODELS:
    if f is None: continue
    d = pd.read_parquet(os.path.join(P,f)) if f.endswith("parquet") else pd.read_csv(os.path.join(P,f))
    if "year" in d.columns: d = d[d["year"]==2022]
    k = "adc" if "adc" in d.columns else "adcid"; d["adc"] = d[k].astype(str).str.replace("-","",regex=False)
    ev = ev.merge(d[["adc",c]].rename(columns={c:nm}).dropna().drop_duplicates("adc"), on="adc", how="left")

tr = pd.read_csv(os.path.join(out, "exante_trust_index.csv"), dtype={"muncode":str})
FEAT = ["log_n_adc","irrig_share","maize_share","log_mean_ha_adc","sd_pred","mean_pred","log_siap_sd",
        "siap_mean_yield","aef_heterogeneity","irrig_share_sd","frac_irrigated","adc_area_cv",
        "maizeland_sd","aef_spread","aef_pc1_frac","aef_eff_dim"]

def pooled_within(p):
    s = ev[["muncode","yield",p]].dropna(); c = s.groupby("muncode")["yield"].transform("size"); s = s[c>=2]
    a = (s["yield"]-s.groupby("muncode")["yield"].transform("mean")).values
    b = (s[p]-s.groupby("muncode")[p].transform("mean")).values
    return 1 - np.sum((a-b)**2)/np.sum(a*a)

def per_mun_within(p):
    rows = []
    for m,g in ev.dropna(subset=["yield",p]).groupby("muncode"):
        if len(g) < 5: continue
        a = g["yield"]-g["yield"].mean(); b = g[p]-g[p].mean()
        if (a**2).sum() > 0: rows.append((m, 1-((a-b)**2).sum()/(a**2).sum()))
    return pd.DataFrame(rows, columns=["muncode","wr2"])

res = []
for nm,_,_ in MODELS:
    pw = pooled_within(nm)
    pm = per_mun_within(nm).merge(tr[["muncode"]+FEAT], on="muncode", how="inner").dropna(subset=FEAT)
    y = pm["wr2"].clip(-2, 1)
    rf = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    pred = cross_val_predict(rf, pm[FEAT], y, cv=KFold(5, shuffle=True, random_state=42))
    corr = np.corrcoef(pred, y)[0, 1]
    res.append({"model": nm, "pooled_within_r2": round(pw, 3), "trust_corr": round(corr, 3), "n_mun": len(pm)})

df = pd.DataFrame(res).sort_values("pooled_within_r2", ascending=False)
print(df.to_string(index=False))
df.to_csv(os.path.join(out, "exante_trust_across_models.csv"), index=False)
print(f"\nWrote {os.path.join(out,'exante_trust_across_models.csv')}")
