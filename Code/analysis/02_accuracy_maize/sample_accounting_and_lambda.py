"""Sample-size accounting + census-free lambda gauging (paper Sec 3.6 + appendix).

Produces the numbers cited in:
  - Sec 3.6: dispersion-ratio gauge from public data (pred within-mun SD 1.09 t/ha
    vs SIAP within-state mun SD 1.66 vs census within-mun ADC SD 1.53; r-hat ~0.66)
  - Sec 2 / Sec 5.8: CIMMYT plot-year counts (24,679 grain plot-years -> 23,670 used)
  - Appendix sample-size accounting paragraph (N waterfall for the accuracy tables)
  - Negative results (not in paper): rho/r estimated at the mun-within-state scale
    from SIAP holdout predictions gives lambda = 1 (overdispersion only appears at
    the ADC scale transfer), so lambda cannot be point-identified from SIAP alone.

Run:  ~/miniforge3/envs/geo_env/bin/python sample_accounting_and_lambda.py
Writes lambda_cimmyt_ncounts.json (set $SCRATCH to redirect; defaults to /tmp).
"""
import os, numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P     =  os.path.join(proj, "Data", "predictions")
aefd  =  os.path.join(proj, "Data", "alpha_earth")
agp   =  os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siapp =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
out = {}

# ── 1. lambda_SIAP ───────────────────────────────────────
aef =  pd.read_parquet(os.path.join(aefd, "alpha_earth_mex_adcs.parquet"), columns=["adcid", "year"])
aef["muncode"] =  aef["adcid"].str[:5]
ag  =  pd.read_csv(agp)
ag["w"] =  np.where(ag["siap_agland_area"] > 0, ag["siap_agland_area"], 1.0)
aef =  aef.merge(ag[["adcid", "w"]], on="adcid", how="left"); aef["w"] = aef["w"].fillna(1.0)

siap =  pd.read_stata(siapp)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
siap["yield"]   =  siap["q"]/siap["ha_planted"]
sm  =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
            (siap["year"] >= 2017)][["muncode", "year", "yield"]].dropna()

valid =  sorted(set(zip(aef["muncode"], aef["year"])) & set(zip(sm["muncode"], sm["year"])))
umuns =  sorted(set(k[0] for k in valid))
np.random.seed(42); np.random.shuffle(umuns)
val_muns =  set(umuns[int(0.8*len(umuns)):])

def rho_r(d, y, p, gcols):
    g  =  d.groupby(gcols)[[y, p]]
    gm =  g.transform("mean")
    cnt = d.groupby(gcols)[y].transform("size")
    a  =  (d[y]-gm[y])[cnt >= 2]; b = (d[p]-gm[p])[cnt >= 2]
    rho =  np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))
    r   =  np.sqrt(np.sum(b*b)/np.sum(a*a))
    return rho, r, float(np.clip(rho/r, 0, 1)), int(len(a))

def agg_mun(pred_file, col="pred"):
    p =  pd.read_parquet(os.path.join(P, pred_file)).dropna(subset=[col])
    p["muncode"] =  p["adcid"].astype(str).str[:5]
    p =  p.merge(aef[["adcid", "year", "w"]], on=["adcid", "year"], how="left")
    p["w"] =  p["w"].fillna(1.0); p["wv"] = p[col]*p["w"]
    g =  p.groupby(["muncode", "year"]).agg(wv=("wv", "sum"), w=("w", "sum")).reset_index()
    g["pred"] =  g["wv"]/g["w"]
    return g[["muncode", "year", "pred"]]

hold =  agg_mun("adc_aef_hist_ens_holdout_preds.parquet")
hv   =  hold.merge(sm, on=["muncode", "year"], how="inner")
hv   =  hv[hv["muncode"].isin(val_muns)].copy()
hv["state"] =  hv["muncode"].str[:2]
res = {}
res["ens_holdout_stateyear_allyrs"] =  rho_r(hv, "yield", "pred", ["state", "year"])
h22  =  hv[hv["year"] == 2022]
res["ens_holdout_state_2022"]       =  rho_r(h22, "yield", "pred", ["state"])
# AEF Hist muni 5-fold CV preds (fully out-of-sample all munis)
kf =  pd.read_parquet(os.path.join(P, "mun_aef_hist_gb_kfold_preds.parquet"))
kf["muncode"] =  kf["muncode"].astype(str).str.zfill(5)
pcol =  [c for c in kf.columns if "pred" in c][0]
kv =  kf.merge(sm.rename(columns={"yield": "y_siap"}), on=["muncode", "year"], how="inner").dropna(subset=[pcol])
kv["state"] =  kv["muncode"].str[:2]
res["hist_kfold_stateyear_allyrs"] =  rho_r(kv, "y_siap", pcol, ["state", "year"])
res["hist_kfold_state_2022"]       =  rho_r(kv[kv["year"] == 2022], "y_siap", pcol, ["state"])
out["lambda_siap"] =  {k: {"rho": round(v[0], 3), "r": round(v[1], 3),
                           "lam": round(v[2], 3), "n": v[3]} for k, v in res.items()}

# ── 2. CIMMYT counts ─────────────────────────────────────
cx =  pd.read_excel(os.path.join(proj, "Data", "CIMMYT", "Farmer_plots",
                                 "2.-Sowing_harvest_yields_2012-2022_02.xlsx"))
yr =  (cx["YEAR"] >= 2017) & (cx["YEAR"] <= 2022)
mz =  cx["CROP"] == "MAIZE"
gr =  cx["PRODUCT.OBTAINED"] == "GRAIN"
yv =  cx["ACTUAL.YIELD.(UNIT/HA)"]
out["cimmyt_counts"] =  {
  "all_rows": int(len(cx)),
  "rows_2017_2022": int(yr.sum()),
  "maize_2017_2022": int((yr & mz).sum()),
  "maize_grain_2017_2022": int((yr & mz & gr).sum()),
  "maize_grain_yield_pos": int((yr & mz & gr & yv.notna() & (yv > 0)).sum()),
  "maize_grain_clean_03_20": int((yr & mz & gr & yv.notna() & (yv >= 0.3) & (yv <= 20)).sum()),
  "maize_any_product_yield_pos": int((yr & mz & yv.notna() & (yv > 0)).sum()),
}
# feature-matched: replicate merge with cimmyt features
try:
    feats = None
    for cand in ["alpha_earth_cimmyt_binned_hist.parquet", "alpha_earth_cimmyt_hist.parquet"]:
        fp = os.path.join(aefd, cand)
        if os.path.exists(fp):
            feats = pd.read_parquet(fp); out["cimmyt_feat_file"] = cand
            out["cimmyt_feat_rows"] = int(len(feats)); break
    if feats is not None and "plot_id" in feats.columns:
        cc = cx[yr & mz & gr & yv.notna() & (yv >= 0.3) & (yv <= 20)][["PLOT.ID", "YEAR"]].copy()
        cc.columns = ["plot_id", "year"]; cc["plot_id"] = cc["plot_id"].astype(str)
        feats["plot_id"] = feats["plot_id"].astype(str)
        m = cc.merge(feats[["plot_id", "year"]].drop_duplicates(), on=["plot_id", "year"], how="inner")
        out["cimmyt_matched_clean"] = int(len(m))
except Exception as e:
    out["cimmyt_feat_err"] = str(e)

# ── 3. N waterfall ───────────────────────────────────────
ev =  pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))
hist =  pd.read_parquet(os.path.join(P, "adc_aef_hist_gb_preds.parquet"))
hist =  hist[hist["year"] == 2022].copy(); hist["adc"] = hist["adcid"].str.replace("-", "", regex=False)
ndvi =  pd.read_parquet(os.path.join(P, "adc_aefn2_masked_preds.parquet"))
ndvi["adc"] =  ndvi["adcid"].str.replace("-", "", regex=False)
sm22 =  siap[(siap["name"] == "Maize") & (siap["year"] == 2022) &
             (~siap["muncode"].str.endswith("000"))]
pvm  =  sm22[sm22["growing_season"] == "Spring-Summer"].groupby("muncode")["yield"].mean().rename("siap_pv")
alm  =  sm22.groupby("muncode").apply(lambda g: g["q"].sum()/g["ha_planted"].sum()).rename("siap_all")
ev =  ev.merge(hist[["adc", "yield_pred"]].rename(columns={"yield_pred": "p_hist"}), on="adc", how="left")
ev =  ev.merge(ndvi[["adc", "pred"]].rename(columns={"pred": "p_ndvi"}), on="adc", how="left")
ev =  ev.merge(pvm, on="muncode", how="left").merge(alm, on="muncode", how="left")
def n(mask): return int(mask.sum())
w = {}
for tag, ycol, anchor in [("PV", "yield_pv", "siap_pv"), ("combined", "yield", "siap_all")]:
    y = ev[ycol].notna()
    w[tag] = {
      "census_rows": n(y),
      "with_siap_anchor(SIAP benchmark)": n(y & ev[anchor].notna()),
      "with_AEF_hist_features": n(y & ev["p_hist"].notna()),
      "with_NDVI_features": n(y & ev["p_ndvi"].notna()),
      "with_ens_pred": n(y & ev["pred"].notna()),
      "ens_and_anchor(Corr row)": n(y & ev["pred"].notna() & ev[anchor].notna()),
      "hist_and_anchor": n(y & ev["p_hist"].notna() & ev[anchor].notna()),
    }
out["n_waterfall"] = w
# ── 4. Census-free dispersion gauge (Sec 3.6) ────────────
d  =  ev.dropna(subset=["yield_pv", "pred"]).copy()
cnt =  d.groupby("muncode")["yield_pv"].transform("size"); d = d[cnt >= 2]
gm  =  d.groupby("muncode")[["yield_pv", "pred"]].transform("mean")
ydev =  (d["yield_pv"]-gm["yield_pv"]).values
pdev =  (d["pred"]-gm["pred"]).values
smpv =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
             (siap["year"] == 2022) & (~siap["muncode"].str.endswith("000"))].copy()
smpv["y"] =  smpv["q"]/smpv["ha_planted"]; smpv = smpv.dropna(subset=["y"])
smpv["state"] =  smpv["muncode"].str[:2]
gs =  smpv.groupby("state")["y"]
sdev =  (smpv["y"] - gs.transform("mean"))[gs.transform("size") >= 2]
out["dispersion_gauge"] =  {
  "sd_pred_dev_adc":     round(float(np.std(pdev)), 3),
  "sd_y_dev_census_adc": round(float(np.std(ydev)), 3),
  "sd_proxy_siap_within_state_mun": round(float(np.std(sdev)), 3),
  "r_true":  round(float(np.std(pdev)/np.std(ydev)), 3),
  "r_hat_public": round(float(np.std(pdev)/np.std(sdev)), 3),
  "rho_true": round(float(np.sum(ydev*pdev)/np.sqrt(np.sum(ydev**2)*np.sum(pdev**2))), 3)}

print(json.dumps(out, indent=1, default=str))
sp = os.environ.get("SCRATCH", "/tmp")
json.dump(out, open(os.path.join(sp, "lambda_cimmyt_ncounts.json"), "w"), indent=1, default=str)
