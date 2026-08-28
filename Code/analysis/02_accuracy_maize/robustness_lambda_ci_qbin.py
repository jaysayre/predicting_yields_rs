"""
Robustness computations cited in the paper prose (2026-08-27):
  A. lambda sensitivity curve for the AEF Hist Ensemble shrinkage (Sec. 3.6):
     within-R2 is flat in lambda; [0.55, 0.80] within 0.01 of optimum.
  B. municipality-cluster bootstrap 95% CIs (Sec. 5.2 and Table 5 prose).
  C. quantile-bin vs fixed-bin ensemble comparison (Sec. 3.7): after
     shrinkage qbin within-gain vanishes while overall R2 drops ~0.02.
  D. mun-agg bootstrap: ensemble (agg) vs SIAP against census.

Writes submission_stats.json next to itself (or $SCRATCH if set).
Run:  ~/miniforge3/envs/geo_env/bin/python robustness_lambda_ci_qbin.py
"""
import os, numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P     =  os.path.join(proj, "Data", "predictions")
INEGI =  os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                      "LM2304-CA22-2025-09-29-superficie_ENTREGA")
agland_path =  os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siap_path   =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")

ev   =  pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))
qb   =  pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_qbin_eval.parquet"))
hist =  pd.read_parquet(os.path.join(P, "adc_aef_hist_gb_preds.parquet"))
hist =  hist[hist["year"] == 2022].copy()
hist["adc"] =  hist["adcid"].str.replace("-", "", regex=False)
ev   =  ev.merge(hist[["adc", "yield_pred"]].rename(columns={"yield_pred": "pred_hist"}),
                 on="adc", how="left")
print("qbin eval cols:", list(qb.columns))
ev   =  ev.merge(qb[["adc", "pred"]].rename(columns={"pred": "pred_qbin"}),
                 on="adc", how="left")

siap =  pd.read_stata(siap_path)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
sm   =  siap[(siap["name"] == "Maize") & (siap["year"] == 2022) &
             (~siap["muncode"].str.endswith("000"))]
# season-matched municipal yields
pv   =  sm[sm["growing_season"] == "Spring-Summer"].groupby("muncode").agg(
            q=("q", "sum"), ha=("ha_planted", "sum"))
pv["siap_pv"] =  pv["q"]/pv["ha"]
al   =  sm.groupby("muncode").agg(q=("q", "sum"), ha=("ha_planted", "sum"))
al["siap_all"] =  al["q"]/al["ha"]
ev   =  ev.merge(pv[["siap_pv"]], on="muncode", how="left")
ev   =  ev.merge(al[["siap_all"]], on="muncode", how="left")

def shrink(d, p, lam):
    g =  d.groupby("muncode")[p]
    return g.transform("mean") + lam*(d[p] - g.transform("mean"))

def metrics(d, y, p):
    s =  d[[y, p, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    gm =  s.groupby("muncode")[[y, p]].transform("mean")
    ov =  1 - ((s[y]-s[p])**2).sum()/((s[y]-s[y].mean())**2).sum()
    wn_num =  (((s[p]-gm[p])-(s[y]-gm[y]))**2).sum()
    wn_den =  ((s[y]-gm[y])**2).sum()
    g  =  s.groupby("muncode")[[y, p]].mean()
    bt =  1 - ((g[y]-g[p])**2).sum()/((g[y]-g[y].mean())**2).sum()
    return dict(N=len(s), R2=round(ov, 3), Btw=round(bt, 3),
                Wtn=round(1-wn_num/wn_den, 3))

out = {}

# ── A. lambda sensitivity ────────────────────────────────
for season, ycol in [("PV", "yield_pv"), ("combined", "yield")]:
    d =  ev[[ycol, "pred", "muncode"]].dropna().copy()
    curve = {}
    for lam in np.round(np.arange(0, 1.01, 0.05), 2):
        d["_s"] =  shrink(d, "pred", lam)
        m =  metrics(d, ycol, "_s")
        curve[float(lam)] =  (m["Wtn"], m["R2"])
    best_lam  =  max(curve, key=lambda k: curve[k][0])
    best_wtn  =  curve[best_lam][0]
    near      =  [k for k, v in curve.items() if v[0] >= best_wtn - 0.01]
    out[f"lambda_curve_{season}"] =  {
        "curve": {str(k): v for k, v in curve.items()},
        "best_lam": best_lam, "best_wtn": best_wtn,
        "within_.01_of_max": [min(near), max(near)],
        "at_0.5": curve[0.5], "at_0.6": curve[0.6], "at_0.7": curve[0.7]}

# ── B. bootstrap CIs (PV season, cluster = municipality) ─
LAM_ENS, LAM_HIST =  0.669, 0.608
d =  ev.copy()
d["ens_sh"]  =  shrink(d, "pred", LAM_ENS)
d["hist_sh"] =  shrink(d, "pred_hist", LAM_HIST)

def suff(d, y, p):
    """Per-muni sufficient stats for bootstrap of overall & within R2."""
    s =  d[[y, p, "muncode"]].dropna().copy()
    gm =  s.groupby("muncode")[[y, p]].transform("mean")
    s["sse"]   =  (s[y]-s[p])**2
    s["y1"]    =  s[y]; s["y2"] = s[y]**2
    s["wnum"]  =  ((s[p]-gm[p])-(s[y]-gm[y]))**2
    s["wden"]  =  (s[y]-gm[y])**2
    g =  s.groupby("muncode").agg(n=("y1", "size"), sse=("sse", "sum"),
        sy=("y1", "sum"), sy2=("y2", "sum"), wnum=("wnum", "sum"), wden=("wden", "sum"))
    return g

def boot(g, nrep=2000):
    munis =  g.index.values
    idx   =  rng.integers(0, len(munis), size=(nrep, len(munis)))
    n     =  g["n"].values[idx].sum(1);   sse =  g["sse"].values[idx].sum(1)
    sy    =  g["sy"].values[idx].sum(1);  sy2 =  g["sy2"].values[idx].sum(1)
    wnum  =  g["wnum"].values[idx].sum(1); wden =  g["wden"].values[idx].sum(1)
    sst   =  sy2 - sy**2/n
    return 1 - sse/sst, 1 - wnum/wden      # overall, within draws

def ci(x): return [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]

# same-sample deltas need aligned rows: use rows where both preds exist
common =  d.dropna(subset=["yield_pv", "pred", "pred_hist", "siap_pv"]).copy()
g_ens  =  suff(common, "yield_pv", "ens_sh")
g_hist =  suff(common, "yield_pv", "hist_sh")
g_siap =  suff(common, "yield_pv", "siap_pv")
# identical muni resampling for paired deltas: same rng stream trick — recompute
rng =  np.random.default_rng(7)
o_e, w_e =  boot(g_ens)
rng =  np.random.default_rng(7)
o_h, w_h =  boot(g_hist)
rng =  np.random.default_rng(7)
o_s, w_s =  boot(g_siap)
out["boot_PV_common"] =  {
    "N": int(len(common)), "n_munis": int(common["muncode"].nunique()),
    "ens_shrink_R2": metrics(common, "yield_pv", "ens_sh"),
    "hist_shrink_R2": metrics(common, "yield_pv", "hist_sh"),
    "siap_R2": metrics(common, "yield_pv", "siap_pv"),
    "ens_sh_overall_CI": ci(o_e), "ens_sh_within_CI": ci(w_e),
    "delta_within_ens_minus_hist_CI": ci(w_e - w_h),
    "delta_overall_ens_minus_siap_CI": ci(o_e - o_s),
    "pr_delta_within_pos": float((w_e - w_h > 0).mean()),
    "pr_delta_siap_pos": float((o_e - o_s > 0).mean())}
# raw ens within CI on its full PV sample
g_raw =  suff(ev.dropna(subset=["yield_pv", "pred"]), "yield_pv", "pred")
rng =  np.random.default_rng(7); o_r, w_r =  boot(g_raw)
out["boot_PV_ens_raw_full"] =  {"overall_CI": ci(o_r), "within_CI": ci(w_r)}

# ── C. qbin vs fixed comparison ─────────────────────────
cq =  ev.dropna(subset=["yield", "pred", "pred_qbin"]).copy()
res =  {}
for name, col in [("fixed", "pred"), ("qbin", "pred_qbin")]:
    # cv lambda quickly: use grid max as proxy (report both)
    best = None
    for lam in np.round(np.arange(0, 1.01, 0.05), 2):
        cq["_s"] =  shrink(cq, col, lam)
        m =  metrics(cq, "yield", "_s")
        if best is None or m["Wtn"] > best[1]["Wtn"]: best = (float(lam), m)
    res[name] =  {"raw": metrics(cq, "yield", col),
                  "shrink_lam": best[0], "shrink": best[1]}
    cq["_spv"] =  shrink(cq, col, best[0])
    res[name]["shrink_PV"] =  metrics(cq.dropna(subset=["yield_pv"]), "yield_pv", "_spv")
    res[name]["raw_PV"]    =  metrics(cq.dropna(subset=["yield_pv"]), "yield_pv", col)
out["qbin_vs_fixed_common"] =  {"N": int(len(cq)), **res}

# ── D. mun-agg bootstrap (census benchmark) ─────────────
ca =  pd.read_stata(os.path.join(INEGI, "adc_land_use_ca22_adc07.dta"))
ca =  ca[ca["name"] == "Maize"][["adc", "muncode", "land_input", "vol_output"]].copy()
ag =  pd.read_csv(agland_path); ag["adc"] =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
ca =  ca.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
ca =  ca.merge(ev[["adc", "pred"]], on="adc", how="left")
cm =  ca.groupby("muncode").apply(lambda g: pd.Series({
        "cen_yield": g["vol_output"].sum()/g["land_input"].sum() if g["land_input"].sum() > 0 else np.nan,
        "pred_agg":  np.average(g["pred"].dropna(),
                     weights=g.loc[g["pred"].notna(), "siap_agland_area"].fillna(0)+1e-9)
                     if g["pred"].notna().any() else np.nan}))
cm =  cm.merge(al[["siap_all"]], on="muncode", how="left").dropna()
def r2v(a, b): return 1 - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)
arr =  cm[["cen_yield", "pred_agg", "siap_all"]].values
nrep =  4000; idx =  np.random.default_rng(3).integers(0, len(arr), size=(nrep, len(arr)))
r2_ens  =  np.array([r2v(arr[i, 0], arr[i, 1]) for i in idx])
r2_siap =  np.array([r2v(arr[i, 0], arr[i, 2]) for i in idx])
out["mun_agg_boot"] =  {
    "N_munis": int(len(cm)),
    "R2_ens_agg": round(float(r2v(arr[:, 0], arr[:, 1])), 3),
    "R2_siap":    round(float(r2v(arr[:, 0], arr[:, 2])), 3),
    "ens_CI": ci(r2_ens), "siap_CI": ci(r2_siap),
    "delta_CI": ci(r2_ens - r2_siap),
    "pr_delta_pos": float((r2_ens - r2_siap > 0).mean())}

sp =  os.environ.get("SCRATCH", "/tmp")
with open(os.path.join(sp, "submission_stats.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
