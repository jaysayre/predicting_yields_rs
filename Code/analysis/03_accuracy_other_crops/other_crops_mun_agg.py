"""
Rebuild accuracy_other_crops_mun_2022.tex (Table \ref{tab:mun_other_crops})
with EX-ANTE agricultural-land weights.

Replaces the legacy notebook table (2026-02) that aggregated ADC predictions
with census planted-area weights -- census information must not enter the
aggregation (2026-08-28). Aggregation now mirrors mun_survey_improvement.py:
each ADC's AEF mean prediction (adc_alpha_earth_preds_{crop}) is weighted by
its 2007 agricultural-land area. Panel A scores aggregated predictions and the
SIAP municipal average against the census municipal yield; Panel B scores
aggregated predictions and the aggregated census against SIAP.

Run:  ~/miniforge3/envs/geo_env/bin/python other_crops_mun_agg.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P     =  os.path.join(proj, "Data", "predictions")
INEGI =  os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                      "LM2304-CA22-2025-09-29-superficie_ENTREGA")
agland_path =  os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siap_path   =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
overleaf    =  os.path.join(home, "Dropbox", "Overleaf", "Predicting Yields at Scale using RS")
table_dir   =  os.path.join(proj, "tables")

CROPS =  [("Sorghum", "sorghum"), ("Sugar", "sugar"), ("Wheat", "wheat"), ("Avocados", "avocados")]
DISP  =  {"Sugar": "Sugarcane"}

def r2(a, b):
    a, b =  np.asarray(a), np.asarray(b)
    return 1 - np.sum((a-b)**2)/np.sum((a-a.mean())**2)
def rmse(a, b):
    a, b =  np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a-b)**2))
def fmt(v):
    return f"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}"

ca   =  pd.read_stata(os.path.join(INEGI, "adc_land_use_ca22_adc07.dta"))
ag   =  pd.read_csv(agland_path)
ag["adc"] =  ag["adc07"].astype(str).str.replace("-", "", regex=False)
siap =  pd.read_stata(siap_path)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))

rowsA, rowsB =  [], []
for crop, tag in CROPS:
    cc =  ca[ca["name"] == crop][["adc", "muncode", "land_input", "vol_output"]].copy()
    pr =  pd.read_parquet(os.path.join(P, f"adc_alpha_earth_preds_{tag}.parquet"))
    pr =  pr[pr["year"] == 2022].copy()
    pcol =  [c for c in pr.columns if c.startswith("yield_pred")][0]
    pr["adc"] =  pr["adcid"].astype(str).str.replace("-", "", regex=False)
    cc =  cc.merge(pr[["adc", pcol]], on="adc", how="left")
    cc =  cc.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
    cc["w"] =  np.where(cc["siap_agland_area"] > 0, cc["siap_agland_area"], 1.0)

    def agg(g):
        cen  =  g["vol_output"].sum()/g["land_input"].sum() if g["land_input"].sum() > 0 else np.nan
        pv   =  g.dropna(subset=[pcol])
        pred =  np.average(pv[pcol], weights=pv["w"]) if len(pv) else np.nan
        return pd.Series({"cen_yield": cen, "pred_agg": pred})
    cm =  cc.groupby("muncode").apply(agg).reset_index()

    sm =  siap[(siap["name"] == crop) & (siap["year"] == 2022) &
               (~siap["muncode"].str.endswith("000"))]
    sm =  sm.groupby("muncode").agg(q=("q", "sum"), ha=("ha_planted", "sum")).reset_index()
    sm["siap_yield"] =  sm["q"]/sm["ha"]
    cm =  cm.merge(sm[["muncode", "siap_yield"]], on="muncode", how="left")

    disp =  DISP.get(crop, crop)
    a =  cm.dropna(subset=["cen_yield", "pred_agg"])
    rowsA.append(f"{disp} & AEF mean (agg.) & {len(a):,} & {fmt(r2(a['cen_yield'], a['pred_agg']))} & {fmt(rmse(a['cen_yield'], a['pred_agg']))} \\\\")
    s =  cm.dropna(subset=["cen_yield", "siap_yield"])
    rowsA.append(f"{disp} & SIAP Mun.\\ Avg. & {len(s):,} & {fmt(r2(s['cen_yield'], s['siap_yield']))} & {fmt(rmse(s['cen_yield'], s['siap_yield']))} \\\\")
    rowsA.append(r"\hline")
    b =  cm.dropna(subset=["siap_yield", "pred_agg"])
    rowsB.append(f"{disp} & AEF mean (agg.) & {len(b):,} & {fmt(r2(b['siap_yield'], b['pred_agg']))} & {fmt(rmse(b['siap_yield'], b['pred_agg']))} \\\\")
    c =  cm.dropna(subset=["siap_yield", "cen_yield"])
    rowsB.append(f"{disp} & INEGI Census (agg.) & {len(c):,} & {fmt(r2(c['siap_yield'], c['cen_yield']))} & {fmt(rmse(c['siap_yield'], c['cen_yield']))} \\\\")
    rowsB.append(r"\hline")
    print(f"{crop:10s} vsCensus AEF={r2(a['cen_yield'], a['pred_agg']):.3f} SIAP={r2(s['cen_yield'], s['siap_yield']):.3f} | vsSIAP AEF={r2(b['siap_yield'], b['pred_agg']):.3f} Census={r2(c['siap_yield'], c['cen_yield']):.3f}")

L =  [r"\begin{table}[!htbp]", r"\centering",
      r"\caption{Municipality-level yield prediction results for non-maize crops, 2022. "
      r"ADC-level AEF mean predictions are aggregated to the municipality level weighting "
      r"each ADC by its \emph{ex-ante} agricultural-land area, as in "
      r"Table~\ref{tab:mun_agg_results}; no census information enters the aggregation. "
      r"RMSE in t/ha.}",
      r"\label{tab:mun_other_crops}", r"\begin{tabular}{llrrr}", r"\hline",
      r"\multicolumn{5}{l}{\textit{Panel A: vs.\ INEGI Census}} \\", r"\hline",
      r"Crop & Model & $N$ & $R^2$ & RMSE \\", r"\hline"] + rowsA + [
      r"\multicolumn{5}{l}{\textit{Panel B: vs.\ SIAP Municipal Estimates}} \\", r"\hline",
      r"Crop & Model & $N$ & $R^2$ & RMSE \\", r"\hline"] + rowsB + [
      r"\end{tabular}", r"\end{table}", ""]

for d in [table_dir, overleaf]:
    out =  os.path.join(d, "accuracy_other_crops_mun_2022.tex")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("Wrote", out)
