"""
Public-data selection of the AEF Hist Ensemble weight w (companion to 17_rho_irrigation_bound.py).

The ensemble is pred_w = w*bins + (1-w)*pct. Its within-municipality validity rho(w)
is bounded/estimated exactly as for lambda (Sec 3.6): under the irrigation projection
y = beta*irr + v with cov_w(pred, v) >= 0,

    rho_lb(w)  =  beta * c_w(pred_w, irr) * sd_w(irr) / sd_w(y)
    rho_mid(w) =  c_w(pred_w, irr) * (sd_u + sd_v/2) / sd_w(y)      [midpoint point estimate]

Every factor except c_w(pred_w, irr) is constant in w, so the public-data argmax of
rho(w) -- and of the shrunk within-R2 rho(w)^2 -- is argmax_w c_w(pred_w, irr): the
within-municipality correlation of the blended prediction with SIAP irrigated share.
No census, no CIMMYT. The unshrunk projection 2*rho*r - r^2 is also reported.

Inputs : Data/predictions/adc_aef_hist_ens_components.parquet  (pred_bin, pred_pct; w-independent,
         written by train/03_train_rf_gb/2_gb_aef_hist_ensemble.py)
         Data/SIAP_agland/Output/2007_adcs_agland_area.csv       (ADC irrigated share)
         SIAP municipal panel                                     (beta)
         Data/predictions/adc_aef_hist_ens_eval.parquet           (ADC universe only: census-maize ADCs)
Outputs: plots/w_public_sweep.csv, output/w_public_selection.json

Run:  python 18_w_public_selection.py      (mpc_env)
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

home  =  os.path.expanduser("~")
proj  =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P     =  os.path.join(proj, "Data", "predictions")
agp   =  os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siapp =  os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
plots =  os.path.join(proj, "plots")
SD_Y_W  =  1.66                                    # public within-municipality yield dispersion gauge (Sec 3.6)
W_GRID  =  np.round(np.arange(0.0, 1.0001, 0.05), 2)

comp  =  pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_components.parquet"))
comp  =  comp[comp["year"] == 2022].copy()
comp["adc"]  =  comp["adcid"].astype(str).str.replace("-", "", regex=False)
ev    =  pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))[["adc"]].drop_duplicates()
ag    =  pd.read_csv(agp)
ag["adc"]  =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
ag["irr"]  =  np.where(ag["siap_agland_area"] > 0, ag["siap_irrig_area"]/ag["siap_agland_area"], np.nan)
d  =  comp.merge(ev, on="adc", how="inner").merge(ag[["adc", "irr"]].drop_duplicates("adc"), on="adc", how="left")
d  =  d.dropna(subset=["pred_bin", "pred_pct", "irr"])
d  =  d[d.groupby("muncode")["adc"].transform("size") >= 2].reset_index(drop=True)

siap  =  pd.read_stata(siapp)
siap["muncode"]  =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
sv  =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
            siap["year"].between(2017, 2022) & (~siap["muncode"].str.endswith("000"))].copy()
sv["yld"]   =  sv["q"]/sv["ha_planted"]
sv["ish"]   =  sv["ha_planted_irrig"]/sv["ha_planted"]
sv  =  sv[np.isfinite(sv["yld"]) & np.isfinite(sv["ish"]) & (sv["yld"] > 0)]
sv["cell"]  =  sv["muncode"].str[:2] + "_" + sv["year"].astype(str)
yd  =  sv["yld"] - sv.groupby("cell")["yld"].transform("mean")
xd  =  sv["ish"] - sv.groupby("cell")["ish"].transform("mean")
beta  =  float(np.sum(yd*xd)/np.sum(xd*xd))

def wdm(s, g):  return s - s.groupby(g).transform("mean")
g       =  d["muncode"]
irr_d   =  wdm(d["irr"], g)
sd_irr  =  float(irr_d.std())
sd_u    =  beta*sd_irr
sd_v    =  float(np.sqrt(SD_Y_W**2 - sd_u**2))

rows  =  []
for w in W_GRID:
    pdv  =  wdm(w*d["pred_bin"] + (1 - w)*d["pred_pct"], g)
    c    =  float(np.corrcoef(pdv, irr_d)[0, 1])
    r    =  float(pdv.std())/SD_Y_W
    rho_lb, rho_mid  =  beta*c*sd_irr/SD_Y_W, (c*sd_u + (c/2)*sd_v)/SD_Y_W
    rows.append({"w": w, "c_pred_irr": c, "r": r, "rho_lb": rho_lb, "rho_mid": rho_mid,
                 "lambda_star": min(rho_mid/r, 1.0), "proj_wtn_raw": 2*rho_mid*r - r**2, "proj_wtn_shrink": rho_mid**2})
sw  =  pd.DataFrame(rows)
os.makedirs(plots, exist_ok=True)
sw.to_csv(os.path.join(plots, "w_public_sweep.csv"), index=False)

i_c  =  int(sw["c_pred_irr"].idxmax()); i_raw  =  int(sw["proj_wtn_raw"].idxmax())
w_pub  =  float(sw.loc[i_c, "w"])
flat   =  sw.loc[sw["proj_wtn_shrink"] >= sw["proj_wtn_shrink"].max() - 0.005, "w"]
print(f"n_adc = {len(d):,} in {g.nunique():,} municipalities | beta = {beta:.2f} | sd_w(irr) = {sd_irr:.3f}")
print(sw.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\npublic argmax of c_w(pred_w, irr) [= argmax rho(w), rho(w)^2]: w = {w_pub:.2f}  (c = {sw.loc[i_c,'c_pred_irr']:.3f})")
print(f"argmax of unshrunk projection 2*rho*r - r^2:                 w = {sw.loc[i_raw,'w']:.2f}")
print(f"flat region (projected shrunk within-R2 within 0.005 of max): w in [{flat.min():.2f}, {flat.max():.2f}]")
out  =  {"w_public": w_pub, "w_argmax_unshrunk": float(sw.loc[i_raw, "w"]), "flat_lo": float(flat.min()), "flat_hi": float(flat.max()),
         "beta": round(beta, 2), "n_adc": int(len(d))}
out_dir =  os.path.join(os.path.expanduser("~"), "Dropbox", "Projects", "Maize_prediction", "output")
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, "w_public_selection.json"), "w"), indent=1)
print(json.dumps(out))
