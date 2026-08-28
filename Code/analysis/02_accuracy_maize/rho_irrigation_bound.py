"""
Public-data lower bound on rho (and lambda*) from irrigation (paper Sec 3.6).

Within a municipality, irrigation is the one yield determinant that public data
measure directly: SIAP's Frontera Agricola gives each ADC's irrigated share of
agricultural land, and the SIAP municipal panel prices its yield gradient
(regressing municipal P-V maize yield on municipal irrigated share within
state-year, 2017-2022). Projecting true ADC yields on irrigation,
y = beta*irr + v, and assuming only cov_w(pred, v) >= 0 (predictions do not
anti-correlate with the non-irrigation component of yields):

    rho     >=  beta * c_w(pred, irr) * sd_w(irr) / sd_w(y)
    lambda* >=  beta * c_w(pred, irr) * sd_w(irr) / sd_w(pred)

The lambda bound is fully free of the unobserved within-municipality yield
dispersion sd_w(y) (it cancels against r = sd_w(pred)/sd_w(y)); the rho bound
uses the public gauge sd_w(y) ~= 1.66 t/ha (Sec 3.6; census value 1.53 would
tighten it). No census, no CIMMYT. Municipality-cluster bootstrap CI included.

Run:  ~/miniforge3/envs/geo_env/bin/python rho_irrigation_bound.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

home =  os.path.expanduser("~")
proj =  os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P    =  os.path.join(proj, "Data", "predictions")
agp  =  os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siapp = os.path.join(proj, "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
SD_Y_W =  1.66   # public within-municipality yield dispersion gauge (Sec 3.6)

ev =  pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))[["adc", "muncode", "pred"]]
ev =  ev[ev["pred"].notna()].copy()          # ensemble-eligible sample
ag =  pd.read_csv(agp)
ag["adc"] =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
ag["irr"] =  np.where(ag["siap_agland_area"] > 0, ag["siap_irrig_area"]/ag["siap_agland_area"], np.nan)
ev =  ev.merge(ag[["adc", "irr"]].drop_duplicates("adc"), on="adc", how="left")
ev =  ev.dropna(subset=["pred", "irr"])
cnt =  ev.groupby("muncode")["pred"].transform("size")
ev  =  ev[cnt >= 2].reset_index(drop=True)

siap =  pd.read_stata(siapp)
siap["muncode"] =  siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
sv =  siap[(siap["name"] == "Maize") & (siap["growing_season"] == "Spring-Summer") &
           (siap["year"].between(2017, 2022)) & (~siap["muncode"].str.endswith("000"))].copy()
sv["yld"]  =  sv["q"]/sv["ha_planted"]
sv["ish"]  =  sv["ha_planted_irrig"]/sv["ha_planted"]
sv =  sv[np.isfinite(sv["yld"]) & np.isfinite(sv["ish"]) & (sv["yld"] > 0)]
sv["cell"] =  sv["muncode"].str[:2] + "_" + sv["year"].astype(str)

def beta_hat(d):
    yd =  d["yld"] - d.groupby("cell")["yld"].transform("mean")
    xd =  d["ish"] - d.groupby("cell")["ish"].transform("mean")
    return float(np.sum(yd*xd)/np.sum(xd*xd))

def bound_stats(adc):
    pd_ =  adc["pred"] - adc.groupby("muncode")["pred"].transform("mean")
    ir_ =  adc["irr"]  - adc.groupby("muncode")["irr"].transform("mean")
    c   =  float(np.sum(pd_*ir_)/np.sqrt(np.sum(pd_**2)*np.sum(ir_**2)))
    return c, float(np.std(ir_)), float(np.std(pd_))

beta =  beta_hat(sv)
c, sd_irr, sd_pred =  bound_stats(ev)
rho_lb =  beta*c*sd_irr/SD_Y_W
lam_lb =  beta*c*sd_irr/sd_pred
print(f"beta = {beta:.2f} t/ha | c_w(pred,irr) = {c:.3f} | sd_w(irr) = {sd_irr:.3f} | sd_w(pred) = {sd_pred:.2f}")
print(f"rho    >= {rho_lb:.3f}   (with public sd_y gauge {SD_Y_W})")
print(f"lambda >= {lam_lb:.3f}   (sd_y-free)")

# municipality-cluster bootstrap over both the ADC frame and the SIAP panel
rng =  np.random.default_rng(42)
adc_groups =  {m: g for m, g in ev.groupby("muncode")}
adc_keys   =  list(adc_groups)
siap_groups =  {m: g for m, g in sv.groupby("muncode")}
siap_keys   =  list(siap_groups)
draws =  []
for _ in range(500):
    bs_adc  =  pd.concat([adc_groups[k] for k in rng.choice(adc_keys, len(adc_keys))])
    bs_siap =  pd.concat([siap_groups[k] for k in rng.choice(siap_keys, len(siap_keys))])
    b  =  beta_hat(bs_siap)
    cc, si, sp =  bound_stats(bs_adc)
    draws.append((b*cc*si/SD_Y_W, b*cc*si/sp))
draws =  np.array(draws)
rl, rh =  np.percentile(draws[:, 0], [2.5, 97.5])
ll, lh =  np.percentile(draws[:, 1], [2.5, 97.5])
print(f"bootstrap 95% CI: rho_lb [{rl:.3f}, {rh:.3f}]   lambda_lb [{ll:.3f}, {lh:.3f}]")

# point estimate under the labeled channel assumption (Sec 3.6): the model's
# validity on the unmeasured within-mun channel is half the measured irrigation
# channel's (midpoint of the polar cases rho_v = 0 and rho_v = rho_u)
sd_u  =  beta*sd_irr
sd_v  =  float(np.sqrt(SD_Y_W**2 - sd_u**2))
rho_u =  c
rho_mid =  (rho_u*sd_u + (rho_u/2)*sd_v)/SD_Y_W
print(f"midpoint point estimate: rho_hat = {rho_mid:.3f}  lambda_hat = {rho_mid/0.655:.3f}")

out =  {"beta": round(beta, 2), "c_pred_irr": round(c, 3), "sd_irr": round(sd_irr, 3),
        "rho_midpoint": round(rho_mid, 3), "lambda_midpoint": round(rho_mid/0.655, 3),
        "sd_pred": round(sd_pred, 2), "rho_lb": round(rho_lb, 3), "lambda_lb": round(lam_lb, 3),
        "rho_lb_ci": [round(rl, 3), round(rh, 3)], "lambda_lb_ci": [round(ll, 3), round(lh, 3)],
        "n_adc": int(len(ev))}
sp_ =  os.environ.get("SCRATCH", "/tmp")
json.dump(out, open(os.path.join(sp_, "rho_irrigation_bound.json"), "w"), indent=1)
print(json.dumps(out))
