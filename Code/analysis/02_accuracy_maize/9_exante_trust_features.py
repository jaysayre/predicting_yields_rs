"""
9_exante_trust_features.py — build the per-municipality EX-ANTE feature table
(exante_trust_index.csv) that the trust-index composite and both ex-ante paper
figures depend on.

Extracted (2026-08-28) from coauthor_extras_exante.ipynb Section B + the
drivers_mun.csv step of coauthor_extras_paper.py, so the paper's Figure 6/7
chain no longer depends on running exploratory notebooks. Reproduces the same
16 ENRICHED features:

  baseline (9): log_n_adc, irrig_share, maize_share, log_mean_ha_adc, sd_pred,
                mean_pred, log_siap_sd, siap_mean_yield, aef_heterogeneity
  new (7):      irrig_share_sd, frac_irrigated, adc_area_cv, maizeland_sd,
                aef_spread, aef_pc1_frac, aef_eff_dim

Also writes within_r2_mun (raw ensemble, >=2-ADC muns) for continuity; note
10_exante_trust_composite.py REPLACES that column with the deployed-Shrink
(lambda=0.74, >=5-ADC) version before validating the index.

Inputs  (~/Dropbox/Projects/Maize_prediction/):
  Data/predictions/adc_aef_hist_ens_eval.parquet   -- ADC yield + ensemble pred
  Data/SIAP/Cleaned/siap_ag_prod_estimation_ca2007.dta  -- SIAP panel 2007-2024
  Data/SIAP_agland/Output/2007_adcs_agland_area.csv     -- ADC agland partition
  Data/alpha_earth/alpha_earth_mex_adcs.parquet         -- AEF embeddings (2022)
Output:
  plots/coauthor_extras_paper/exante_trust_index.csv

Run: ~/miniforge3/envs/ml_cuda/bin/python 9_exante_trust_features.py
"""
import os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
data = os.path.join(proj, "Data")
out  = os.path.join(proj, "plots", "coauthor_extras_paper")
os.makedirs(out, exist_ok=True)

YEAR = 2022

# ── targets + per-mun prediction moments (raw ensemble) ──
ev = pd.read_parquet(os.path.join(data, "predictions", "adc_aef_hist_ens_eval.parquet"))
ev = ev[["muncode", "adc", "yield", "pred"]].dropna(subset=["yield", "pred"]).copy()
ev["muncode"] = ev["muncode"].astype(str).str.zfill(5)
gmean = ev["yield"].mean()
rows = []
for mun, g in ev.groupby("muncode"):
    if len(g) < 2: continue
    ym, pm = g["yield"].mean(), g["pred"].mean()
    ssr  = float(((g["yield"] - g["pred"])**2).sum())
    sstw = float(((g["yield"] - ym)**2).sum())
    sstg = float(((g["yield"] - gmean)**2).sum())
    rows.append({"muncode": mun,
                 "within_r2_mun": 1 - ssr/sstw if sstw > 0 else np.nan,
                 "total_contrib": 1 - ssr/sstg if sstg > 0 else np.nan,
                 "mun_mean_abserr": abs(pm - ym),
                 "sd_pred": g["pred"].std(),
                 "mean_pred": pm})
targets = pd.DataFrame(rows)
print(f"targets: {len(targets):,} municipalities")

# ── SIAP 2022 maize share of planted area ────────────────
siap_path = os.path.join(data, "SIAP", "Cleaned", "siap_ag_prod_estimation_ca2007.dta")
siap = pd.read_stata(siap_path, columns=["year", "muncode", "name", "q", "ha_planted"])
siap["muncode"] = siap["muncode"].astype(str).str.zfill(5)
s22 = siap[siap["year"] == YEAR].copy()
s22["is_maize"] = s22["name"].astype(str).str.lower().eq("maize")
s22["maize_ha"] = np.where(s22["is_maize"], s22["ha_planted"], 0.0)
mun_share = (s22.groupby("muncode", as_index=False)
                .agg(maize_ha=("maize_ha", "sum"), total_ha=("ha_planted", "sum")))
mun_share["maize_share"] = mun_share["maize_ha"] / mun_share["total_ha"]

# ── SIAP cross-year maize yield SD + mean (2007-2024, >=5 yrs) ──
sm = siap[(siap["name"] == "Maize") & siap["year"].between(2007, 2024)].copy()
sm["yld"] = sm["q"] / sm["ha_planted"].replace(0, np.nan)
sm = sm.dropna(subset=["yld"])
sa = sm.groupby("muncode").agg(siap_sd_yield=("yld", "std"),
                               siap_mean_yield=("yld", "mean"),
                               nyr=("year", "nunique")).reset_index()
sa = sa[sa["nyr"] >= 5]

# ── ADC agland partition: irrigation share + within-mun heterogeneity ──
ag = pd.read_csv(os.path.join(data, "SIAP_agland", "Output", "2007_adcs_agland_area.csv"))
ag["muncode"] = ag["adcid"].astype(str).str.replace("-", "", regex=False).str[:5]
mun_irrig = (ag.groupby("muncode", as_index=False)
               .agg(siap_irrig_area=("siap_irrig_area", "sum"),
                    siap_agland_area=("siap_agland_area", "sum")))
mun_irrig["irrig_share"] = (mun_irrig["siap_irrig_area"] /
                            mun_irrig["siap_agland_area"].replace(0, np.nan))
ag["irr_sh"]   = ag["siap_irrig_area"] / ag["siap_agland_area"].replace(0, np.nan)
ag["maize_sh"] = ag["maize_land"] / ag["siap_agland_area"].replace(0, np.nan)
het = ag.groupby("muncode").agg(
    irrig_share_sd=("irr_sh", "std"),
    frac_irrigated=("irr_sh", lambda s: float((s > 0.05).mean())),
    adc_area_sd=("adc_area", "std"), adc_area_mean=("adc_area", "mean"),
    n_adc_geo=("adc_area", "size"), mean_ha_adc=("adc_area", "mean"),
    maizeland_sd=("maize_sh", "std")).reset_index()
het["adc_area_cv"] = het["adc_area_sd"] / het["adc_area_mean"].replace(0, np.nan)

# ── AEF within-mun spread metrics (2022) ─────────────────
print("loading AEF embeddings (2022) + spread metrics (~1 min)")
t0 = time.time()
aef = pd.read_parquet(os.path.join(data, "alpha_earth", "alpha_earth_mex_adcs.parquet"),
                      filters=[("year", "==", YEAR)])
acols = [f"A{i:02d}" for i in range(64)]
aef["muncode"] = aef["adcid"].astype(str).str.replace("-", "", regex=False).str[:5]
het_rows = []
for mun, g in aef.groupby("muncode"):
    X = g[acols].to_numpy(dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < 3:
        het_rows.append({"muncode": mun, "aef_heterogeneity": np.nan, "aef_spread": np.nan,
                         "aef_pc1_frac": np.nan, "aef_eff_dim": np.nan}); continue
    perdim_var = X.var(axis=0)
    tr = float(perdim_var.sum())
    pc1_frac = np.nan; eff_dim = np.nan
    try:
        C = np.cov(X, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        evals = np.linalg.eigvalsh(C); evals = evals[evals > 1e-12]
        if evals.size and evals.sum() > 0:
            pc1_frac = float(evals.max() / evals.sum())
            eff_dim  = float((evals.sum()**2) / np.sum(evals**2))
    except np.linalg.LinAlgError:
        pass
    het_rows.append({"muncode": mun,
                     "aef_heterogeneity": float(np.sqrt(perdim_var).mean()),
                     "aef_spread": float(np.sqrt(tr)) if tr > 0 else np.nan,
                     "aef_pc1_frac": pc1_frac, "aef_eff_dim": eff_dim})
aef_het = pd.DataFrame(het_rows)
print(f"  {len(aef_het):,} municipalities in {(time.time()-t0)/60:.1f} min")

# ── assemble + transforms ────────────────────────────────
D = (targets.merge(mun_share[["muncode", "maize_share"]], on="muncode", how="left")
            .merge(mun_irrig[["muncode", "irrig_share"]], on="muncode", how="left")
            .merge(sa[["muncode", "siap_sd_yield", "siap_mean_yield"]], on="muncode", how="left")
            .merge(het[["muncode", "irrig_share_sd", "frac_irrigated", "adc_area_cv",
                        "maizeland_sd", "n_adc_geo", "mean_ha_adc"]], on="muncode", how="left")
            .merge(aef_het, on="muncode", how="left"))
D["log_n_adc"]       = np.log(D["n_adc_geo"].clip(lower=1))
D["log_mean_ha_adc"] = np.log(D["mean_ha_adc"].clip(lower=1e-3))
D["log_siap_sd"]     = np.log(D["siap_sd_yield"].clip(lower=1e-3))

ENRICHED = ["log_n_adc", "irrig_share", "maize_share", "log_mean_ha_adc", "sd_pred",
            "mean_pred", "log_siap_sd", "siap_mean_yield", "aef_heterogeneity",
            "irrig_share_sd", "frac_irrigated", "adc_area_cv", "maizeland_sd",
            "aef_spread", "aef_pc1_frac", "aef_eff_dim"]
D = D.dropna(subset=ENRICHED + ["within_r2_mun"]).copy()
print(f"feature frame: {len(D):,} municipalities x {len(ENRICHED)} features")

out_csv = os.path.join(out, "exante_trust_index.csv")
D[["muncode", "within_r2_mun"] + ENRICHED].to_csv(out_csv, index=False)
print(f"wrote {out_csv}")
