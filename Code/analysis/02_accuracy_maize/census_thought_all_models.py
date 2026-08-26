"""
Census thought-experiment table (Table \ref{tab:census_thought}) for ALL models.

For each model's ADC-level 2022 predictions, evaluate vs the INEGI 2022 census
(combined season) under three additive ex-post corrections:
  Raw              : no correction
  Corr. -> SIAP    : shift each municipality's predictions to the SIAP mun mean
  Corr. -> Census  : shift each municipality's predictions to the INEGI-census mun mean
The Census correction isolates the effect of training/anchoring on cleaner
(census) municipality yields rather than survey-based SIAP yields.

Models: RS, 3-Period Hist. (Landsat-derived); AEF RF, Agg-NN, AEF Hist,
AEF Hist Ens. (AEF-derived). The Agg-NN (Census-trained) panel comes from the
GPU retrain in census_thought_experiment.py and is appended verbatim.

Run:  ~/miniforge3/envs/geo_env/bin/python census_thought_all_models.py
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

home_dir  = os.path.expanduser("~")
proj_dir  = os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
pred_dir  = os.path.join(proj_dir, "Data", "predictions")
table_dir = os.path.join(proj_dir, "tables")
siap_path = os.path.join(home_dir, "Dropbox", "Projects", "The Promise of Crop Substitution",
                         "data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
os.makedirs(table_dir, exist_ok=True)

# ── metrics ─────────────────────────────────────────────
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    if len(y) < 2: return np.nan
    st = np.sum((y - y.mean())**2); return 1 - np.sum((y - yh)**2)/st if st > 0 else np.nan

def within_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(gc).size(); s = s[s[gc].isin(c[c >= 2].index)]
    if len(s) == 0: return np.nan
    gm = s.groupby(gc)[[y, p]].transform("mean"); return r2(s[y]-gm[y], s[p]-gm[p])

def between_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(gc)[[y, p]].mean(); return r2(g[y], g[p])

def metrics(df, p):
    s = df[["yield", p, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    return dict(N=len(s), R2=r2(s["yield"], s[p]), Btw=between_r2(s, "yield", p),
                Wtn=within_r2(s, "yield", p),
                RMSE=np.sqrt(np.mean((s["yield"].values - s[p].values)**2)))

def correct_to(df, pcol, target_col):
    """Additive correction: shift each mun's preds (area-wtd) to target mun mean."""
    d = df[df[pcol].notna()].copy()
    d["wv"] = d[pcol] * d["w"]
    agg = d.groupby("muncode").agg(wv=("wv", "sum"), w=("w", "sum"),
                                   tgt=(target_col, "first")).reset_index()
    agg["diff"] = agg["wv"]/agg["w"] - agg["tgt"]
    out = df.merge(agg[["muncode", "diff"]], on="muncode", how="left")
    return (out[pcol] - out["diff"]).clip(lower=0)

# ── ground truth + per-mun targets ─────────────────────
gt = pd.read_parquet(os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet"))
gt = gt[["adc", "muncode", "yield", "land_input", "pred"]].dropna(subset=["yield"]).copy()
# ex-ante agricultural-land proxy for the correction weight (not census land_input)
_ag = pd.read_csv(os.path.join(proj_dir, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv"))
_ag["adc"] = _ag["adc07"].astype(str).str.replace("-", "", regex=False)
gt = gt.merge(_ag[["adc", "siap_agland_area"]], on="adc", how="left")
gt["w"] = np.where(gt["siap_agland_area"] > 0, gt["siap_agland_area"],
                   np.where(gt["land_input"] > 0, gt["land_input"], 1.0))
gt = gt.rename(columns={"pred": "AEF Hist Ens."})

# Census mun yield = area-weighted mean of INEGI ADC yields
# census municipal yield = census's own aggregate (census planted-area weighted);
# this is the benchmark target, so it legitimately uses land_input.
cm = gt.assign(wy=gt["yield"]*gt["land_input"]).groupby("muncode").agg(
    wy=("wy", "sum"), la=("land_input", "sum")).reset_index()
cm["census_mun"] = cm["wy"]/cm["la"]
gt = gt.merge(cm[["muncode", "census_mun"]], on="muncode", how="left")

# SIAP mun yield (combined, maize 2022)
siap = pd.read_stata(siap_path)
siap["muncode"] = siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
s22 = siap[(siap["name"] == "Maize") & (siap["year"] == 2022)]
s22 = s22[~s22["muncode"].str.endswith("000")].groupby("muncode").agg(
    q=("q", "sum"), ha=("ha_planted", "sum")).reset_index()
s22["siap_mun"] = s22["q"]/s22["ha"]
gt = gt.merge(s22[["muncode", "siap_mun"]], on="muncode", how="left")

# ── merge each model's 2022 ADC predictions ─────────────
MODELS = [  # (panel label, file, col)
    # 2026-08-16: the two unmasked h3 harmonic variants (NDVI Hist. / NDVI Q-Hist.)
    # are folded into the single cropland-masked aefn2 baseline, matching the
    # accuracy and mun-level tables.
    ("NDVI (masked)", "adc_aefn2_masked_preds.parquet",        "pred"),
    ("AEF mean",         "adc_alpha_earth_preds.csv",          "yield_pred"),
    ("Agg-NN",         "adc_mlp_yield_preds.csv",            "pred_yield"),
    ("AEF Hist",       "adc_aef_hist_gb_preds.parquet",      "yield_pred"),
]
for label, f, col in MODELS:
    df = pd.read_parquet(os.path.join(pred_dir, f)) if f.endswith("parquet") else pd.read_csv(os.path.join(pred_dir, f))
    if "year" in df.columns: df = df[df["year"] == 2022]
    key = "adc" if "adc" in df.columns else "adcid"
    df["adc"] = df[key].astype(str).str.replace("-", "", regex=False)
    df = df[["adc", col]].rename(columns={col: label}).dropna().drop_duplicates("adc")
    gt = gt.merge(df, on="adc", how="left")

ORDER = ["NDVI (masked)", "AEF mean", "Agg-NN", "AEF Hist", "AEF Hist Ens."]

def f3(v):
    if v is None or not np.isfinite(v): return "---"
    return (f"{v:.3f}").replace("-", "$-$") if v < 0 else f"{v:.3f}"

# ── build panels ────────────────────────────────────────
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Thought experiment: ADC-level yield prediction accuracy when correcting "
     r"toward INEGI census municipality yields vs.\ SIAP municipality yields, for all models. "
     r"All evaluated against the INEGI 2022 census at the ADC level (combined season).}",
     r"\label{tab:census_thought}", r"\begin{tabular}{lrrrrr}", r"\hline",
     r"Model & $N$ & $R^2$ & Between $R^2$ & Within $R^2$ & RMSE \\", r"\hline"]

print(f"  {'Model / correction':<28s} {'N':>7s} {'R2':>6s} {'Btw':>6s} {'Wtn':>6s} {'RMSE':>6s}")
for label in ORDER:
    L.append(rf"\multicolumn{{6}}{{l}}{{\textit{{Panel: {label}}}}} \\")
    L.append(r"\hline")
    sub = gt[["muncode", "yield", "w", "census_mun", "siap_mun", label]].copy()
    sub["raw"]  = sub[label]
    sub["cs"]   = correct_to(sub, label, "siap_mun")
    sub["cc"]   = correct_to(sub, label, "census_mun")
    for tag, pcol in [("Raw", "raw"),
                      (r"Corr.\ ($\rightarrow$ SIAP)", "cs"),
                      (r"Corr.\ ($\rightarrow$ Census)", "cc")]:
        m = metrics(sub, pcol)
        L.append(f"{tag} & {m['N']:,} & {f3(m['R2'])} & {f3(m['Btw'])} & {f3(m['Wtn'])} & {f3(m['RMSE'])} \\\\")
        print(f"  {label+' '+tag:<28s} {m['N']:>7,} {m['R2']:>6.3f} {m['Btw']:>6.3f} {m['Wtn']:>6.3f} {m['RMSE']:>6.3f}")
    L.append(r"\hline")

# Agg-NN (Census-trained) — from the GPU retrain in census_thought_experiment.py
L.append(r"\multicolumn{6}{l}{\textit{Panel: Agg-NN (Census-trained)}} \\")
L.append(r"\hline")
L.append(r"Raw (trained on census) & 72,581 & 0.665 & 0.938 & $-$0.238 & 1.657 \\")
L.append(r"\hline")
L += [r"\end{tabular}", r"\end{table}", ""]

out = os.path.join(table_dir, "census_thought_experiment_2022.tex")
with open(out, "w") as fh: fh.write("\n".join(L))
print(f"\nWrote {out}")
