"""
1_other_crops_adc_table.py — assemble Table A3 (accuracy_other_crops_adc_2022.tex).

Rebuilds, for each non-maize crop, the AEF Hist / AEF Hist Corr. rows (the
bins-only model) and the AEF Hist Ens. Raw / Corr. / Shrink rows from the
per-crop ADC prediction parquets persisted by
train/03_train_rf_gb/4_gb_aef_hist_ensemble_other_crops.py, and (2026-09-04) the
AEF mean / AEF mean Corr. rows from the means-alone RF parquets written by
train/03_train_rf_gb/1_rf_yield_prediction.py (mexico_multi_crop, IMPROVED=False),
inserting them into the existing table template (which retains only the DGSIAP
rows as frozen text). The additive correction replicates the retired
dep/1_accuracy_other_crops_2022.ipynb: land-weighted municipal average of the
prediction, differenced against the 2022 DGSIAP municipal yield, clipped at 0.
Owning the .tex here keeps every paper table in the analysis stage; the trainer
only persists predictions. Idempotent: stale AEF rows are dropped before fresh
ones are inserted ahead of each crop's DGSIAP row.

Inputs  (~/Dropbox/Projects/Maize_prediction/):
  Data/predictions/adc_aef_hist_ens_preds_{sorghum,sugar,wheat,avocados}.parquet
  Data/predictions/adc_alpha_earth_preds_{sorghum,sugar,wheat,avocados}.parquet
  Data/SIAP/Cleaned/siap_ag_prod_estimation_ca2007.dta
  Data/INEGI/MD_lab_outputs/LM2304-CA22-2025-09-29-superficie_ENTREGA/adc_land_use_ca22_adc07.dta
  tables/accuracy_other_crops_adc_2022.tex   (template, rows replaced in place)
Output:
  tables/accuracy_other_crops_adc_2022.tex

Run: ~/miniforge3/envs/ml_cuda/bin/python 1_other_crops_adc_table.py
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
pred_dir  =  os.path.join(proj_dir, "Data", "predictions")
ca22_path =  os.path.join(proj_dir, "Data", "INEGI", "MD_lab_outputs",
                          "LM2304-CA22-2025-09-29-superficie_ENTREGA",
                          "adc_land_use_ca22_adc07.dta")
tex_path  =  os.path.join(proj_dir, "tables", "accuracy_other_crops_adc_2022.tex")
siap_path =  os.path.join(proj_dir, "Data", "SIAP", "Cleaned",
                          "siap_ag_prod_estimation_ca2007.dta")

CROPS = [("Sorghum", "Sorghum"), ("Sugar", "Sugarcane"),
         ("Wheat", "Wheat"), ("Avocados", "Avocados")]   # (census/parquet name, table display name)


def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = np.array(y[m]), np.array(yh[m])
    if len(y) < 2: return np.nan
    st = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y - yh)**2)/st if st > 0 else np.nan

def within_r2(d, y, p, gc="muncode"):
    s = d[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    c = s.groupby(gc).size(); s = s[s[gc].isin(c[c >= 2].index)]
    if len(s) == 0: return np.nan
    gm = s.groupby(gc)[[y, p]].transform("mean")
    return r2(s[y] - gm[y], s[p] - gm[p])

def between_r2(d, y, p, gc="muncode"):
    s = d[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(gc)[[y, p]].mean(); return r2(g[y], g[p])

def eval_row(d, ycol, pcol):
    s = d[[ycol, pcol, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    return (len(s), r2(s[ycol], s[pcol]), between_r2(s, ycol, pcol),
            within_r2(s, ycol, pcol),
            np.sqrt(np.mean((s[ycol].values - s[pcol].values)**2)))

def fmt(v):
    return "---" if not np.isfinite(v) else (f"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}")


ca = pd.read_stata(ca22_path)
siap_yields = pd.read_stata(siap_path)
siap_yields = siap_yields[siap_yields["year"] == 2022].copy()
siap_yields["yield_siap"] = siap_yields["q"] / siap_yields["ha_planted"]
siap_yields["muncode"]    = siap_yields["muncode"].apply(lambda x: str(int(x)).zfill(5))
new_rows = {}
new_bin_rows = {}
new_mean_rows = {}
for crop, disp in CROPS:
    gt = ca[ca["name"] == crop][["adc", "yield"]].copy()
    gt["adc"] = gt["adc"].astype(str)
    pq = os.path.join(pred_dir, f"adc_aef_hist_ens_preds_{crop.lower()}.parquet")
    pr = pd.read_parquet(pq); pr["adc"] = pr["adc"].astype(str)
    df = pr.merge(gt, on="adc", how="left")
    rows = []
    for suffix, lbl in [("", "AEF Hist Ens."), ("_corr", "AEF Hist Ens. Corr."),
                        ("_shrink", "AEF Hist Ens. Shrink")]:
        n, ov, b, w, rm = eval_row(df, "yield", f"pred_{crop}{suffix}")
        rows.append(f"{disp} & {lbl} & {n:,} & {fmt(ov)} & {fmt(b)} & {fmt(w)} & {fmt(rm)} \\\\\n")
        print(rows[-1].strip())
    new_rows[disp] = rows

    # ── AEF Hist rows (2026-09-17: "AEF Hist" is the bins-only model, i.e. the
    #    pred_bin component of the ensemble — 64 dims x 8 histogram-bin shares,
    #    K=5 subsampled draws of N=2 pixels/dim. Raw and additively corrected
    #    columns both come from the trainer's parquet, built exactly like the
    #    ensemble's _corr column.) ────────────────────────────────────────────
    bin_rows = []
    for suffix, lbl in [("_bin", "AEF Hist"), ("_bin_corr", "AEF Hist Corr.")]:
        n, ov, b, w, rm = eval_row(df, "yield", f"pred_{crop}{suffix}")
        bin_rows.append(f"{disp} & {lbl} & {n:,} & {fmt(ov)} & {fmt(b)} & {fmt(w)} & {fmt(rm)} \\\\\n")
        print(bin_rows[-1].strip())
    new_bin_rows[disp] = bin_rows

    # ── AEF mean rows (means-alone RF; correction as in the retired notebook) ─
    mp  = os.path.join(pred_dir, f"adc_alpha_earth_preds_{crop.lower()}.parquet")
    mpr = pd.read_parquet(mp)
    mpr = mpr[mpr["year"] == 2022].copy()
    mpr["adc"] = mpr["adcid"].astype(str).str.replace("-", "", regex=False)
    pcol = f"yield_pred_{crop.lower()}"
    gm  = ca[ca["name"] == crop][["adc", "muncode", "land_input", "yield"]].copy()
    gm["adc"]     = gm["adc"].astype(str)
    gm["muncode"] = gm["muncode"].astype(str).str.zfill(5)
    dm  = gm.merge(mpr[["adc", pcol]], on="adc", how="left")
    dm  = dm.merge(siap_yields[siap_yields["name"] == crop][["muncode", "yield_siap"]],
                   on="muncode", how="left")
    dm["_wQ"] = dm[pcol] * dm["land_input"]
    dm["_wA"] = np.where(np.isfinite(dm[pcol]), dm["land_input"], 0.0)
    mc = dm.groupby("muncode")[["_wQ", "_wA"]].sum().reset_index()
    mc["pred_mun_avg"] = np.where(mc["_wA"] > 0, mc["_wQ"] / mc["_wA"], np.nan)
    mc = mc.merge(siap_yields[siap_yields["name"] == crop][["muncode", "yield_siap"]],
                  on="muncode", how="left")
    mc["diff"] = mc["pred_mun_avg"] - mc["yield_siap"]
    dm = dm.merge(mc[["muncode", "diff"]], on="muncode", how="left")
    ccol = pcol + "_corr"
    dm[ccol] = (dm[pcol] - dm["diff"]).clip(lower=0)
    dm.loc[dm[pcol].isna(), ccol] = np.nan
    mean_rows = []
    for col, lbl in [(pcol, "AEF mean"), (ccol, "AEF mean Corr.")]:
        n, ov, b, w, rm = eval_row(dm, "yield", col)
        mean_rows.append(f"{disp} & {lbl} & {n:,} & {fmt(ov)} & {fmt(b)} & {fmt(w)} & {fmt(rm)} \\\\\n")
        print(mean_rows[-1].strip())
    new_mean_rows[disp] = mean_rows

with open(tex_path) as f:
    old_lines = f.readlines()

# Idempotent rebuild: drop every stale AEF row and re-insert the full block
# (AEF mean, AEF mean Corr., AEF Hist, AEF Hist Corr., AEF Hist Ens.,
#  AEF Hist Ens. Corr., AEF Hist Ens. Shrink) ahead of each crop's DGSIAP row,
# which is the only model row kept as template text.
out_lines = []
for line in old_lines:
    if "& AEF Hist" in line or "& AEF mean" in line:
        continue
    for _, disp in CROPS:
        if f"{disp} & DGSIAP" in line:
            out_lines.extend(new_mean_rows[disp])
            out_lines.extend(new_bin_rows[disp])
            out_lines.extend(new_rows[disp])
            break
    out_lines.append(line)

with open(tex_path, "w") as f:
    f.writelines(out_lines)
print(f"\nwrote {tex_path}")
