"""
1_other_crops_adc_table.py — assemble Table A3 (accuracy_other_crops_adc_2022.tex).

Rebuilds the AEF Hist Ens. Raw / Corr. / Shrink rows for each non-maize crop
from the per-crop ADC prediction parquets persisted by
train/03_train_rf_gb/gb_aef_hist_ensemble_other_crops.py, and inserts them into
the existing table template (which carries the AEF mean / AEF Hist / SIAP rows).
Owning the .tex here keeps every paper table in the analysis stage; the trainer
only persists predictions. Idempotent: stale "AEF Hist Ens" rows are dropped
before fresh ones are inserted ahead of each crop's SIAP row.

Inputs  (~/Dropbox/Projects/Maize_prediction/):
  Data/predictions/adc_aef_hist_ens_preds_{sorghum,sugar,wheat,avocados}.parquet
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
new_rows = {}
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

with open(tex_path) as f:
    old_lines = f.readlines()

out_lines = []
for line in old_lines:
    if "AEF Hist Ens" in line:        # drop stale ensemble rows (idempotent)
        continue
    for _, disp in CROPS:
        if f"{disp} & SIAP" in line:
            out_lines.extend(new_rows[disp])
            break
    out_lines.append(line)

with open(tex_path, "w") as f:
    f.writelines(out_lines)
print(f"\nwrote {tex_path}")
