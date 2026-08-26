"""
Table \ref{tab:mun_agg_results} -- "Improving aggregated survey data".

ADC-level predictions are aggregated to the municipality level to test whether
the downscale-then-re-aggregate procedure beats existing survey data (SIAP).

IMPORTANT: the aggregation weights must be EX-ANTE. We therefore proxy each
ADC's maize area with its agricultural-land area (siap_agland_area, from the
2007 agricultural-land file) -- NOT the 2022 census planted area (land_input),
which would not be available to a researcher reproducing this procedure and
would smuggle census information into the comparison. This matches the proxy
described in the methodology ("the amount of agricultural land found in the
sub-region"). The census municipal yield used as the benchmark target is still
the census's own reported yield (total volume / total planted area); only the
weights used to combine PREDICTIONS are held ex-ante.

Run:  ~/miniforge3/envs/geo_env/bin/python mun_survey_improvement.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P    = os.path.join(proj, "Data", "predictions")
INEGI= os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                    "LM2304-CA22-2025-09-29-superficie_ENTREGA")
agland_path = os.path.join(proj, "Data", "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
siap_path   = os.path.join(home, "Dropbox", "Projects", "The Promise of Crop Substitution",
                           "data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")
table_dir   = os.path.join(proj, "tables")
os.makedirs(table_dir, exist_ok=True)

WEIGHT = "siap_agland_area"   # ex-ante agricultural-land proxy for maize area

def r2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1 - np.sum((a - b)**2) / np.sum((a - a.mean())**2)
def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a - b)**2))

# census ADC base + ex-ante agland proxy
ca = pd.read_stata(os.path.join(INEGI, "adc_land_use_ca22_adc07.dta"))
ca = ca[ca['name'] == 'Maize'][['adc', 'muncode', 'land_input', 'vol_output']].copy()
ag = pd.read_csv(agland_path); ag['adc'] = ag['adc07'].astype(str).str.replace('-', '', regex=False)
ca = ca.merge(ag[['adc', WEIGHT]], on='adc', how='left')

# SIAP 2022 maize municipal yield
siap = pd.read_stata(siap_path)
siap['muncode'] = siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
sm = siap[(siap['name'] == 'Maize') & (siap['year'] == 2022)]
sm = sm[~sm['muncode'].str.endswith('000')].groupby('muncode').agg(
    q=('q', 'sum'), ha=('ha_planted', 'sum')).reset_index()
sm['yield_siap'] = sm['q'] / sm['ha']

AGG = [  # (label, file, col) — NDVI row is the cropland-masked aefn2 baseline
         # (partial_masked_mun_train_adc_eval.py); it replaced the two unmasked
         # h3 harmonic variants on 2026-08-15.
    ("NDVI (masked)\\ (agg.)", "adc_aefn2_masked_preds.parquet",        "pred"),
    ("AEF mean (agg.)",  "adc_alpha_earth_preds.csv",          "yield_pred"),
    ("AEF Hist Ens.\\ (agg.)","adc_aef_hist_ens_preds.parquet",    "pred"),
    ("Agg-NN (agg.)",        "adc_mlp_yield_preds.csv",            "pred_yield"),
]
def load(f, col):
    d = pd.read_parquet(os.path.join(P, f)) if f.endswith("parquet") else pd.read_csv(os.path.join(P, f))
    if 'year' in d.columns: d = d[d['year'] == 2022]
    k = 'adc' if 'adc' in d.columns else 'adcid'
    d['adc'] = d[k].astype(str).str.replace('-', '', regex=False)
    return d[['adc', col]].dropna().drop_duplicates('adc')

panelA, panelB = [], []   # (label, N, R2, RMSE)
for label, f, col in AGG:
    d = ca.merge(load(f, col).rename(columns={col: 'p'}), on='adc', how='inner')
    d = d[d[WEIGHT] > 0].copy(); d['Q'] = d['p'] * d[WEIGHT]
    g = d.groupby('muncode').agg(Q=('Q', 'sum'), w=(WEIGHT, 'sum'),
                                 vol=('vol_output', 'sum'), la=('land_input', 'sum')).reset_index()
    g['pred'] = g['Q'] / g['w']; g['census'] = g['vol'] / g['la']
    g = g.merge(sm[['muncode', 'yield_siap']], on='muncode', how='left')
    a = g.dropna(subset=['census', 'pred']); b = g.dropna(subset=['yield_siap', 'pred'])
    panelA.append((label, len(a), r2(a['census'], a['pred']), rmse(a['census'], a['pred'])))
    panelB.append((label, len(b), r2(b['yield_siap'], b['pred']), rmse(b['yield_siap'], b['pred'])))

# NDVI (masked) GB (mun): trained directly at municipality level, random muni-year K-fold CV (no aggregation weight)
# Source switched 2026-08-15 from the unmasked h3 model to the masked aefn2 model (masked_muni_cv.py).
hr = pd.read_parquet(os.path.join(P, "mun_aefn2_masked_gb_kfold_preds.parquet"))
hr = hr[hr['year'] == 2022].copy()
hr['muncode'] = hr['muncode'].astype(str).str.zfill(5)
hr = hr.dropna(subset=['yield_pred']).drop_duplicates('muncode')
cmun = ca.groupby('muncode').agg(vol=('vol_output', 'sum'), la=('land_input', 'sum')).reset_index()
cmun['census'] = cmun['vol'] / cmun['la']
h = hr.merge(cmun[['muncode', 'census']], on='muncode', how='inner').merge(sm[['muncode', 'yield_siap']], on='muncode', how='left')
ha = h.dropna(subset=['census', 'yield_pred']); hb = h.dropna(subset=['yield_siap', 'yield_pred'])
panelA.append(("NDVI (masked)\\ GB (mun.)", len(ha), r2(ha['census'], ha['yield_pred']), rmse(ha['census'], ha['yield_pred'])))
panelB.append(("NDVI (masked)\\ GB (mun.)", len(hb), r2(hb['yield_siap'], hb['yield_pred']), rmse(hb['yield_siap'], hb['yield_pred'])))

# benchmarks
bench = cmun.merge(sm[['muncode', 'yield_siap']], on='muncode', how='inner').dropna(subset=['census', 'yield_siap'])
panelA.append(("SIAP Mun.\\ Avg.", len(bench), r2(bench['census'], bench['yield_siap']), rmse(bench['census'], bench['yield_siap'])))
panelB.append(("INEGI Census (agg.)", len(bench), r2(bench['yield_siap'], bench['census']), rmse(bench['yield_siap'], bench['census'])))

# order: Landsat (NDVI masked agg, mun-trained GB), AEF (mean, Hist Ens, Agg-NN), benchmark
def reorder(rows, last):
    order = ["NDVI (masked)\\ (agg.)", "NDVI (masked)\\ GB (mun.)",
             "AEF mean (agg.)", "AEF Hist Ens.\\ (agg.)", "Agg-NN (agg.)", last]
    return sorted([r for r in rows], key=lambda r: order.index(r[0]) if r[0] in order else 99)
panelA = reorder(panelA, "SIAP Mun.\\ Avg.")
panelB = reorder(panelB, "INEGI Census (agg.)")

def fmt(rows):
    out = []
    for lab, n, rr, rm in rows:
        out.append(f"{lab} & {n:,} & {rr:.3f} & {rm:.3f} \\\\")
    return out

print("Panel A (vs census):");  [print("  ", r) for r in fmt(panelA)]
print("Panel B (vs SIAP):");    [print("  ", r) for r in fmt(panelB)]

L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Municipality-level maize yield prediction results, 2022. ADC-level predictions are "
     r"aggregated to the municipality level weighting each ADC by its \emph{ex-ante} agricultural-land "
     r"area (a proxy for maize area that does not use the census), except NDVI (masked)\ GB (mun.), which is "
     r"trained directly at the municipality level (random municipality-year cross-validation).}", r"\label{tab:mun_agg_results}",
     r"\begin{tabular}{lrrr}", r"\hline",
     r"\multicolumn{4}{l}{\textit{Panel A: vs.\ INEGI Census (aggregated)}} \\", r"\hline",
     r"Model & $N$ & $R^2$ & RMSE \\", r"\hline"] + fmt(panelA) + [r"\hline",
     r"\multicolumn{4}{l}{\textit{Panel B: vs.\ SIAP Municipal Estimates}} \\", r"\hline",
     r"Model & $N$ & $R^2$ & RMSE \\", r"\hline"] + fmt(panelB) + [r"\hline",
     r"\end{tabular}", r"\end{table}", ""]
out = os.path.join(table_dir, "accuracy_mun_level_2022.tex")
with open(out, "w") as f: f.write("\n".join(L))
print(f"\nWrote {out}")
