"""
Municipality-aggregated accuracy of the AEF Hist Ensemble (the row added to
Table \ref{tab:mun_agg_results}). ADC-level Hist-Ensemble predictions are
aggregated to the municipality level using census planted-area (land_input)
weights---identical to the methodology in 1_accuracy_metrics_2022.ipynb---and
scored against the INEGI 2022 census municipal yield (Panel A) and the SIAP
municipal estimate (Panel B).

Reproduces the notebook's AEF Standard / Agg-NN rows to within rounding
(vs-SIAP matches exactly: AEF 0.615, Agg-NN 0.885), confirming the weighting.

Run:  ~/miniforge3/envs/geo_env/bin/python mun_agg_hist_ens.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P    = os.path.join(proj, "Data", "predictions")
INEGI= os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                    "LM2304-CA22-2025-09-29-superficie_ENTREGA")
siap_path = os.path.join(home, "Dropbox", "Projects", "Maize_prediction", "Data", "SIAP", "Cleaned", "siap_ag_prod_estimation_by_season.dta")

def r2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1 - np.sum((a - b)**2) / np.sum((a - a.mean())**2)
def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a - b)**2))

# census ADC base (all maize ADCs)
ca = pd.read_stata(os.path.join(INEGI, "adc_land_use_ca22_adc07.dta"))
ca = ca[ca['name'] == 'Maize'][['adc', 'muncode', 'land_input', 'vol_output']].copy()

# SIAP 2022 maize mun yield
siap = pd.read_stata(siap_path)
siap['muncode'] = siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
sm = siap[(siap['name'] == 'Maize') & (siap['year'] == 2022)]
sm = sm[~sm['muncode'].str.endswith('000')].groupby('muncode').agg(
    q=('q', 'sum'), ha=('ha_planted', 'sum')).reset_index()
sm['yield_siap'] = sm['q'] / sm['ha']

# AEF Hist Ensemble ADC predictions (2022)
he = pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_preds.parquet"))
he = he[he['year'] == 2022].copy()
he['adc'] = he['adcid'].str.replace('-', '', regex=False)
ca = ca.merge(he[['adc', 'pred']].dropna().drop_duplicates('adc'), on='adc', how='left')

# aggregate to mun (land_input weights)
d = ca.dropna(subset=['pred']).copy()
d['Q'] = d['pred'] * d['land_input']
g = d.groupby('muncode').agg(Q=('Q', 'sum'), area=('land_input', 'sum'),
                             vol=('vol_output', 'sum'), la=('land_input', 'sum')).reset_index()
g['pred'] = g['Q'] / g['area']
g['census'] = g['vol'] / g['la']
g = g.merge(sm[['muncode', 'yield_siap']], on='muncode', how='left')

a = g.dropna(subset=['census', 'pred'])
b = g.dropna(subset=['yield_siap', 'pred'])
print(f"Panel A (vs Census): N={len(a):,}  R2={r2(a['census'],a['pred']):.3f}  RMSE={rmse(a['census'],a['pred']):.3f}")
print(f"Panel B (vs SIAP):   N={len(b):,}  R2={r2(b['yield_siap'],b['pred']):.3f}  RMSE={rmse(b['yield_siap'],b['pred']):.3f}")
print("\nLaTeX rows for Table tab:mun_agg_results:")
print(f"  Panel A:  AEF Hist Ens.\\ (agg.) & {len(a):,} & {r2(a['census'],a['pred']):.3f} & {rmse(a['census'],a['pred']):.3f} \\\\")
print(f"  Panel B:  AEF Hist Ens.\\ (agg.) & {len(b):,} & {r2(b['yield_siap'],b['pred']):.3f} & {rmse(b['yield_siap'],b['pred']):.3f} \\\\")
