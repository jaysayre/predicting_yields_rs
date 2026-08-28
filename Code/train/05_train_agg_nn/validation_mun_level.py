"""
Generate validation_mun_level_2022.tex  (Table \ref{tab:validation_mun}).

Municipality-level held-out validation: every model is evaluated on the SAME
held-out municipality-years (SIAP Spring-Summer maize, 2017-2024), and every
model is trained with those municipalities excluded.

The 80/20 municipality split is reproduced EXACTLY from agg_constrained_nn.py
(np.random.seed(42); shuffle over sorted unique municipalities with both AEF
embeddings and a SIAP Spring-Summer maize label, 2017--2024).

STANDARDIZED 2026-08-14. The previous version of this table mixed three
protocols, so N varied by row (18,428 / 3,734 / 450) and several rows were
partially in-sample:
  * NDVI rows were full-sample random muni-year CV (all municipalities);
  * AEF mean / AEF Hist / AEF Hist Ens. were trained on ALL municipalities
    (their producers refit on everything before predicting ADCs), so the
    "held-out" municipalities were in their training labels;
  * the ensemble prediction file covered 2022 only.
All five non-Agg-NN models are now retrained by
train_holdout_validation_models.py with the held-out municipalities excluded
(same architectures and hyperparameter grids; config selection on an inner
muni-grouped split of the training municipalities), predicting 2017-2024.
This script then scores all six models on the intersection of held-out
municipality-years where every model has a prediction, so N is identical
across rows. Previous published numbers, for reference:
  NDVI Hist. 18,428/0.632/1.333   NDVI Q-Hist. 18,428/0.618/1.359
  AEF mean 3,734/0.743/1.092      Agg-NN 3,734/0.813/0.932
  AEF Hist 3,734/0.713/1.155      AEF Hist Ens. 450/0.730/1.159

UPDATED 2026-08-15: the two unmasked NDVI rows (NDVI Hist. / NDVI Q-Hist.,
h3 2D-histograms) are replaced by a SINGLE "NDVI" row built from the
cropland-masked aefn2 features, now that that panel is complete (4,784/4,784
batches). Its holdout model comes from train_holdout_validation_models.py
train_ndvi_masked(), which reads the muni feature cache built by
analysis/02_accuracy_maize/masked_muni_cv.py.

AEF models' ADC-level predictions are aggregated to the municipality level
using SIAP agricultural-land area weights; NDVI predicts at the
municipality level directly. The Agg-NN row uses the inner-ES model
(adc_agg_nn_preds_maize_phase2_inner_es.parquet, from agg_constrained_nn.py
--inner_es, 2026-08-28): trained on the 80% training municipalities with
early stopping on an inner 10% slice of them, so -- like every other row --
the validation municipalities never influence any training choice. (The old
phase2 dev model early-stopped on the validation municipalities themselves;
the *_final model is trained on all data and would be in-sample.)

Writes to tables/ (does NOT overwrite the Overleaf copy). Review, then copy.

Run:  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
      python3 validation_mun_level.py
"""
import os
import numpy as np
import pandas as pd

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir   =  os.path.join(proj_dir, "Data", "predictions")
agland_path=  os.path.join(proj_dir, "Data", "SIAP_agland", "Output",
                           "2007_adcs_agland_area.csv")
table_dir  =  os.path.join(proj_dir, "tables")
siap_path  =  os.path.join(home_dir, "Dropbox", "Projects",
                           "Maize_prediction", "Data", "SIAP", "Cleaned",
                           "siap_ag_prod_estimation_by_season.dta")
os.makedirs(table_dir, exist_ok=True)

SEED      =  42

# ── Load AEF mun universe + area weights ────────────────
aef = pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"),
                      columns=['adcid', 'year'])
aef['muncode'] = aef['adcid'].str[:5]

ag = pd.read_csv(agland_path)
ag['w'] = np.where(ag['siap_agland_area'] > 0, ag['siap_agland_area'], 1.0)
aef = aef.merge(ag[['adcid', 'w']], on='adcid', how='left')
aef['w'] = aef['w'].fillna(1.0)

# ── SIAP Spring-Summer maize labels (2017+) ─────────────
siap = pd.read_stata(siap_path)
siap['muncode'] = siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap['yield']   = siap['q'] / siap['ha_planted']
sm = siap[(siap['name'] == 'Maize') &
          (siap['growing_season'] == 'Spring-Summer') &
          (siap['year'] >= 2017)][['muncode', 'year', 'yield']]

# ── Reproduce the 80/20 municipality split (seed 42) ────
valid = sorted(set(zip(aef['muncode'], aef['year'])) &
               set(zip(sm['muncode'], sm['year'])))
umuns = sorted(set(k[0] for k in valid))
np.random.seed(SEED)
np.random.shuffle(umuns)
val_muns = set(umuns[int(0.8 * len(umuns)):])
print(f"  municipalities: {len(umuns)} total, {len(val_muns)} held out (20%)")


def agg_mun_preds(pred_file, col):
    """Area-weighted aggregate of ADC-level predictions -> mun-year frame."""
    p = (pd.read_parquet(os.path.join(pred_dir, pred_file)) if pred_file.endswith("parquet")
         else pd.read_csv(os.path.join(pred_dir, pred_file)))
    p = p.dropna(subset=[col]).copy()
    p['muncode'] = p['adcid'].astype(str).str[:5]
    p = p.merge(aef[['adcid', 'year', 'w']], on=['adcid', 'year'], how='left')
    p['w'] = p['w'].fillna(1.0)
    p['wv'] = p[col] * p['w']
    g = (p.groupby(['muncode', 'year']).agg(wv=('wv', 'sum'), w=('w', 'sum')).reset_index())
    g['pred'] = g['wv'] / g['w']
    return g[['muncode', 'year', 'pred']]

def mun_preds(pred_file, col):
    """Mun-level prediction file (muncode/year/pred) -- no ADC aggregation."""
    p = pd.read_parquet(os.path.join(pred_dir, pred_file))
    p['muncode'] = p['muncode'].astype(str).str.zfill(5)
    p = p.dropna(subset=[col]).rename(columns={col: 'pred'})
    return p[['muncode', 'year', 'pred']]

# (group, label, file, column) — Landsat-derived then AEF-derived.
# All *_holdout_* files come from train_holdout_validation_models.py (models
# trained with val_muns excluded). Agg-NN uses the inner-ES model,
# already trained on the 80% training municipalities only.
# 2026-08-15: the two unmasked h3 variants (NDVI Hist. / NDVI Q-Hist.) are folded
# into a single cropland-masked "NDVI" row — the aefn2 baseline, which
# uses the same per-dim distributional recipe as AEF, so the comparison isolates
# the embeddings with methodology held fixed.
MUN_LEVEL = {"NDVI"}
MODELS = [
    ("Landsat-derived features", "NDVI", "mun_aefn2_masked_gb_holdout_preds.parquet", "yield_pred"),
    ("AEF-derived features",     "AEF mean",       "adc_alpha_earth_holdout_preds_maize.parquet", "yield_pred"),
    ("AEF-derived features",     "Agg-NN",         "adc_agg_nn_preds_maize_phase2_inner_es.parquet",       "yield_pred_agg_nn"),
    ("AEF-derived features",     "AEF Hist",       "adc_aef_hist_gb_holdout_preds.parquet",       "yield_pred"),
    ("AEF-derived features",     "AEF Hist Ens.",  "adc_aef_hist_ens_holdout_preds.parquet",      "pred"),
]

# ── Build per-model mun-year predictions ────────────────
frames = {}
for grp, label, f, col in MODELS:
    fr = mun_preds(f, col) if label in MUN_LEVEL else agg_mun_preds(f, col)
    frames[label] = fr
    print(f"  {label:14s}: {len(fr):,} mun-year predictions")

# ── Common held-out evaluation sample ───────────────────
# intersection of mun-years covered by EVERY model, restricted to the held-out
# municipalities and to mun-years with a SIAP label
common = None
for fr in frames.values():
    s = set(zip(fr['muncode'], fr['year']))
    common = s if common is None else common & s
sm_val = sm[sm['muncode'].isin(val_muns)]
common &= set(zip(sm_val['muncode'], sm_val['year']))
print(f"  common held-out sample: {len(common):,} mun-years")

# ── Score all models on the common sample ───────────────
rows, lines, cur = [], [], None
for grp, label, f, col in MODELS:
    g = frames[label].merge(sm, on=['muncode', 'year'], how='inner')
    g = g[[k in common for k in zip(g['muncode'], g['year'])]]
    y, yh = g['yield'].values, g['pred'].values
    n    = len(g)
    r2v  = 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean((y - yh)**2))
    print(f"  {label:14s}: N={n:,}  R2={r2v:.3f}  RMSE={rmse:.3f}")
    if grp != cur:
        if cur is not None:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{grp}}}}} \\"); cur = grp
    lines.append(f"{label} & {n:,} & {r2v:.3f} & {rmse:.3f} \\\\")

# ── Write LaTeX table ───────────────────────────────────
tex = ("\\begin{table}[!htbp]\n\\centering\n"
       "\\caption{Municipality-level validation performance (SIAP Spring-Summer maize, "
       "2017--2024). Every model is trained with the same randomly held-out 20\\% of "
       "municipalities excluded from training and evaluated on the same held-out "
       "municipality-years (the intersection of municipality-years for which all models "
       "produce a prediction), so $N$ is identical across rows. AEF models' ADC-level "
       "predictions are aggregated to the municipality level using agricultural-land-area "
       "weights; the NDVI model is trained and scored directly at the "
       "municipality level. RMSE in t/ha.}\n"
       "\\label{tab:validation_mun}\n\\begin{tabular}{lrrr}\n\\hline\n"
       "Model & $N$ & $R^2$ & RMSE \\\\\n\\hline\n"
       + "\n".join(lines) +
       "\n\\hline\n\\end{tabular}\n\\end{table}\n")
out = os.path.join(table_dir, "validation_mun_level_2022.tex")
with open(out, "w") as f:
    f.write(tex)
print(f"\nWrote {out}")
