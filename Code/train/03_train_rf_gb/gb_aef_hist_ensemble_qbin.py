"""
AEF Histogram Ensemble — Quantile-Bin Pilot

Identical protocol to gb_aef_hist_ensemble.py (subsampled bins model, N=2
multinomial draws, K=5, ensembled with the percentile model at w=0.4), but the
bin features come from per-dim equal-mass quantile edges
(ee_alpha_earth_quantile_hist.py) instead of fixed-width bins on [-0.8, 0.8].

Motivation: fixed-width bins deliver a median of ~2 effective levels per dim
(99.4% of mass in the central 4 bins); quantile bins restore 8.

Prints a common-sample comparison against the original fixed-bin ensemble
(adc_aef_hist_ens_eval.parquet) at the end.

Usage:
  laptop:  ~/miniforge3/envs/geo_env/bin/python gb_aef_hist_ensemble_qbin.py
  server:  conda activate mpc_env && python3 gb_aef_hist_ensemble_qbin.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# CONFIGURATION (matches gb_aef_hist_ensemble.py)
# ============================================================
N_PIX       =  2       # pixels per subsample draw
K_SAMP      =  5       # subsamples per mun-year
W_BIN       =  0.4     # ensemble weight on bins model
EVAL_YEAR   =  2022    # INEGI census year for evaluation
CROP        =  'Maize'
SEASON      =  'Spring-Summer'
MIN_YEAR    =  2017    # first year of AEF embeddings
# ============================================================


# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir   =  os.path.join(proj_dir, "Data", "predictions")
inegi_dir  =  os.path.join(proj_dir, "Data", "INEGI", "MD_lab_outputs")
ca2022_dir =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")
siap_path  =  os.path.join(home_dir, "Dropbox", "Projects",
                            "Maize_prediction", "Data", "SIAP", "Cleaned",
                            "siap_ag_prod_estimation_by_season.dta")

os.makedirs(pred_dir, exist_ok=True)


# ── Feature column names ────────────────────────────────
mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  []
for d in range(64):
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"A{d:02d}{s}")
pct_combined =  pct_cols + mean_cols   # 448 features


# ── HistGB config ────────────────────────────────────────
cfg = {
    'max_iter':        1500,
    'max_depth':       8,
    'learning_rate':   0.03,
    'min_samples_leaf': 5,
    'random_state':    42,
    'early_stopping':  False,
}


# ── Helpers ──────────────────────────────────────────────
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = np.array(y[m]), np.array(yh[m])
    if len(y) < 2:
        return np.nan
    st = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y - yh)**2) / st if st > 0 else np.nan

def within_r2(df, y, p, gc='muncode'):
    sub = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    c = sub.groupby(gc).size()
    sub = sub[sub[gc].isin(c[c >= 2].index)]
    if len(sub) == 0:
        return np.nan
    gm = sub.groupby(gc)[[y, p]].transform('mean')
    return r2(sub[y] - gm[y], sub[p] - gm[p])

def between_r2(df, y, p, gc='muncode'):
    sub = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = sub.groupby(gc)[[y, p]].mean()
    return r2(g[y], g[p])

def eval_row(df, ycol, pcol, label):
    sub = df[[ycol, pcol, 'muncode']].replace([np.inf, -np.inf], np.nan).dropna()
    n    = len(sub)
    ov   = r2(sub[ycol], sub[pcol])
    b    = between_r2(sub, ycol, pcol)
    w    = within_r2(sub, ycol, pcol)
    rmse = np.sqrt(np.mean((sub[ycol].values - sub[pcol].values)**2))
    print(f"  {label:<45s} {n:>8,} {ov:>6.3f} {b:>6.3f} {w:>7.3f} {rmse:>6.3f}")
    return n, ov, b, w, rmse


def subsample_bins(train_df, bin_cols, K, N, seed=42):
    """Create K synthetic ADC-like observations per mun-year by
    multinomial subsampling from the mun bin distributions."""
    rng = np.random.default_rng(seed)
    n_dims, n_bins = 64, 8
    n_mun = len(train_df)

    bin_arr = train_df[bin_cols].values.reshape(n_mun, n_dims, n_bins).astype(np.float64)
    bin_arr = np.clip(np.nan_to_num(bin_arr, nan=0.0), 0, None)
    sums = bin_arr.sum(axis=2, keepdims=True)
    sums[sums < 1e-8] = 1.0
    bin_arr = bin_arr / sums

    yields = train_df['yield'].values
    all_bin, all_y = [], []

    for k in range(K):
        bin_out = np.zeros((n_mun, n_dims, n_bins), dtype=np.float32)
        for d in range(n_dims):
            for i in range(n_mun):
                bin_out[i, d, :] = rng.multinomial(N, bin_arr[i, d, :]) / N
        all_bin.append(bin_out.reshape(n_mun, n_dims * n_bins))
        all_y.append(yields)
        print(f"    subsample {k+1}/{K}", flush=True)

    return np.vstack(all_bin), np.concatenate(all_y)


t0 = time.time()


# ── 1. Load training data ────────────────────────────────
print("Loading data...")

mun_bh   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_qbin_hist.parquet"))
bin_cols  =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])

mun_pct  =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
mun_pct['CVE_ENT']  =  mun_pct['CVE_ENT'].astype(str).str.zfill(2)
mun_pct['CVE_MUN']  =  mun_pct['CVE_MUN'].astype(str).str.zfill(3)
mun_pct['muncode']  =  mun_pct['CVE_ENT'] + mun_pct['CVE_MUN']

mun_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
mun_mean['CVE_ENT']  =  mun_mean['CVE_ENT'].astype(str).str.zfill(2)
mun_mean['CVE_MUN']  =  mun_mean['CVE_MUN'].astype(str).str.zfill(3)
mun_mean['muncode']  =  mun_mean['CVE_ENT'] + mun_mean['CVE_MUN']

mun_pct_full = mun_pct.merge(
    mun_mean[['muncode', 'year'] + mean_cols],
    on=['muncode', 'year'], how='inner'
)

siap = pd.read_stata(siap_path)
siap['muncode']  =  siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap['yield']    =  siap['q'] / siap['ha_planted']
siap_train = siap[
    (siap['name'] == CROP)
    & (siap['growing_season'] == SEASON)
    & (siap['year'] >= MIN_YEAR)
]
siap_train = siap_train[
    siap_train['yield'].notna()
    & (siap_train['yield'] > 0)
    & ~siap_train['muncode'].str.endswith('000')
][['muncode', 'year', 'yield']]

train_bin =  mun_bh.merge(siap_train, on=['muncode', 'year'], how='inner')
train_pct =  mun_pct_full.merge(siap_train, on=['muncode', 'year'], how='inner')
print(f"  Training mun-years: bin={len(train_bin):,}, pct={len(train_pct):,}")


# ── 2. Train percentile model ───────────────────────────
print("\nTraining percentile model...")
m_pct = HistGradientBoostingRegressor(**cfg)
m_pct.fit(
    train_pct[pct_combined].fillna(0).values.astype(np.float32),
    train_pct['yield'].values
)


# ── 3. Build subsampled bins & train bins model ─────────
print(f"\nSubsampling bins (N={N_PIX}, K={K_SAMP})...")
aug_bin, aug_y = subsample_bins(train_bin, bin_cols, K=K_SAMP, N=N_PIX, seed=42)

print("Training subsampled bins model...")
m_bin = HistGradientBoostingRegressor(**cfg)
m_bin.fit(aug_bin.astype(np.float32), aug_y)


# ── 4. Load ADC test data (EVAL_YEAR) ───────────────────
print(f"\nLoading ADC test data ({EVAL_YEAR})...")

adc_bh =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_qbin_hist.parquet"))
adc_bh =  adc_bh[adc_bh['year'] == EVAL_YEAR].copy()

adc_pct_d =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"))
adc_pct_d =  adc_pct_d[adc_pct_d['year'] == EVAL_YEAR].copy()
adc_pct_d['muncode'] = adc_pct_d['adcid'].str[:5]

adc_m =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
adc_m =  adc_m[adc_m['year'] == EVAL_YEAR].copy()

adc_pct_full = adc_pct_d.merge(
    adc_m[['adcid', 'year'] + mean_cols],
    on=['adcid', 'year'], how='inner'
)

adc_combo = adc_bh.merge(
    adc_pct_full[['adcid', 'year'] + pct_combined],
    on=['adcid', 'year'], how='inner'
)


# ── 5. Ensemble predictions ─────────────────────────────
print(f"\nGenerating ensemble predictions (w_bin={W_BIN})...")

pred_bin = m_bin.predict(
    adc_combo[bin_cols].fillna(0).values.astype(np.float32)
).clip(0)
pred_pct = m_pct.predict(
    adc_combo[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

adc_combo['pred'] = W_BIN * pred_bin + (1 - W_BIN) * pred_pct
adc_combo['pred_bin_only'] = pred_bin
adc_combo['adc']  = adc_combo['adcid'].str.replace('-', '', regex=False)


# ── 6. Load INEGI ground truth ───────────────────────────
ca = pd.read_stata(os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta"))
ca_maize = ca[ca['name'] == CROP].copy()
gt = ca_maize[['adc', 'muncode', 'yield', 'land_input']].copy()

ca_szn = pd.read_stata(os.path.join(ca2022_dir, "adc_land_szn_ca22_adc07.dta"))
gt_pv = ca_szn[(ca_szn['name'] == CROP) & (ca_szn['type'] == 'p-v')][
    ['adc', 'muncode', 'yield']
].rename(columns={'yield': 'yield_pv'})

gt = gt.merge(gt_pv, on=['adc', 'muncode'], how='left')


# ── 7. Merge predictions with GT ────────────────────────
df = gt.merge(adc_combo[['adc', 'pred', 'pred_bin_only']], on='adc', how='left')

agland = pd.read_csv(os.path.join(proj_dir, "Data", "SIAP_agland", "Output",
                                  "2007_adcs_agland_area.csv"))
agland['adc'] = agland['adc07'].astype(str).str.replace('-', '', regex=False)
df = df.merge(agland[['adc', 'siap_agland_area']], on='adc', how='left')
df['corr_w'] = np.where(df['siap_agland_area'] > 0, df['siap_agland_area'],
                        df['land_input'])


# ── 8. Additive ex-post correction ──────────────────────
print("\nApplying additive correction (ex-ante ag-land weights)...")

df['wQ'] = df['pred'] * df['corr_w']
df['wA'] = df.apply(
    lambda x: x['corr_w'] if np.isfinite(x['pred']) else 0, axis=1
)
mun_agg = df.groupby('muncode').agg({'wQ': 'sum', 'wA': 'sum'}).reset_index()
mun_agg['pred_mun_avg'] = mun_agg['wQ'] / mun_agg['wA']

siap_2022 = siap[(siap['name'] == CROP) & (siap['year'] == EVAL_YEAR)].copy()
siap_2022 = siap_2022[~siap_2022['muncode'].str.endswith('000')]
siap_mun  = siap_2022.groupby('muncode').agg(
    {'q': 'sum', 'ha_planted': 'sum'}
).reset_index()
siap_mun['yield_siap'] = siap_mun['q'] / siap_mun['ha_planted']
siap_mun = siap_mun[
    siap_mun['yield_siap'].notna() & (siap_mun['yield_siap'] > 0)
][['muncode', 'yield_siap']]

# Spring-Summer-only anchor for the P-V rows (fixed 2026-08-15). The anchor
# above sums ALL growing seasons, which is right for the combined-season target
# (`yield`) but a season mismatch for the P-V target (`yield_pv`): it removes a
# bias defined on a different quantity than the one being scored, and inflates
# the P-V corrected R2. Same fix as gb_aef_hist_ensemble.py / accuracy_main_2022.py.
siap_mun_pv = siap_2022[siap_2022['growing_season'] == SEASON].groupby('muncode').agg(
    {'q': 'sum', 'ha_planted': 'sum'}
).reset_index()
siap_mun_pv['yield_siap'] = siap_mun_pv['q'] / siap_mun_pv['ha_planted']
siap_mun_pv = siap_mun_pv[
    siap_mun_pv['yield_siap'].notna() & (siap_mun_pv['yield_siap'] > 0)
][['muncode', 'yield_siap']]
print(f"  anchors: combined {len(siap_mun):,} munis | {SEASON} {len(siap_mun_pv):,} munis")


def _apply_correction(anchor, colname):
    """Additive ex-post correction against a given municipal anchor."""
    a = mun_agg[['muncode', 'pred_mun_avg']].merge(anchor, on='muncode', how='left')
    a['diff'] = a['pred_mun_avg'] - a['yield_siap']
    m = df.merge(a[['muncode', 'diff']], on='muncode', how='left')
    out = (m['pred'] - m['diff']).clip(lower=0)
    out[m['pred'].isna()] = np.nan
    df[colname] = out.values


_apply_correction(siap_mun,    'pred_corr')      # combined season: all seasons
_apply_correction(siap_mun_pv, 'pred_corr_pv')   # P-V: Spring-Summer only


# ── 8b. Save predictions ────────────────────────────────
print("\nSaving predictions ...")
adc_out  =  adc_combo[['adcid', 'year', 'pred']].copy()
adc_out['muncode']  =  adc_out['adcid'].str[:5]
out_path  =  os.path.join(pred_dir, "adc_aef_hist_ens_qbin_preds.parquet")
adc_out[['adcid', 'muncode', 'year', 'pred']].to_parquet(out_path, index=False)
print(f"  {out_path}  ({len(adc_out):,} rows)")

df_out  =  df[['adc', 'muncode', 'land_input', 'yield', 'yield_pv',
                 'pred', 'pred_corr', 'pred_corr_pv']].copy()
out_eval_path  =  os.path.join(pred_dir, "adc_aef_hist_ens_qbin_eval.parquet")
df_out.to_parquet(out_eval_path, index=False)
print(f"  {out_eval_path}  ({len(df_out):,} rows)")


# ── 9. Evaluate ─────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  {'Model':<45s} {'N':>8s} {'R²':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
print(f"  {'-'*77}")

print("\n  --- Combined season ---")
eval_row(df, 'yield', 'pred',           'Qbin Ens. Raw')
eval_row(df, 'yield', 'pred_corr',      'Qbin Ens. Corr.')
eval_row(df, 'yield', 'pred_bin_only',  'Qbin Bins-only Raw')

print("\n  --- Spring-summer (P-V) ---")
df_pv = df[df['yield_pv'].notna()].copy()
eval_row(df_pv, 'yield_pv', 'pred',      'Qbin Ens. Raw')
eval_row(df_pv, 'yield_pv', 'pred_corr_pv', 'Qbin Ens. Corr.')


# ── 10. Common-sample comparison vs fixed-bin original ──
base_path = os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet")
if os.path.exists(base_path):
    base =  pd.read_parquet(base_path)
    m    =  base.merge(df[['adc', 'pred', 'pred_corr', 'pred_bin_only']],
                       on='adc', suffixes=('_orig', '_qbin'))
    m    =  m[m['pred_orig'].notna() & m['pred_qbin'].notna()]
    print(f"\n  --- Common sample vs fixed-bin original ({len(m):,} ADCs) ---")
    eval_row(m, 'yield', 'pred_orig',      'Fixed-bin Ens. Raw')
    eval_row(m, 'yield', 'pred_qbin',      'Qbin Ens. Raw')
    eval_row(m, 'yield', 'pred_corr_orig', 'Fixed-bin Ens. Corr.')
    eval_row(m, 'yield', 'pred_corr_qbin', 'Qbin Ens. Corr.')
    eval_row(m, 'yield', 'pred_bin_only',  'Qbin Bins-only Raw')
    m_pv = m[m['yield_pv'].notna()].copy()
    print("  (P-V)")
    eval_row(m_pv, 'yield_pv', 'pred_orig', 'Fixed-bin Ens. Raw')
    eval_row(m_pv, 'yield_pv', 'pred_qbin', 'Qbin Ens. Raw')

print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")
