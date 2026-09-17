"""
AEF Histogram Ensemble v2 — Dirichlet-Multinomial Coarsening

Variant of 2_gb_aef_hist_ensemble.py that replaces the fixed N=2 multinomial
subsampling with distributionally faithful coarsening:

  1. Effective pixel count N is drawn from the empirical distribution of ADC
     histogram thickness (estimated per ADC as 1 / smallest positive bin share).
  2. Bin probabilities are overdispersed via a Dirichlet draw before the
     multinomial, with concentration ALPHA tuned (feature-side only, no labels)
     so simulated subsamples match real ADC sparsity moments
     (share of empty bins, mean max-bin share).
  3. log1p(N) is appended as a feature so one model handles thick and thin
     histograms; each mun-year also contributes its uncorrupted histogram.

Everything else (percentile model, ensemble weight, additive correction,
evaluation vs INEGI CA22) matches the original for comparability. Baseline
metrics are recomputed from adc_aef_hist_ens_preds/_eval.parquet if present.

Usage:
  laptop:  ~/miniforge3/envs/geo_env/bin/python gb_aef_hist_ensemble_dm.py
  server:  conda activate mpc_env && python3 gb_aef_hist_ensemble_dm.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# CONFIGURATION
# ============================================================
K_SAMP      =  5       # coarse subsamples per mun-year (as in original)
W_BIN       =  0.5     # ensemble weight on bins model (as in original)
EVAL_YEAR   =  2022    # INEGI census year for evaluation
CROP        =  'Maize'
SEASON      =  'Spring-Summer'
MIN_YEAR    =  2017    # first year of AEF embeddings
N_CAP       =  2000    # cap on estimated effective pixel count
ALPHA_GRID  =  [2, 3, 4, 6, 8, 12, np.inf]   # Dirichlet concentration grid
N_SIM_TUNE  =  3000    # simulations for the alpha moment match
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


def normalize_hists(arr):
    """(n, 64, 8) raw shares -> row-normalized, plus validity mask."""
    arr  = np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    sums = arr.sum(axis=2, keepdims=True)
    ok   = (sums.squeeze(-1) > 1e-6).all(axis=1)
    sums[sums < 1e-8] = 1.0
    return arr / sums, ok


def estimate_n(arr):
    """Effective pixel count per unit: 1 / smallest positive bin share."""
    pos  = np.where(arr > 1e-6, arr, np.inf)
    smin = pos.min(axis=(1, 2))
    return np.clip(np.round(1.0 / smin), 1, N_CAP).astype(np.int64)


def dm_coarsen(hists, Ns, alpha, rng):
    """Dirichlet-multinomial coarsening of (n, 64, 8) histograms to counts/N."""
    out = np.zeros_like(hists, dtype=np.float32)
    for j in range(len(hists)):
        p = hists[j]
        if np.isfinite(alpha):
            g = rng.gamma(np.maximum(alpha * p, 1e-12)) * (p > 0)
            p = g / np.clip(g.sum(axis=1, keepdims=True), 1e-12, None)
        for d in range(64):
            pv = p[d] / max(p[d].sum(), 1e-12)
            out[j, d] = rng.multinomial(Ns[j], pv) / Ns[j]
    return out


t0 = time.time()


# ── 1. Load training data ────────────────────────────────
print("Loading data...")

mun_bh   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet"))
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


# ── 2. ADC features + effective pixel counts ─────────────
print(f"\nLoading ADC test data ({EVAL_YEAR})...")

adc_bh =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_binned_hist.parquet"))
adc_bh =  adc_bh[adc_bh['year'] == EVAL_YEAR].copy()

adc_arr, adc_ok =  normalize_hists(
    adc_bh[bin_cols].values.astype(np.float64).reshape(len(adc_bh), 64, 8))
adc_bh  =  adc_bh[adc_ok].copy()
adc_arr =  adc_arr[adc_ok]
adc_N   =  estimate_n(adc_arr)
adc_bh['log_n'] =  np.log1p(adc_N)

# real ADC sparsity moments (targets for the alpha moment match)
tgt_empty =  (adc_arr < 1e-6).mean()
tgt_maxsh =  adc_arr.max(axis=2).mean()
print(f"  ADCs: {len(adc_bh):,} | N quantiles "
      f"{np.percentile(adc_N, [5, 50, 95]).astype(int)} | "
      f"targets: empty={tgt_empty:.3f}, maxsh={tgt_maxsh:.3f}")


# ── 3. Tune Dirichlet concentration (features only) ──────
print("\nTuning Dirichlet concentration...")
mun_arr, mun_ok =  normalize_hists(
    train_bin[bin_cols].values.astype(np.float64).reshape(len(train_bin), 64, 8))
train_bin =  train_bin[mun_ok].reset_index(drop=True)
mun_arr   =  mun_arr[mun_ok]

rng   =  np.random.default_rng(42)
idx   =  rng.integers(0, len(mun_arr), N_SIM_TUNE)
Ns_t  =  rng.choice(adc_N, N_SIM_TUNE)
best  =  (np.inf, None)
for alpha in ALPHA_GRID:
    sim  =  dm_coarsen(mun_arr[idx], Ns_t, alpha, np.random.default_rng(7))
    e, m =  (sim < 1e-6).mean(), sim.max(axis=2).mean()
    d    =  (e - tgt_empty)**2 + (m - tgt_maxsh)**2
    print(f"  alpha={alpha}: empty={e:.3f}, maxsh={m:.3f}, dist={d:.5f}")
    if d < best[0]:
        best = (d, alpha)
ALPHA = best[1]
print(f"  -> ALPHA = {ALPHA}")


# ── 4. Build augmented training set & train bins model ───
print(f"\nCoarsening (K={K_SAMP}, alpha={ALPHA}, empirical N)...")
yields  =  train_bin['yield'].values
all_X, all_y =  [], []

# uncorrupted mun histograms (thick end of the curriculum)
mun_N  =  estimate_n(mun_arr)
all_X.append(np.column_stack([mun_arr.reshape(len(mun_arr), -1),
                              np.log1p(mun_N)]).astype(np.float32))
all_y.append(yields)

for k in range(K_SAMP):
    rk  =  np.random.default_rng(100 + k)
    Ns  =  rk.choice(adc_N, len(mun_arr))
    Xk  =  dm_coarsen(mun_arr, Ns, ALPHA, rk)
    all_X.append(np.column_stack([Xk.reshape(len(Xk), -1),
                                  np.log1p(Ns)]).astype(np.float32))
    all_y.append(yields)
    print(f"    subsample {k+1}/{K_SAMP}", flush=True)

aug_X, aug_y =  np.vstack(all_X), np.concatenate(all_y)
print(f"  augmented set: {aug_X.shape}")

print("Training DM bins model...")
m_bin = HistGradientBoostingRegressor(**cfg)
m_bin.fit(aug_X, aug_y)


# ── 5. Train percentile model (unchanged) ────────────────
print("\nTraining percentile model...")
m_pct = HistGradientBoostingRegressor(**cfg)
m_pct.fit(
    train_pct[pct_combined].fillna(0).values.astype(np.float32),
    train_pct['yield'].values
)


# ── 6. Ensemble predictions on ADCs ──────────────────────
print(f"\nGenerating ensemble predictions (w_bin={W_BIN})...")

adc_pct_d =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"))
adc_pct_d =  adc_pct_d[adc_pct_d['year'] == EVAL_YEAR].copy()
adc_pct_d['muncode'] = adc_pct_d['adcid'].str[:5]

adc_m =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
adc_m =  adc_m[adc_m['year'] == EVAL_YEAR].copy()

adc_pct_full = adc_pct_d.merge(
    adc_m[['adcid', 'year'] + mean_cols],
    on=['adcid', 'year'], how='inner'
)

adc_bh['_row'] =  np.arange(len(adc_bh))
adc_combo = adc_bh.merge(
    adc_pct_full[['adcid', 'year'] + pct_combined],
    on=['adcid', 'year'], how='inner'
)

X_adc =  np.column_stack([
    adc_arr[adc_combo['_row'].values].reshape(len(adc_combo), -1),
    adc_combo['log_n'].values
]).astype(np.float32)

pred_bin = m_bin.predict(X_adc).clip(0)
pred_pct = m_pct.predict(
    adc_combo[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

adc_combo['pred'] = W_BIN * pred_bin + (1 - W_BIN) * pred_pct
adc_combo['pred_bin_only'] = pred_bin
adc_combo['adc']  = adc_combo['adcid'].str.replace('-', '', regex=False)


# ── 7. Load INEGI ground truth ───────────────────────────
ca = pd.read_stata(os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta"))
ca_maize = ca[ca['name'] == CROP].copy()
gt = ca_maize[['adc', 'muncode', 'yield', 'land_input']].copy()

ca_szn = pd.read_stata(os.path.join(ca2022_dir, "adc_land_szn_ca22_adc07.dta"))
gt_pv = ca_szn[(ca_szn['name'] == CROP) & (ca_szn['type'] == 'p-v')][
    ['adc', 'muncode', 'yield']
].rename(columns={'yield': 'yield_pv'})

gt = gt.merge(gt_pv, on=['adc', 'muncode'], how='left')


# ── 8. Merge predictions, additive correction ────────────
df = gt.merge(adc_combo[['adc', 'pred', 'pred_bin_only']], on='adc', how='left')

agland = pd.read_csv(os.path.join(proj_dir, "Data", "SIAP_agland", "Output",
                                  "2007_adcs_agland_area.csv"))
agland['adc'] = agland['adcid'].astype(str).str.replace('-', '', regex=False)
df = df.merge(agland[['adc', 'siap_agland_area']], on='adc', how='left')
df['corr_w'] = np.where(df['siap_agland_area'] > 0, df['siap_agland_area'],
                        df['land_input'])

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
# the P-V corrected R2. Same fix as 2_gb_aef_hist_ensemble.py / 4_accuracy_main_2022.py.
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
out_path  =  os.path.join(pred_dir, "adc_aef_hist_ens_dm_preds.parquet")
adc_out[['adcid', 'muncode', 'year', 'pred']].to_parquet(out_path, index=False)
print(f"  {out_path}  ({len(adc_out):,} rows)")

df_out  =  df[['adc', 'muncode', 'land_input', 'yield', 'yield_pv',
                 'pred', 'pred_corr', 'pred_corr_pv']].copy()
out_eval_path  =  os.path.join(pred_dir, "adc_aef_hist_ens_dm_eval.parquet")
df_out.to_parquet(out_eval_path, index=False)
print(f"  {out_eval_path}  ({len(df_out):,} rows)")


# ── 9. Evaluate (incl. original-ensemble baseline) ───────
print(f"\n{'='*80}")
print(f"  {'Model':<45s} {'N':>8s} {'R²':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
print(f"  {'-'*77}")

print("\n  --- Combined season ---")
eval_row(df, 'yield', 'pred',           'DM Ens. Raw')
eval_row(df, 'yield', 'pred_corr',      'DM Ens. Corr.')
eval_row(df, 'yield', 'pred_bin_only',  'DM Bins-only Raw')

print("\n  --- Spring-summer (P-V) ---")
df_pv = df[df['yield_pv'].notna()].copy()
eval_row(df_pv, 'yield_pv', 'pred',      'DM Ens. Raw')
eval_row(df_pv, 'yield_pv', 'pred_corr_pv', 'DM Ens. Corr.')

base_path = os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet")
if os.path.exists(base_path):
    base = pd.read_parquet(base_path)
    print("\n  --- Baseline: original N=2 multinomial ensemble ---")
    eval_row(base, 'yield', 'pred',      'AEF Hist Ens. Raw (orig)')
    eval_row(base, 'yield', 'pred_corr', 'AEF Hist Ens. Corr. (orig)')
    base_pv = base[base['yield_pv'].notna()].copy()
    print("  (P-V)")
    eval_row(base_pv, 'yield_pv', 'pred',      'AEF Hist Ens. Raw (orig)')
    # Season-matched anchor: 2_gb_aef_hist_ensemble.py now writes pred_corr_pv
    # (Spring-Summer anchor) alongside pred_corr (all seasons). Scoring the P-V
    # target against pred_corr would repeat the mismatch fixed 2026-08-15.
    # Fall back only if the baseline parquet predates that fix.
    _pv_col = 'pred_corr_pv' if 'pred_corr_pv' in base_pv.columns else 'pred_corr'
    if _pv_col == 'pred_corr':
        print("    (warning: baseline parquet has no pred_corr_pv — stale, "
              "re-run 2_gb_aef_hist_ensemble.py)")
    eval_row(base_pv, 'yield_pv', _pv_col, 'AEF Hist Ens. Corr. (orig)')

print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")
