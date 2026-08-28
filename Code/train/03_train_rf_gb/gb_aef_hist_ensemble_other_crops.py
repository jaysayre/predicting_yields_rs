"""
AEF Histogram Ensemble for non-maize crops.

Adapts the subsampled-bins + percentile ensemble approach from
gb_aef_hist_ensemble.py to Sorghum, Sugar, Wheat, and Avocados.
Uses crop-specific SIAP yields for training with the same
crop-agnostic AEF features.

For each crop:
  1. Train bins model on subsampled mun histograms (N=2, K=5)
  2. Train percentile model on mun percentile+mean features
  3. Ensemble: w*bins + (1-w)*pct, w=0.4
  4. Evaluate vs INEGI 2022 census
  5. Apply additive ex-post correction

Updates the Overleaf other-crops ADC accuracy table.

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 gb_aef_hist_ensemble_other_crops.py
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
N_PIX       =  2
K_SAMP      =  5
W_BIN       =  0.4
EVAL_YEAR   =  2022
MIN_YEAR    =  2017

# Crop -> dominant SIAP season (matches rf_yield_prediction.py)
CROP_SEASONS = {
    'Sorghum':  'Spring-Summer',
    'Sugar':    'Perennial',
    'Wheat':    'Fall-Winter',
    'Avocados': 'Perennial',
}
# ============================================================


# -- Directories ---------------------------------------------
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir   =  os.path.join(proj_dir, "Data", "predictions")
inegi_dir  =  os.path.join(proj_dir, "Data", "INEGI", "MD_lab_outputs")
ca2022_dir =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")
overleaf   =  os.path.join(home_dir, "Dropbox", "Overleaf",
                           "Predicting Yields at Scale using RS")
siap_path  =  os.path.join(home_dir, "Dropbox", "Projects",
                            "Maize_prediction", "Data", "SIAP", "Cleaned",
                            "siap_ag_prod_estimation_by_season.dta")

os.makedirs(pred_dir, exist_ok=True)


# -- Feature column names ------------------------------------
mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  []
for d in range(64):
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"A{d:02d}{s}")
pct_combined =  pct_cols + mean_cols


# -- HistGB config -------------------------------------------
cfg = {
    'max_iter':        1500,
    'max_depth':       8,
    'learning_rate':   0.03,
    'min_samples_leaf': 5,
    'random_state':    42,
    'early_stopping':  False,
}


# -- Helpers -------------------------------------------------
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
    return {'N': n, 'R2': ov, 'Btw': b, 'Wtn': w, 'RMSE': rmse}

def cv_lambda(df, pcol, ycol, gc='muncode'):
    """Leave-municipalities-out CV shrinkage factor lambda* = rho/r in [0,1]."""
    from sklearn.model_selection import GroupKFold
    s = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    cnt = s.groupby(gc)[ycol].transform('size'); s = s[cnt >= 2].reset_index(drop=True)
    if s[gc].nunique() < 5:
        return np.nan
    lams = []
    for tr, _ in GroupKFold(n_splits=5).split(s, groups=s[gc]):
        d = s.iloc[tr]
        a = (d[ycol] - d.groupby(gc)[ycol].transform('mean')).values
        b = (d[pcol] - d.groupby(gc)[pcol].transform('mean')).values
        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if den <= 0:
            continue
        rho = np.sum(a * b) / den; rr = np.sqrt(np.sum(b * b) / np.sum(a * a))
        if rr > 0:
            lams.append(rho / rr)
    return float(np.clip(np.mean(lams), 0, 1)) if lams else np.nan

def apply_shrink(df, pcol, lam, gc='muncode'):
    g = df.groupby(gc)[pcol]
    return g.transform('mean') + lam * (df[pcol] - g.transform('mean'))

def subsample_bins(train_df, bin_cols, K, N, seed=42):
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
        print(f"      subsample {k+1}/{K}", flush=True)
    return np.vstack(all_bin), np.concatenate(all_y)


# ============================================================
# MAIN
# ============================================================
t0 = time.time()

# -- 1. Load shared data (AEF features, SIAP, INEGI) --------
print("Loading shared data...")

# Mun binned histograms
mun_bh   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet"))
bin_cols  =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])

# Mun percentile + mean features
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

# ADC features (2022)
adc_bh =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_binned_hist.parquet"))
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
adc_combo['adc'] = adc_combo['adcid'].str.replace('-', '', regex=False)

# Drop rows whose features are ENTIRELY absent (ported from gb_aef_hist_ensemble.py,
# where this was fixed 2026-07-27; this script still had the old behaviour).
# The parquets carry a row for every ADC but leave values null where the ESA
# WorldCover cropland mask found no pixels. The fillna(0) further down would
# otherwise turn those into all-zero vectors and emit a confident-looking constant
# (measured R2 = 0.006 on such rows in the maize run), inflating every crop's
# metrics. Applied once here, so it covers all four crops in the loop below.
_bin_null = adc_combo[bin_cols].isna().all(axis=1)
_pct_null = adc_combo[pct_cols].isna().all(axis=1)
print(f"  dropping {int((_bin_null | _pct_null).sum()):,} ADCs with no cropland pixels "
      f"(all-null features)")
adc_combo = adc_combo[~(_bin_null | _pct_null)].copy()

# SIAP yields (all crops)
siap = pd.read_stata(siap_path)
siap['muncode']  =  siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap['yield']    =  siap['q'] / siap['ha_planted']

# INEGI ground truth
ca = pd.read_stata(os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta"))

print(f"  ADC features: {len(adc_combo):,}")


# -- 2. Loop over crops --------------------------------------
all_adc_results = []

for crop_name, season in CROP_SEASONS.items():
    print(f"\n{'='*70}")
    print(f"  {crop_name} (season: {season})")
    print(f"{'='*70}")

    # Training data: SIAP yields for this crop/season
    siap_crop = siap[
        (siap['name'] == crop_name)
        & (siap['growing_season'] == season)
        & (siap['year'] >= MIN_YEAR)
    ]
    siap_crop = siap_crop[
        siap_crop['yield'].notna()
        & (siap_crop['yield'] > 0)
        & ~siap_crop['muncode'].str.endswith('000')
    ][['muncode', 'year', 'yield']]

    train_bin = mun_bh.merge(siap_crop, on=['muncode', 'year'], how='inner')
    train_pct = mun_pct_full.merge(siap_crop, on=['muncode', 'year'], how='inner')
    print(f"  Training mun-years: bin={len(train_bin):,}, pct={len(train_pct):,}")

    if len(train_bin) < 50 or len(train_pct) < 50:
        print(f"  SKIPPING: too few training obs")
        continue

    # Train percentile model
    print(f"  Training percentile model...")
    m_pct_crop = HistGradientBoostingRegressor(**cfg)
    m_pct_crop.fit(
        train_pct[pct_combined].fillna(0).values.astype(np.float32),
        train_pct['yield'].values
    )

    # Subsample bins & train bins model
    print(f"  Subsampling bins (N={N_PIX}, K={K_SAMP})...")
    aug_bin, aug_y = subsample_bins(train_bin, bin_cols, K=K_SAMP, N=N_PIX, seed=42)

    print(f"  Training subsampled bins model...")
    m_bin_crop = HistGradientBoostingRegressor(**cfg)
    m_bin_crop.fit(aug_bin.astype(np.float32), aug_y)

    # Ensemble predictions
    pred_bin = m_bin_crop.predict(
        adc_combo[bin_cols].fillna(0).values.astype(np.float32)
    ).clip(0)
    pred_pct = m_pct_crop.predict(
        adc_combo[pct_combined].fillna(0).values.astype(np.float32)
    ).clip(0)
    adc_combo[f'pred_{crop_name}'] = W_BIN * pred_bin + (1 - W_BIN) * pred_pct

    # Ground truth
    gt_crop = ca[ca['name'] == crop_name][['adc', 'muncode', 'yield', 'land_input']].copy()
    df = gt_crop.merge(
        adc_combo[['adc', f'pred_{crop_name}']],
        on='adc', how='left'
    )

    # SIAP 2022 for correction
    siap_2022 = siap[
        (siap['name'] == crop_name)
        & (siap['year'] == EVAL_YEAR)
        & ~siap['muncode'].str.endswith('000')
    ]
    siap_mun = siap_2022.groupby('muncode').agg(
        {'q': 'sum', 'ha_planted': 'sum'}
    ).reset_index()
    siap_mun['yield_siap'] = siap_mun['q'] / siap_mun['ha_planted']
    siap_mun = siap_mun[
        siap_mun['yield_siap'].notna() & (siap_mun['yield_siap'] > 0)
    ][['muncode', 'yield_siap']]

    # Additive correction with EX-ANTE ag-land weights (not census land_input)
    pcol = f'pred_{crop_name}'
    if 'corr_w' not in df.columns:
        _ag = pd.read_csv(os.path.join(proj_dir, "Data", "SIAP_agland", "Output",
                                       "2007_adcs_agland_area.csv"))
        _ag['adc'] = _ag['adcid'].astype(str).str.replace('-', '', regex=False)
        df = df.merge(_ag[['adc', 'siap_agland_area']], on='adc', how='left')
        df['corr_w'] = np.where(df['siap_agland_area'] > 0, df['siap_agland_area'],
                                df['land_input'])
    df['wQ'] = df[pcol] * df['corr_w']
    df['wA'] = df.apply(
        lambda x, c=pcol: x['corr_w'] if np.isfinite(x[c]) else 0, axis=1)
    mun_agg = df.groupby('muncode').agg({'wQ': 'sum', 'wA': 'sum'}).reset_index()
    mun_agg['pred_mun_avg'] = mun_agg['wQ'] / mun_agg['wA']
    mun_agg.loc[mun_agg['wA'] == 0, 'pred_mun_avg'] = np.nan
    mun_agg = mun_agg.merge(siap_mun, on='muncode', how='left')
    mun_agg['diff'] = mun_agg['pred_mun_avg'] - mun_agg['yield_siap']

    df = df.merge(mun_agg[['muncode', 'diff']], on='muncode', how='left')
    corr_col = f'pred_{crop_name}_corr'
    df[corr_col] = (df[pcol] - df['diff']).clip(lower=0)
    df.loc[df[pcol].isna(), corr_col] = np.nan

    # Evaluate
    print(f"\n  {'Model':<45s} {'N':>8s} {'R2':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
    print(f"  {'-'*77}")

    m_raw  = eval_row(df, 'yield', pcol,     f'{crop_name} AEF Hist Ens. Raw')
    m_corr = eval_row(df, 'yield', corr_col, f'{crop_name} AEF Hist Ens. Corr.')

    # within-municipality shrinkage (leave-municipalities-out CV lambda)
    lam = cv_lambda(df, pcol, 'yield')
    shr_col = f'pred_{crop_name}_shrink'
    df[shr_col] = apply_shrink(df, pcol, lam) if np.isfinite(lam) else df[pcol]
    m_shr = eval_row(df, 'yield', shr_col, f'{crop_name} AEF Hist Ens. Shrink (lam={lam:.2f})')

    disp_name = {'Sugar': 'Sugarcane'}.get(crop_name, crop_name)  # display name in the paper table
    all_adc_results.append({'Crop': disp_name, 'Model': 'AEF Hist Ens.', **m_raw})
    all_adc_results.append({'Crop': disp_name, 'Model': 'AEF Hist Ens. Corr.', **m_corr})
    all_adc_results.append({'Crop': disp_name, 'Model': 'AEF Hist Ens. Shrink', **m_shr})

    # Persist ADC-level predictions so downstream municipal aggregation
    # (other_crops_mun_agg.py, Table A4) can use the ensemble instead of AEF mean
    _out_pq = os.path.join(proj_dir, "Data", "predictions",
                           f"adc_aef_hist_ens_preds_{crop_name.lower()}.parquet")
    df[['adc', 'muncode', pcol, corr_col, shr_col]].to_parquet(_out_pq, index=False)
    print(f"  Saved ADC predictions -> {_out_pq}")


# -- 3. Print summary & update Overleaf table ----------------
print(f"\n\n{'='*70}")
print("SUMMARY — New rows for accuracy_other_crops_adc_2022.tex")
print(f"{'='*70}")

for r in all_adc_results:
    def fmt(v):
        if np.isnan(v):
            return "---"
        if v < 0:
            return f"$-${abs(v):.3f}"
        return f"{v:.3f}"
    print(f"{r['Crop']} & {r['Model']} & {r['N']:,} & {fmt(r['R2'])} & {fmt(r['Btw'])} & {fmt(r['Wtn'])} & {fmt(r['RMSE'])} \\\\")


# -- 4. Read existing table and insert new rows --------------
# Writes to tables/ ONLY by default (2026-08-16). This script used to overwrite
# the live Overleaf copy on every run, which makes an exploratory re-run edit the
# paper silently. Pass --write_overleaf to update it deliberately; the template is
# still READ from Overleaf so the surrounding table structure is preserved.
WRITE_OVERLEAF = '--write_overleaf' in sys.argv
tex_path = os.path.join(overleaf, "accuracy_other_crops_adc_2022.tex")
proj_tex = os.path.join(proj_dir, "tables", "accuracy_other_crops_adc_2022.tex")
src_tex  = tex_path if os.path.exists(tex_path) else proj_tex
print(f"\nBuilding table (template: {src_tex})")

with open(src_tex, 'r') as f:
    old_lines = f.readlines()

def fmt(v):
    if np.isnan(v):
        return "---"
    return f"$-${abs(v):.3f}" if v < 0 else f"{v:.3f}"

def hist_ens_rows(crop_name):
    return [f"{r['Crop']} & {r['Model']} & {r['N']:,} & {fmt(r['R2'])} & {fmt(r['Btw'])} "
            f"& {fmt(r['Wtn'])} & {fmt(r['RMSE'])} \\\\\n"
            for r in all_adc_results if r['Crop'] == crop_name]

# Idempotent rebuild: drop any previously-inserted AEF Hist Ens rows, then insert
# the fresh Raw/Corr/Shrink rows just before each crop's SIAP row.
new_lines = []
for line in old_lines:
    if 'AEF Hist Ens' in line:          # remove stale ensemble rows (re-runnable)
        continue
    for crop_name in CROP_SEASONS:
        crop_disp = {'Sugar': 'Sugarcane'}.get(crop_name, crop_name)
        if f"{crop_disp} & SIAP" in line:
            new_lines.extend(hist_ens_rows(crop_name))
            break
    new_lines.append(line)

os.makedirs(os.path.dirname(proj_tex), exist_ok=True)
with open(proj_tex, 'w') as f:
    f.writelines(new_lines)
print(f"  Saved: {proj_tex}")

if WRITE_OVERLEAF:
    with open(tex_path, 'w') as f:
        f.writelines(new_lines)
    print(f"  Updated Overleaf: {tex_path}")
else:
    print("  Overleaf copy NOT touched (pass --write_overleaf to update it)")

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
