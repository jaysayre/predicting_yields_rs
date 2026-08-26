"""
irrigation_adjustment.py
========================
Phase 1: Two-Stage Irrigation Adjustment for within-municipality R².

Adjusts existing RF predictions using known irrigation-yield relationship:
  ADC_pred_adjusted = mun_pred + beta_hat * (irrig_share_adc - irrig_share_mun_mean)

Steps:
  1. Compute irrig_share per ADC from SIAP agland shapefile data
  2. Estimate within-mun irrigation-yield gradient from CA2007 census
  3. Apply post-hoc correction to AEF RF predictions
  4. Save adjusted predictions

Author: Jay Sayre
Date: 2026-02-22
"""

import os
import numpy  as np
import pandas as pd
from   scipy  import stats

### ------------------------------------------------------------------ ###
### Directories
### ------------------------------------------------------------------ ###

home_dir      =  os.path.expanduser("~")
proj_dir      =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir      =  os.path.join(proj_dir, "Data")
pred_dir      =  os.path.join(data_dir, "predictions")
crop_sub_dir  =  os.path.join(home_dir, "Dropbox", "Projects",
                               "Maize_prediction")
siap_dir      =  os.path.join(crop_sub_dir, "Data", "SIAP", "Cleaned")
py_md_lab_dir =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
agland_path   =  os.path.join(data_dir, "SIAP_agland", "Output",
                               "2007_adcs_agland_area.csv")
ca2007_path   =  os.path.join(py_md_lab_dir, "ca2007_maize_amca_adcs.dta")

### ------------------------------------------------------------------ ###
### Step 1: Load irrigation shares per ADC
### ------------------------------------------------------------------ ###

print("=" * 70)
print("  Step 1: Load ADC irrigation shares")
print("=" * 70)

agland =  pd.read_csv(agland_path)
agland['irrig_share'] =  (agland['siap_irrig_area'] / agland['siap_agland_area'])
agland['irrig_share'] =  agland['irrig_share'].replace([np.inf, -np.inf], np.nan)

# Fill NaN irrig_share with 0 (ADCs with no SIAP agland → likely non-agricultural)
agland['irrig_share'] =  agland['irrig_share'].fillna(0.0)

# Create muncode from adc07
agland['muncode'] =  agland['adc07'].str[:5]

# Also create dash-free version for matching with INEGI 2022
agland['adc_nodash'] =  agland['adc07'].str.replace('-', '', regex=False)

print(f"  Loaded {len(agland):,} ADCs from agland data")
print(f"  irrig_share: mean={agland['irrig_share'].mean():.3f}, "
      f"median={agland['irrig_share'].median():.3f}")

# Compute municipality-level mean irrig_share (for demeaning)
mun_irrig =  agland.groupby('muncode')['irrig_share'].mean().reset_index()
mun_irrig.columns =  ['muncode', 'irrig_share_mun']

agland =  agland.merge(mun_irrig, on='muncode', how='left')
agland['irrig_share_dev'] =  agland['irrig_share'] - agland['irrig_share_mun']

print(f"  irrig_share_dev: mean={agland['irrig_share_dev'].mean():.6f}, "
      f"std={agland['irrig_share_dev'].std():.3f}")

### ------------------------------------------------------------------ ###
### Step 2: Estimate irrigation-yield gradient from CA2007 census
### ------------------------------------------------------------------ ###

print("\n" + "=" * 70)
print("  Step 2: Estimate within-mun irrigation-yield gradient (CA2007)")
print("=" * 70)

ca2007 =  pd.read_stata(ca2007_path)

# Filter to spring-summer maize (type='p-v')
ca2007_pv =  ca2007[ca2007['type'] == 'p-v'].copy()
ca2007_pv =  ca2007_pv[ca2007_pv['yield'].notna() & (ca2007_pv['yield'] > 0)]

print(f"  CA2007 spring-summer maize: {len(ca2007_pv):,} ADC obs")
print(f"  Yield: mean={ca2007_pv['yield'].mean():.2f}, median={ca2007_pv['yield'].median():.2f}")

# Merge with agland to get irrigation share
ca2007_pv['adc_nodash'] =  ca2007_pv['adc'].str.replace('-', '', regex=False)
ca2007_merged =  ca2007_pv.merge(
    agland[['adc07', 'irrig_share', 'irrig_share_dev']].rename(columns={'adc07': 'adc'}),
    on='adc', how='inner'
)

print(f"  After merge with agland: {len(ca2007_merged):,} obs")

# Demean both yield and irrig_share by municipality (FE regression)
mun_means =  ca2007_merged.groupby('muncode')[['yield', 'irrig_share']].transform('mean')
ca2007_merged['yield_dm']  =  ca2007_merged['yield'] - mun_means['yield']
ca2007_merged['irrig_dm']  =  ca2007_merged['irrig_share'] - mun_means['irrig_share']

# Only keep municipalities with variation in irrigation
mun_var =  ca2007_merged.groupby('muncode')['irrig_dm'].std()
valid_muns =  mun_var[mun_var > 0.01].index
ca2007_fe  =  ca2007_merged[ca2007_merged['muncode'].isin(valid_muns)]

print(f"  Municipalities with irrig variation: {len(valid_muns):,} "
      f"({len(ca2007_fe):,} obs)")

# OLS on demeaned data: yield_dm = beta * irrig_dm + epsilon
slope, intercept, r_value, p_value, std_err =  stats.linregress(
    ca2007_fe['irrig_dm'], ca2007_fe['yield_dm']
)

print(f"\n  Within-mun regression (FE, CA2007):")
print(f"    beta_hat (irrig→yield) = {slope:.3f} t/ha per unit irrig_share")
print(f"    SE = {std_err:.3f}")
print(f"    t-stat = {slope / std_err:.1f}")
print(f"    R² (within) = {r_value**2:.4f}")
print(f"    N = {len(ca2007_fe):,}")

beta_hat =  slope

### ------------------------------------------------------------------ ###
### Step 3: Apply adjustment to AEF RF predictions
### ------------------------------------------------------------------ ###

print("\n" + "=" * 70)
print("  Step 3: Apply irrigation adjustment to RF predictions")
print("=" * 70)

# Load RF predictions
rf_preds =  pd.read_parquet(os.path.join(pred_dir, "adc_alpha_earth_preds_maize.parquet"))
rf_preds['muncode'] =  rf_preds['adcid'].str[:5]
print(f"  RF predictions: {len(rf_preds):,} ADC-year obs")

# Merge with irrig_share (using adc07 = adcid format)
rf_adj =  rf_preds.merge(
    agland[['adc07', 'irrig_share', 'irrig_share_dev']].rename(columns={'adc07': 'adcid'}),
    on='adcid', how='left'
)

# For ADCs without irrigation data, deviation = 0 (no adjustment)
rf_adj['irrig_share']     =  rf_adj['irrig_share'].fillna(0.0)
rf_adj['irrig_share_dev'] =  rf_adj['irrig_share_dev'].fillna(0.0)

n_matched =  (rf_adj['irrig_share_dev'] != 0).sum()
print(f"  ADC-years with non-zero irrigation deviation: {n_matched:,}")

# Apply adjustment
rf_adj['yield_pred_irrig_adj'] =  (
    rf_adj['yield_pred'] + beta_hat * rf_adj['irrig_share_dev']
).clip(lower=0)

# Also compute municipality-mean prediction and recentered version
mun_mean_pred =  rf_adj.groupby(['muncode', 'year'])['yield_pred'].transform('mean')
rf_adj['yield_pred_irrig_adj_recentered'] =  (
    mun_mean_pred + beta_hat * rf_adj['irrig_share_dev']
).clip(lower=0)

print(f"  Adjustment range: [{beta_hat * rf_adj['irrig_share_dev'].min():.2f}, "
      f"{beta_hat * rf_adj['irrig_share_dev'].max():.2f}] t/ha")
print(f"  Adjusted yield: mean={rf_adj['yield_pred_irrig_adj'].mean():.2f}, "
      f"std={rf_adj['yield_pred_irrig_adj'].std():.2f}")

### ------------------------------------------------------------------ ###
### Step 4: Save predictions
### ------------------------------------------------------------------ ###

print("\n" + "=" * 70)
print("  Step 4: Save adjusted predictions")
print("=" * 70)

out_cols =  ['adcid', 'year', 'yield_pred', 'yield_pred_irrig_adj',
             'yield_pred_irrig_adj_recentered', 'irrig_share']
out_path =  os.path.join(pred_dir, "adc_alpha_earth_preds_maize_irrig_adj.parquet")
rf_adj[out_cols].to_parquet(out_path, index=False)
print(f"  Saved: {out_path}")
print(f"  Shape: {rf_adj[out_cols].shape}")

### ------------------------------------------------------------------ ###
### Step 5: Evaluate vs INEGI 2022 census
### ------------------------------------------------------------------ ###

print("\n" + "=" * 70)
print("  Step 5: Evaluate vs INEGI 2022 census")
print("=" * 70)


def standard_r2(y, yhat):
    mask =  np.isfinite(y) & np.isfinite(yhat)
    y, yhat =  np.array(y[mask]), np.array(yhat[mask])
    if len(y) < 2:
        return np.nan
    ss_res =  np.sum((y - yhat) ** 2)
    ss_tot =  np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def within_r2(df, y_col, yhat_col, group_col='muncode'):
    sub =  df[[y_col, yhat_col, group_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    grp_counts =  sub.groupby(group_col).size()
    valid_grps =  grp_counts[grp_counts >= 2].index
    sub =  sub[sub[group_col].isin(valid_grps)]
    if len(sub) == 0:
        return np.nan
    grp_means   =  sub.groupby(group_col)[[y_col, yhat_col]].transform('mean')
    y_w         =  sub[y_col]    - grp_means[y_col]
    yhat_w      =  sub[yhat_col] - grp_means[yhat_col]
    return standard_r2(y_w, yhat_w)


def between_r2(df, y_col, yhat_col, group_col='muncode'):
    sub =  df[[y_col, yhat_col, group_col]].replace([np.inf, -np.inf], np.nan).dropna()
    grp =  sub.groupby(group_col)[[y_col, yhat_col]].mean()
    return standard_r2(grp[y_col], grp[yhat_col])


def compute_rmse(y, yhat):
    mask =  np.isfinite(y) & np.isfinite(yhat)
    y, yhat =  np.array(y[mask]), np.array(yhat[mask])
    return np.sqrt(np.mean((y - yhat) ** 2)) if len(y) > 0 else np.nan


def compute_all_metrics(df, y_col, yhat_col, group_col='muncode'):
    sub =  df[[y_col, yhat_col, group_col]].replace([np.inf, -np.inf], np.nan).dropna()
    mask =  np.isfinite(sub[y_col]) & np.isfinite(sub[yhat_col])
    return {
        'N':          int(mask.sum()),
        'R2':         standard_r2(sub[y_col], sub[yhat_col]),
        'Between_R2': between_r2(sub, y_col, yhat_col, group_col),
        'Within_R2':  within_r2(sub, y_col, yhat_col, group_col),
        'RMSE':       compute_rmse(sub[y_col], sub[yhat_col]),
    }


# Load INEGI 2022
ca2022_path =  os.path.join(py_md_lab_dir,
               "LM2304-CA22-2025-09-29-superficie_ENTREGA",
               "adc_land_use_ca22_adc07.dta")

adc_gt =  pd.read_stata(ca2022_path)
adc_gt =  adc_gt[adc_gt['name'] == 'Maize']
adc_gt =  adc_gt[['adc', 'muncode', 'land_input', 'vol_output',
                   'yield', 'share_irrig']].copy()
print(f"  INEGI 2022 maize ADCs: {len(adc_gt):,}")

# Get 2022 RF adjusted predictions
rf_2022 =  rf_adj[rf_adj['year'] == 2022].copy()
rf_2022['adc'] =  rf_2022['adcid'].str.replace('-', '', regex=False)

# Merge
eval_df =  adc_gt.merge(
    rf_2022[['adc', 'yield_pred', 'yield_pred_irrig_adj',
             'yield_pred_irrig_adj_recentered']],
    on='adc', how='left'
)

# Also add SIAP municipal average as baseline
siap_szn =  pd.read_stata(os.path.join(siap_dir,
            "siap_ag_prod_estimation_by_season.dta"))
siap_szn['muncode'] =  siap_szn['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap_szn['yield_siap'] =  siap_szn['q'] / siap_szn['ha_planted']
siap_2022 =  siap_szn[
    (siap_szn['name'] == 'Maize') &
    (siap_szn['growing_season'] == 'Spring-Summer') &
    (siap_szn['year'] == 2022)
][['muncode', 'yield_siap']]
eval_df =  eval_df.merge(siap_2022, on='muncode', how='left')

print(f"  Merged evaluation data: {len(eval_df):,} ADCs")

# Compute metrics for each model
models =  {
    'yield_pred':                     'AEF RF (baseline)',
    'yield_pred_irrig_adj':           'AEF RF + Irrig Adj',
    'yield_pred_irrig_adj_recentered': 'AEF RF + Irrig Adj (recentered)',
    'yield_siap':                     'SIAP Mun. Avg.',
}

print(f"\n{'Model':40s} {'N':>7s} {'R²':>7s} {'Between':>8s} {'Within':>8s} {'RMSE':>7s}")
print("-" * 80)
for pred_col, name in models.items():
    m =  compute_all_metrics(eval_df, 'yield', pred_col, group_col='muncode')
    print(f"  {name:38s} {m['N']:>7,} {m['R2']:>7.3f} {m['Between_R2']:>8.3f} "
          f"{m['Within_R2']:>8.3f} {m['RMSE']:>7.3f}")

# Diagnostic: correlation between adjusted predictions and share_irrig within muns
print("\n  Diagnostic: Within-mun correlation of predictions with census share_irrig")
for pred_col, name in models.items():
    sub =  eval_df[['muncode', pred_col, 'share_irrig']].dropna()
    if len(sub) < 10:
        continue
    grp_means =  sub.groupby('muncode')[[pred_col, 'share_irrig']].transform('mean')
    p_dev     =  sub[pred_col] - grp_means[pred_col]
    s_dev     =  sub['share_irrig'] - grp_means['share_irrig']
    mask      =  (p_dev.std() > 0) & (s_dev.std() > 0)
    if p_dev.std() > 0 and s_dev.std() > 0:
        corr =  np.corrcoef(p_dev, s_dev)[0, 1]
        print(f"    {name:38s}  r = {corr:.4f}")

print("\nDone.")
