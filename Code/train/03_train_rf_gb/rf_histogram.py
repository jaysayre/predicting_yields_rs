"""
rf_histogram.py
===============
Train GradientBoosting / RF on distributional AEF histogram features.

Same approach as rf_yield_prediction.py but using:
  - Municipality-level histogram features (384) for training
  - ADC-level histogram features (384) for prediction

Trains on SIAP mun yields, evaluates ADC predictions vs INEGI 2022 census.

Usage:
  conda activate ML_env
  python3 rf_histogram.py

Author: Jay Sayre
Date: 2026-02-24
"""

import os
import time
import numpy  as np
import pandas as pd
from   sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import r2_score, mean_squared_error

### ------------------------------------------------------------------ ###
### Directories
### ------------------------------------------------------------------ ###

home_dir      =  os.path.expanduser("~")
proj_dir      =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir      =  os.path.join(proj_dir, "Data")
pred_dir      =  os.path.join(data_dir, "predictions")
aef_dir       =  os.path.join(data_dir, "alpha_earth")
inegi_dir     =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
crop_sub_dir  =  os.path.join(home_dir, "Dropbox", "Projects",
                               "The Promise of Crop Substitution")
siap_dir      =  os.path.join(crop_sub_dir, "data", "SIAP", "Cleaned")


### ------------------------------------------------------------------ ###
### Evaluation helpers
### ------------------------------------------------------------------ ###

def standard_r2(y, yhat):
    mask =  np.isfinite(y) & np.isfinite(yhat)
    y, yhat =  np.array(y[mask]), np.array(yhat[mask])
    if len(y) < 2: return np.nan
    ss_res =  np.sum((y - yhat) ** 2)
    ss_tot =  np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def within_r2(df, y_col, yhat_col, group_col='muncode'):
    sub =  df[[y_col, yhat_col, group_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    grp_counts =  sub.groupby(group_col).size()
    valid_grps =  grp_counts[grp_counts >= 2].index
    sub =  sub[sub[group_col].isin(valid_grps)]
    if len(sub) == 0: return np.nan
    grp_means  =  sub.groupby(group_col)[[y_col, yhat_col]].transform('mean')
    y_w        =  sub[y_col]    - grp_means[y_col]
    yhat_w     =  sub[yhat_col] - grp_means[yhat_col]
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
    return {
        'N':          int(sub[y_col].notna().sum()),
        'R2':         standard_r2(sub[y_col], sub[yhat_col]),
        'Between_R2': between_r2(sub, y_col, yhat_col, group_col),
        'Within_R2':  within_r2(sub, y_col, yhat_col, group_col),
        'RMSE':       compute_rmse(sub[y_col], sub[yhat_col]),
    }


### ------------------------------------------------------------------ ###
### Feature column names
### ------------------------------------------------------------------ ###

def add_zeros(x, n=2):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x

feat_names    =  [f"A{add_zeros(x)}" for x in range(64)]
pct_suffixes  =  ['_p10', '_p25', '_p50', '_p75', '_p90']
std_suffix    =  '_stdDev'
hist_cols     =  []
for feat in feat_names:
    for suf in pct_suffixes:
        hist_cols.append(feat + suf)
    hist_cols.append(feat + std_suffix)

mean_cols =  [f"A{add_zeros(x)}" for x in range(64)]


### ------------------------------------------------------------------ ###
### Main
### ------------------------------------------------------------------ ###

def main():
    t0 =  time.time()

    print("=" * 70)
    print("  RF/GB with Histogram Features")
    print("=" * 70)

    ### ---------------------------------------------------------------- ###
    ### Load SIAP yields
    ### ---------------------------------------------------------------- ###

    print("\n--- Loading data ---")

    siap_szn =  pd.read_stata(os.path.join(siap_dir,
                "siap_ag_prod_estimation_by_season.dta"))
    siap_szn['muncode'] =  siap_szn['muncode'].apply(lambda x: str(int(x)).zfill(5))
    siap_szn['yield']   =  siap_szn['q'] / siap_szn['ha_planted']
    siap_maize =  siap_szn[
        (siap_szn['name'] == 'Maize') &
        (siap_szn['growing_season'] == 'Spring-Summer') &
        (siap_szn['year'] >= 2017)
    ][['muncode', 'year', 'yield']].copy()
    siap_maize =  siap_maize[siap_maize['yield'].notna() & (siap_maize['yield'] > 0)]
    siap_maize =  siap_maize[~siap_maize['muncode'].str.endswith('000')]
    print(f"  SIAP mun-years: {len(siap_maize):,}")

    ### ---------------------------------------------------------------- ###
    ### Load municipality histogram features
    ### ---------------------------------------------------------------- ###

    mun_hist =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
    mun_hist['CVE_ENT'] =  mun_hist['CVE_ENT'].astype(str).str.zfill(2)
    mun_hist['CVE_MUN'] =  mun_hist['CVE_MUN'].astype(str).str.zfill(3)
    mun_hist['muncode'] =  mun_hist['CVE_ENT'] + mun_hist['CVE_MUN']
    print(f"  Mun hist: {len(mun_hist):,} mun-years, {len(hist_cols)} features")

    # Also load mean embeddings for comparison
    mun_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
    mun_mean['CVE_ENT'] =  mun_mean['CVE_ENT'].astype(str).str.zfill(2)
    mun_mean['CVE_MUN'] =  mun_mean['CVE_MUN'].astype(str).str.zfill(3)
    mun_mean['muncode'] =  mun_mean['CVE_ENT'] + mun_mean['CVE_MUN']

    ### ---------------------------------------------------------------- ###
    ### Load ADC histogram + mean features (for prediction)
    ### ---------------------------------------------------------------- ###

    adc_hist =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"))
    adc_hist['muncode'] =  adc_hist['adcid'].str[:5]
    print(f"  ADC hist: {len(adc_hist):,} ADC-years")

    adc_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
    adc_mean['muncode'] =  adc_mean['adcid'].str[:5]

    ### ---------------------------------------------------------------- ###
    ### Load INEGI 2022 for evaluation
    ### ---------------------------------------------------------------- ###

    ca2022_path =  os.path.join(inegi_dir,
                   "LM2304-CA22-2025-09-29-superficie_ENTREGA",
                   "adc_land_use_ca22_adc07.dta")
    adc_gt =  pd.read_stata(ca2022_path)
    adc_gt =  adc_gt[adc_gt['name'] == 'Maize']
    adc_gt =  adc_gt[['adc', 'muncode', 'yield']].copy()
    print(f"  INEGI ground truth: {len(adc_gt):,} ADCs")

    ### ---------------------------------------------------------------- ###
    ### Merge training data
    ### ---------------------------------------------------------------- ###

    # Histogram: only 2022
    train_hist =  mun_hist.merge(siap_maize, on=['muncode', 'year'], how='inner')
    print(f"\n  Hist training obs (2022 only): {len(train_hist):,}")

    # Mean: all years (for comparison baseline)
    train_mean =  mun_mean.merge(siap_maize, on=['muncode', 'year'], how='inner')
    train_mean_2022 =  train_mean[train_mean['year'] == 2022].copy()
    print(f"  Mean training obs (all years): {len(train_mean):,}")
    print(f"  Mean training obs (2022 only): {len(train_mean_2022):,}")

    ### ---------------------------------------------------------------- ###
    ### GB configs (same grid as rf_yield_prediction.py)
    ### ---------------------------------------------------------------- ###

    gb_configs =  [
        {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05,
         'min_samples_leaf': 5, 'subsample': 0.8},
        {'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.05,
         'min_samples_leaf': 5, 'subsample': 0.8},
        {'n_estimators': 2000, 'max_depth': 8, 'learning_rate': 0.03,
         'min_samples_leaf': 5, 'subsample': 0.8},
    ]

    ### ---------------------------------------------------------------- ###
    ### Helper: train + predict + evaluate
    ### ---------------------------------------------------------------- ###

    def train_and_evaluate(train_df, feat_list, adc_df, label, loc_feats=True):
        """Train GB on mun data, predict at ADC level, eval vs INEGI."""

        # Feature columns for training
        train_feats =  list(feat_list)
        if loc_feats:
            train_feats =  train_feats + ['CVE_ENT_num', 'CVE_MUN_num']
            train_df =  train_df.copy()
            train_df['CVE_ENT_num'] =  train_df['CVE_ENT'].astype(int)
            train_df['CVE_MUN_num'] =  train_df['CVE_MUN'].astype(int)

        X =  train_df[train_feats].fillna(0).values
        y =  train_df['yield'].values

        # Train/val split
        X_train, X_val, y_train, y_val =  train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Grid search
        best_r2   =  -999
        best_cfg  =  None
        for cfg in gb_configs:
            model =  GradientBoostingRegressor(random_state=42, **cfg)
            model.fit(X_train, y_train)
            val_r2 =  r2_score(y_val, model.predict(X_val))
            if val_r2 > best_r2:
                best_r2, best_cfg =  val_r2, cfg
        print(f"  {label}: best val R²={best_r2:.3f}")

        # Retrain on all data
        final_model =  GradientBoostingRegressor(random_state=42, **best_cfg)
        final_model.fit(X, y)

        # Predict at ADC level
        adc_df =  adc_df.copy()
        adc_feats =  list(feat_list)
        if loc_feats:
            adc_df['CVE_ENT_num'] =  adc_df['muncode'].str[:2].astype(int)
            adc_df['CVE_MUN_num'] =  adc_df['muncode'].str[2:5].astype(int)
            adc_feats =  adc_feats + ['CVE_ENT_num', 'CVE_MUN_num']

        X_adc =  adc_df[adc_feats].fillna(0).values
        adc_df['pred'] =  final_model.predict(X_adc).clip(min=0)

        # Merge with INEGI
        adc_df['adc'] =  adc_df['adcid'].str.replace('-', '', regex=False)
        eval_df =  adc_gt.merge(adc_df[['adc', 'pred']], on='adc', how='left')
        metrics =  compute_all_metrics(eval_df, 'yield', 'pred', group_col='muncode')
        return metrics, final_model

    ### ================================================================ ###
    ### Experiment 1: Histogram features (2022 only)
    ### ================================================================ ###

    print("\n" + "=" * 70)
    print("  Experiment 1: GB with histogram features (2022 only)")
    print("=" * 70)

    # 1a: Histogram features only
    m_hist, _ =  train_and_evaluate(
        train_hist, hist_cols, adc_hist[adc_hist['year'] == 2022],
        "Hist (384)", loc_feats=False)

    # 1b: Histogram + location
    m_hist_loc, _ =  train_and_evaluate(
        train_hist, hist_cols, adc_hist[adc_hist['year'] == 2022],
        "Hist+loc (386)", loc_feats=True)

    ### ================================================================ ###
    ### Experiment 2: Mean features (2022 only, fair comparison)
    ### ================================================================ ###

    print("\n" + "=" * 70)
    print("  Experiment 2: GB with mean features (2022 only, fair comparison)")
    print("=" * 70)

    # 2a: Mean features only (2022)
    m_mean_22, _ =  train_and_evaluate(
        train_mean_2022, mean_cols, adc_mean[adc_mean['year'] == 2022],
        "Mean (64, 2022)", loc_feats=False)

    # 2b: Mean + location (2022)
    m_mean_22_loc, _ =  train_and_evaluate(
        train_mean_2022, mean_cols, adc_mean[adc_mean['year'] == 2022],
        "Mean+loc (66, 2022)", loc_feats=True)

    ### ================================================================ ###
    ### Experiment 3: Mean features (all years, existing approach)
    ### ================================================================ ###

    print("\n" + "=" * 70)
    print("  Experiment 3: GB with mean features (all years)")
    print("=" * 70)

    # 3a: Mean + location (all years)
    m_mean_all, _ =  train_and_evaluate(
        train_mean, mean_cols, adc_mean[adc_mean['year'] == 2022],
        "Mean+loc (66, all years)", loc_feats=True)

    ### ================================================================ ###
    ### Experiment 4: Combined mean + histogram (2022 only)
    ### ================================================================ ###

    print("\n" + "=" * 70)
    print("  Experiment 4: GB with mean + histogram combined (2022 only)")
    print("=" * 70)

    # Merge mean and hist at mun level
    mun_combined =  mun_hist.merge(
        mun_mean[['muncode', 'year'] + mean_cols],
        on=['muncode', 'year'], how='inner')
    train_combined =  mun_combined.merge(siap_maize, on=['muncode', 'year'], how='inner')

    # Merge at ADC level
    adc_combined =  adc_hist[adc_hist['year'] == 2022].merge(
        adc_mean[adc_mean['year'] == 2022][['adcid', 'year'] + mean_cols],
        on=['adcid', 'year'], how='inner')

    combined_cols =  hist_cols + mean_cols
    m_combined, _ =  train_and_evaluate(
        train_combined, combined_cols, adc_combined,
        "Mean+Hist (448)", loc_feats=True)

    ### ================================================================ ###
    ### Summary
    ### ================================================================ ###

    print("\n" + "=" * 70)
    print("  Summary: ADC-level vs INEGI 2022")
    print("=" * 70)

    print(f"\n{'Model':<45s} {'R²':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s} {'N':>7s}")
    print("-" * 80)
    rows =  [
        ("GB Mean (64, 2022 only)",              m_mean_22),
        ("GB Mean+loc (66, 2022 only)",          m_mean_22_loc),
        ("GB Mean+loc (66, all years)",           m_mean_all),
        ("GB Hist (384, 2022 only)",             m_hist),
        ("GB Hist+loc (386, 2022 only)",         m_hist_loc),
        ("GB Mean+Hist+loc (450, 2022 only)",    m_combined),
    ]
    for name, m in rows:
        print(f"{name:<45s} {m['R2']:>6.3f} {m['Between_R2']:>6.3f} "
              f"{m['Within_R2']:>7.3f} {m['RMSE']:>6.3f} {m['N']:>7,}")

    print(f"\n  Runtime: {(time.time()-t0)/60:.1f} min")
    print("\nDone.")


if __name__ == '__main__':
    main()
