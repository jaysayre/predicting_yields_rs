"""
Combined RF/GB yield prediction script.
Set MODE below to select which variant to run.

Modes:
  "mexico_maize"      - Mexico mun-level maize yields, RF or GB (IMPROVED toggle)
  "mexico_multi_crop" - Mexico mun-level multi-crop RF, predict on ADCs
"""
import os

import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ============================================================
# CONFIGURATION — set MODE to control which variant runs
# ============================================================
MODE     =  "mexico_maize"  # options: mexico_maize, mexico_multi_crop
IMPROVED =  False  # mexico_maize and mexico_multi_crop: False = RF, True = GB
# ============================================================


# ===== Shared helpers =====

def add_zeros(x, n):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


embed_cols   =  [f"A{add_zeros(x, 2)}" for x in range(64)]
# Paths below are relative to Maize_prediction/Data; the makefile invokes this from
# Maize_prediction, so anchor the working directory explicitly (fix 2026-09-02).
os.chdir(os.path.join(os.path.expanduser("~"), "Dropbox", "Projects", "Maize_prediction", "Data"))
os.makedirs("predictions", exist_ok=True)


# ===== Mode-specific functions =====

def run_mexico_maize():
    """Mexico mun-level maize yields (from RF.ipynb).

    IMPROVED=False: RF with max_depth search, embeddings only.
    IMPROVED=True:  GradientBoosting with tuned hyperparameters,
                    embeddings + CVE_ENT/CVE_MUN/year as features.
    """
    # Load municipality-level AEF embeddings
    alpha_earth =  pd.read_parquet("alpha_earth/alpha_earth_mex_muns.parquet")
    print(f"AlphaEarth: {alpha_earth.shape}")

    # Load SIAP maize yields (spring-summer)
    siap_path  =  os.path.join(os.path.expanduser("~"), "Dropbox", "Projects",
                   "Maize_prediction", "Data", "SIAP", "Cleaned",
                   "siap_ag_prod_estimation_by_season.dta")
    yields_raw =  pd.read_stata(siap_path)
    yields     =  yields_raw[(yields_raw['name'] == 'Maize') &
                              (yields_raw['growing_season'] == 'Spring-Summer')].copy()
    yields     =  yields[~yields['muncode'].str.endswith('000')]
    yields['muncode'] =  yields['muncode'].astype(int)
    yields['yield']   =  yields['q'] / yields['ha_planted']
    yields =  yields[['muncode', 'year', 'q', 'ha_planted', 'yield']]
    yields =  yields[yields['yield'].notna() & (yields['yield'] > 0)]
    print(f"Yields: {yields.shape}")

    # Merge
    alpha_earth['muncode'] =  (alpha_earth['CVE_ENT'].apply(lambda x: add_zeros(x, 2)) +
                                alpha_earth['CVE_MUN'].apply(lambda x: add_zeros(x, 3))).astype(int)
    alpha_yields =  pd.merge(alpha_earth, yields, on=['muncode', 'year'])
    print(f"Merged: {alpha_yields.shape}")

    # Feature columns and train/val split
    np.random.seed(42)

    if IMPROVED:
        feature_cols =  embed_cols + ['CVE_ENT', 'CVE_MUN', 'year']
        n_train      =  int(len(alpha_yields) * 0.8)
    else:
        feature_cols =  embed_cols
        n_train      =  7000

    train_idx =  np.random.choice(alpha_yields.index, n_train, replace=False)
    X_train   =  alpha_yields.loc[alpha_yields.index.isin(train_idx),  feature_cols].to_numpy()
    Y_train   =  alpha_yields.loc[alpha_yields.index.isin(train_idx),  'yield'].to_numpy()
    X_val     =  alpha_yields.loc[~alpha_yields.index.isin(train_idx), feature_cols].to_numpy()
    Y_val     =  alpha_yields.loc[~alpha_yields.index.isin(train_idx), 'yield'].to_numpy()
    print(f"IMPROVED={IMPROVED}  features={len(feature_cols)}  train={len(Y_train)}  val={len(Y_val)}")

    # Fit model and evaluate
    if IMPROVED:
        best_r2  =  -1
        best_cfg =  {}
        for lr_gb, n_est, d_gb in [(0.05, 1000, 7), (0.05, 1500, 8), (0.03, 2000, 8)]:
            model =  GradientBoostingRegressor(
                n_estimators=n_est, max_depth=d_gb, learning_rate=lr_gb,
                min_samples_leaf=5, random_state=42, subsample=0.8
            )
            model.fit(X_train, Y_train)
            Y_val_hat =  model.predict(X_val)
            r2 =  1 - np.mean((Y_val - Y_val_hat)**2) / np.mean((Y_val - np.mean(Y_val))**2)
            print(f"  GB lr={lr_gb} n_est={n_est} depth={d_gb}  R2={r2:.4f}")
            if r2 > best_r2:
                best_r2  =  r2
                best_cfg =  {'learning_rate': lr_gb, 'n_estimators': n_est, 'max_depth': d_gb}
        print(f"\n  Best: {best_cfg}  R2={best_r2:.4f}")
    else:
        best_r2 =  -1
        best_d  =  None
        for d in [1, 5, 10, 20, 50, 100, None]:
            model =  RandomForestRegressor(max_depth=d, n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, Y_train)
            Y_val_hat =  model.predict(X_val)
            r2 =  1 - np.mean((Y_val - Y_val_hat)**2) / np.mean((Y_val - np.mean(Y_val))**2)
            print(f"  max_depth={str(d):>4s}  R2={r2:.4f}")
            if r2 > best_r2:
                best_r2 =  r2
                best_d  =  d
        best_cfg =  {'max_depth': best_d}
        print(f"\n  Best: max_depth={best_d}  R2={best_r2:.4f}")

    # Retrain best model on all municipality data
    X_all =  alpha_yields[feature_cols].to_numpy()
    Y_all =  alpha_yields['yield'].to_numpy()

    if IMPROVED:
        rf =  GradientBoostingRegressor(**best_cfg, min_samples_leaf=5,
                                         random_state=42, subsample=0.8)
    else:
        rf =  RandomForestRegressor(**best_cfg, n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X_all, Y_all)
    print(f"Retrained on {len(Y_all)} obs with {best_cfg}")

    # Predict on ADC-level embeddings
    adc_df =  pd.read_parquet("alpha_earth/alpha_earth_mex_adcs.parquet")
    adc_df['CVE_ENT'] =  adc_df['adcid'].str[:2].astype(int)
    adc_df['CVE_MUN'] =  adc_df['adcid'].str[2:5].astype(int)
    print(f"ADC embeddings: {adc_df.shape}")
    print(f"  {adc_df['adcid'].nunique():,} unique ADCs, {adc_df['year'].nunique()} years")

    ### Fill NaN in feature columns (GradientBoosting can't handle NaN)
    X_adc =  adc_df[feature_cols].fillna(0).to_numpy()
    adc_df['yield_pred'] =  rf.predict(X_adc)

    adc_df[['adcid', 'year', 'yield_pred']].to_parquet("predictions/adc_alpha_earth_preds_maize.parquet", index=False)
    print(f"Saved {len(adc_df):,} predictions to predictions/adc_alpha_earth_preds_maize.parquet")


def run_mexico_multi_crop():
    """Mexico mun-level multi-crop RF/GB, predict on ADCs.

    Uses season-specific SIAP yields (not the combined file) so that
    training labels match evaluation.  IMPROVED=True switches to
    GradientBoosting with location + year features, mirroring the
    mexico_maize pipeline.
    """

    # Crop → season mapping (must match accuracy_metrics.ipynb)
    crop_seasons =  {
        'Avocados': 'Perennial',
        'Wheat':    'Fall-Winter',
        'Sorghum':  'Spring-Summer',
        'Sugar':    'Perennial',
    }

    # Load AEF embeddings (municipality-level)
    alpha_earth =  pd.read_parquet('alpha_earth/alpha_earth_mex_muns.parquet')
    alpha_earth['muncode'] =  (
        alpha_earth['CVE_ENT'].apply(lambda x: add_zeros(x, 2)) +
        alpha_earth['CVE_MUN'].apply(lambda x: add_zeros(x, 3))
    ).astype(int)
    print(f"AEF data: {alpha_earth.shape}")

    # Load season-specific SIAP yields (same file as maize pipeline)
    siap_path =  os.path.join(os.path.expanduser("~"), "Dropbox", "Projects",
                   "Maize_prediction", "Data", "SIAP", "Cleaned",
                   "siap_ag_prod_estimation_by_season.dta")
    siap      =  pd.read_stata(siap_path)
    print(f"SIAP shape: {siap.shape}")

    # Filter to target crops with correct seasons
    target_crops =  list(crop_seasons.keys())
    siap_crops   =  siap[siap['name'].isin(target_crops)].copy()
    siap_crops   =  siap_crops[
        siap_crops.apply(lambda r: r['growing_season'] == crop_seasons[r['name']], axis=1)
    ]
    siap_crops   =  siap_crops[~siap_crops['muncode'].astype(str).str.endswith('000')]
    siap_crops['muncode'] =  siap_crops['muncode'].astype(int)
    siap_crops   =  siap_crops[siap_crops['ha_planted'] > 0].copy()
    siap_crops['yield']   =  siap_crops['q'] / siap_crops['ha_planted']
    siap_crops   =  siap_crops[siap_crops['yield'].notna() & (siap_crops['yield'] > 0)]

    for crop in target_crops:
        sub =  siap_crops[siap_crops['name'] == crop]
        print(f"{crop} ({crop_seasons[crop]}): {len(sub)} obs")
    print(f"\nTotal: {len(siap_crops)} obs")

    # Feature columns
    if IMPROVED:
        feature_cols =  embed_cols + ['CVE_ENT', 'CVE_MUN', 'year']
    else:
        feature_cols =  embed_cols

    np.random.seed(42)

    # Train models per crop
    results =  {}

    for crop in target_crops:
        print(f"\n{'='*60}")
        print(f"Training {'GB' if IMPROVED else 'RF'} for: {crop} ({crop_seasons[crop]})")
        print(f"{'='*60}")

        crop_yields =  siap_crops[siap_crops['name'] == crop][['muncode', 'year', 'yield']].copy()
        merged      =  pd.merge(alpha_earth, crop_yields, on=['muncode', 'year'])
        print(f"Merged obs: {len(merged)}  features: {len(feature_cols)}")

        if len(merged) < 100:
            print(f"Skipping {crop}: too few observations after merge")
            continue

        n_train   =  int(len(merged) * 0.8)
        train_idx =  np.random.choice(merged.index, n_train, replace=False)
        X_train   =  merged.loc[merged.index.isin(train_idx),  feature_cols].to_numpy()
        Y_train   =  merged.loc[merged.index.isin(train_idx),  'yield'].to_numpy()
        X_val     =  merged.loc[~merged.index.isin(train_idx), feature_cols].to_numpy()
        Y_val     =  merged.loc[~merged.index.isin(train_idx), 'yield'].to_numpy()
        print(f"  train={len(Y_train)}  val={len(Y_val)}")

        if IMPROVED:
            best_r2  =  -np.inf
            best_cfg =  {}
            for lr_gb, n_est, d_gb in [(0.05, 1000, 7), (0.05, 1500, 8), (0.03, 2000, 8)]:
                model =  GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=d_gb, learning_rate=lr_gb,
                    min_samples_leaf=5, random_state=42, subsample=0.8
                )
                model.fit(X_train, Y_train)
                Y_hat =  model.predict(X_val)
                r2    =  1 - np.mean((Y_val - Y_hat)**2) / np.mean((Y_val - np.mean(Y_val))**2)
                print(f"  GB lr={lr_gb} n_est={n_est} depth={d_gb}  R2={r2:.4f}")
                if r2 > best_r2:
                    best_r2  =  r2
                    best_cfg =  {'learning_rate': lr_gb, 'n_estimators': n_est, 'max_depth': d_gb}
            print(f"  Best: {best_cfg}  R2={best_r2:.4f}")
        else:
            best_r2 =  -np.inf
            best_d  =  None
            for d in [1, 5, 10, 20, 50, 100, None]:
                rf    =  RandomForestRegressor(max_depth=d, n_estimators=100,
                                               random_state=42, n_jobs=-1)
                rf.fit(X_train, Y_train)
                Y_hat =  rf.predict(X_val)
                r2    =  1 - np.mean((Y_val - Y_hat)**2) / np.mean((Y_val - np.mean(Y_val))**2)
                print(f"  max_depth={str(d):>4s}  R2={r2:.4f}")
                if r2 > best_r2:
                    best_r2 =  r2
                    best_d  =  d
            best_cfg =  {'max_depth': best_d}
            print(f"  Best: max_depth={best_d}, R2={best_r2:.4f}")

        # Retrain on all data
        X_all =  merged[feature_cols].to_numpy()
        Y_all =  merged['yield'].to_numpy()

        if IMPROVED:
            final_model =  GradientBoostingRegressor(**best_cfg, min_samples_leaf=5,
                                                      random_state=42, subsample=0.8)
        else:
            final_model =  RandomForestRegressor(**best_cfg, n_estimators=100,
                                                  random_state=42, n_jobs=-1)
        final_model.fit(X_all, Y_all)

        results[crop] =  {
            'model':    final_model,
            'best_cfg': best_cfg,
            'best_r2':  best_r2,
            'n_train':  n_train,
            'n_merged': len(merged),
        }

    # Summary
    print(f"\n{'Crop':<12s} {'Season':<16s} {'N obs':>8s} {'Best config':>20s} {'Val R2':>8s}")
    print('-' * 68)
    for crop, res in results.items():
        print(f"{crop:<12s} {crop_seasons[crop]:<16s} {res['n_merged']:>8d} "
              f"{str(res['best_cfg']):>20s} {res['best_r2']:>8.4f}")

    # Predict on ADC polygons
    adc_df =  pd.read_parquet('alpha_earth/alpha_earth_mex_adcs.parquet')
    adc_df['CVE_ENT'] =  adc_df['adcid'].str[:2].astype(int)
    adc_df['CVE_MUN'] =  adc_df['adcid'].str[2:5].astype(int)
    print(f"\nADC data: {adc_df.shape}  ({adc_df['adcid'].nunique():,} ADCs)")

    X_adc =  adc_df[feature_cols].fillna(0).to_numpy()

    for crop, res in results.items():
        preds      =  res['model'].predict(X_adc)
        out        =  adc_df[['adcid', 'year']].copy()
        crop_label =  crop.lower().replace(' ', '_')
        out[f'yield_pred_{crop_label}'] =  preds

        out_path =  os.path.join('predictions', f'adc_alpha_earth_preds_{crop_label}.parquet')
        out.to_parquet(out_path, index=False)
        print(f"Saved {crop}: {len(out):,} rows to {out_path}")


# ===== Main dispatch =====

if __name__ == '__main__':
    modes =  {
        'mexico_maize':      run_mexico_maize,
        'mexico_multi_crop': run_mexico_multi_crop,
    }
    if MODE not in modes:
        raise ValueError(f"Unknown MODE '{MODE}'. Choose from: {list(modes.keys())}")

    print(f"Running mode: {MODE}")
    print(f"{'='*60}\n")
    modes[MODE]()
