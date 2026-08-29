"""
agg_nn_histogram.py
===================
Train Agg-NN using distributional AEF features extracted via GEE
(percentiles + stdDev at 10m resolution).

ADC-level features: 64 dims × 6 stats (p10, p25, p50, p75, p90, stdDev) = 384
Municipality-level features: same 384 (optional, merged as context)
Auxiliary: irrig_share, log_adc_area, agland_share = 3

Trains 5-seed ensemble with Config B, evaluates vs INEGI 2022 census.

Usage:
  conda activate ML_env
  python3 agg_nn_histogram.py --epochs 300
  python3 agg_nn_histogram.py --epochs 300 --use_mun_hist   # include mun context

Author: Jay Sayre
Date: 2026-02-24
"""

import os
os.environ['OMP_NUM_THREADS']      =  '4'
os.environ['MKL_NUM_THREADS']      =  '4'
os.environ['OPENBLAS_NUM_THREADS'] =  '4'

import sys
import time
import argparse
import numpy  as np
import pandas as pd
import torch
torch.set_num_threads(4)
import torch.nn as nn
from   sklearn.preprocessing import StandardScaler

### ------------------------------------------------------------------ ###
### Directories
### ------------------------------------------------------------------ ###

home_dir      =  os.path.expanduser("~")
proj_dir      =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir      =  os.path.join(proj_dir, "Data")
pred_dir      =  os.path.join(data_dir, "predictions")
aef_dir       =  os.path.join(data_dir, "alpha_earth")
agland_path   =  os.path.join(data_dir, "SIAP_agland", "Output",
                               "2007_adcs_agland_area.csv")
inegi_dir     =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
crop_sub_dir  =  os.path.join(home_dir, "Dropbox", "Projects",
                               "Maize_prediction")
siap_dir      =  os.path.join(crop_sub_dir, "Data", "SIAP", "Cleaned")


### ------------------------------------------------------------------ ###
### Model + helpers (same as agg_nn_sweep.py)
### ------------------------------------------------------------------ ###

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_blocks, dropout=0.2):
        super().__init__()
        self.input_proj =  nn.Linear(input_dim, hidden_dim)
        self.relu       =  nn.ReLU()
        self.blocks     =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.head =  nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x =  self.relu(self.input_proj(x))
        for block in self.blocks:
            x =  self.relu(x + block(x))
        return self.head(x).squeeze(-1)


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

def build_flat_batch(keys, grouped, features, weights, rf_preds, target_dict,
                     max_adcs=200):
    all_X, all_W, all_RF, all_idx, all_Y =  [], [], [], [], []
    for i, key in enumerate(keys):
        idxs =  np.asarray(grouped[key])
        if len(idxs) > max_adcs:
            idxs =  np.random.choice(idxs, max_adcs, replace=False)
        all_X.append(features[idxs])
        all_W.append(weights[idxs])
        all_RF.append(rf_preds[idxs])
        all_idx.append(np.full(len(idxs), i, dtype=np.int64))
        all_Y.append(target_dict[key])
    return (torch.from_numpy(np.concatenate(all_X)).float(),
            torch.from_numpy(np.concatenate(all_W)).float(),
            torch.from_numpy(np.concatenate(all_RF)).float(),
            torch.from_numpy(np.concatenate(all_idx)).long(),
            torch.tensor(all_Y, dtype=torch.float32))

def aggregate_predictions(preds, weights, mun_idx, n_muns):
    w_preds    =  preds * weights
    sum_wpreds =  torch.zeros(n_muns, device=preds.device)
    sum_wpreds.scatter_add_(0, mun_idx, w_preds)
    sum_w =  torch.zeros(n_muns, device=preds.device)
    sum_w.scatter_add_(0, mun_idx, weights)
    return sum_wpreds / sum_w.clamp(min=1e-8)

def train_model(model, train_keys, val_keys, grouped, features, weights,
                rf_preds, target_dict, cfg, device, epochs, verbose=True):
    optimizer =  torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                   weight_decay=cfg['weight_decay'])
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)
    best_val_loss =  float('inf')
    best_state    =  None
    patience      =  40
    wait          =  0
    stopped_epoch =  0

    for epoch in range(epochs):
        model.train()
        shuffled_keys =  list(train_keys)
        np.random.shuffle(shuffled_keys)
        train_losses =  []
        for batch_start in range(0, len(shuffled_keys), cfg['batch_size']):
            batch_keys =  shuffled_keys[batch_start:batch_start + cfg['batch_size']]
            X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                batch_keys, grouped, features, weights, rf_preds,
                target_dict, max_adcs=cfg['max_adcs'])
            X_flat, W_flat, RF_flat =  X_flat.to(device), W_flat.to(device), RF_flat.to(device)
            mun_idx, Y =  mun_idx.to(device), Y.to(device)
            if X_flat.shape[0] < 2: continue
            optimizer.zero_grad()
            residuals =  torch.tanh(model(X_flat)) * cfg['max_residual']
            adc_pred  =  (RF_flat + residuals).clamp(min=0)
            mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx, len(batch_keys))
            mse_loss  =  nn.functional.mse_loss(mun_pred, Y)
            loss      =  mse_loss + cfg['residual_lambda'] * (residuals ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(mse_loss.item())

        model.eval()
        val_losses =  []
        with torch.no_grad():
            for batch_start in range(0, len(val_keys), cfg['batch_size']):
                batch_keys =  val_keys[batch_start:batch_start + cfg['batch_size']]
                X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                    batch_keys, grouped, features, weights, rf_preds,
                    target_dict, max_adcs=10000)
                X_flat, W_flat, RF_flat =  X_flat.to(device), W_flat.to(device), RF_flat.to(device)
                mun_idx, Y =  mun_idx.to(device), Y.to(device)
                if X_flat.shape[0] < 2: continue
                residuals =  torch.tanh(model(X_flat)) * cfg['max_residual']
                adc_pred  =  (RF_flat + residuals).clamp(min=0)
                mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx, len(batch_keys))
                val_losses.append(nn.functional.mse_loss(mun_pred, Y).item())

        avg_train =  np.mean(train_losses)
        avg_val   =  np.mean(val_losses) if val_losses else float('inf')
        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            best_val_loss, best_state, wait =  avg_val, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            marker =  " *"
        else:
            wait += 1
            marker =  ""
        stopped_epoch =  epoch + 1
        if verbose and ((epoch + 1) % 10 == 0 or wait == 0):
            lr =  optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}: train={avg_train:.4f}  val={avg_val:.4f}  lr={lr:.1e}{marker}")
        if wait >= patience:
            if verbose: print(f"    Early stop at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    model.eval()
    return best_state, best_val_loss, stopped_epoch

def generate_adc_predictions(model, features, rf_preds, max_residual, device,
                             chunk_size=50000):
    model.eval()
    all_preds =  np.zeros(len(features))
    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            end =  min(start + chunk_size, len(features))
            X_chunk  =  torch.from_numpy(features[start:end]).to(device)
            RF_chunk =  torch.from_numpy(rf_preds[start:end]).to(device)
            residuals =  torch.tanh(model(X_chunk)) * max_residual
            adc_pred  =  (RF_chunk + residuals).clamp(min=0, max=25)
            all_preds[start:end] =  adc_pred.cpu().numpy()
    return all_preds


### ------------------------------------------------------------------ ###
### Config B (best from sweep)
### ------------------------------------------------------------------ ###

CFG =  {
    'hidden': 256, 'blocks': 2, 'dropout': 0.2, 'max_residual': 5,
    'residual_lambda': 0.01, 'batch_size': 64,
    'lr': 1e-4, 'weight_decay': 1e-4, 'max_adcs': 200,
}

ENSEMBLE_SEEDS =  [42, 123, 456, 789, 1024]


### ------------------------------------------------------------------ ###
### Main
### ------------------------------------------------------------------ ###

def main():
    parser =  argparse.ArgumentParser(
        description="Agg-NN with GEE histogram features")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_mun_hist', action='store_true',
                        help='Include mun-level histogram features as context')
    args =  parser.parse_args()

    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  Agg-NN with GEE Histogram Features")
    print(f"  Device: {device}, Epochs: {args.epochs}")
    print(f"  Mun-level context: {args.use_mun_hist}")
    print("=" * 70)

    ### ---------------------------------------------------------------- ###
    ### Load data
    ### ---------------------------------------------------------------- ###

    print("\n--- Loading data ---")
    t0 =  time.time()

    # ADC-level histogram features (384 cols)
    adc_hist =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"))
    print(f"  ADC hist parquet: {len(adc_hist):,} rows, {len(adc_hist.columns)} cols")

    # Also load mean embeddings (needed for RF baseline merge + adcid/year/muncode)
    aef_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
    aef_mean['muncode'] =  aef_mean['adcid'].str[:5]

    # Build feature column names for histogram
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

    print(f"  Histogram feature cols: {len(hist_cols)}")

    # Merge histogram features onto mean-based DataFrame (by adcid + year)
    # adc_hist has 'adcid' and 'year' columns
    aef =  aef_mean.merge(
        adc_hist[['adcid', 'year'] + hist_cols],
        on=['adcid', 'year'], how='inner'
    )
    print(f"  After merge (inner): {len(aef):,} ADC-years")

    # Mun-level histogram features (optional)
    mun_hist_cols =  []
    if args.use_mun_hist:
        mun_hist =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
        mun_hist_cols =  [f"mun_{c}" for c in hist_cols]
        rename_map =  {c: f"mun_{c}" for c in hist_cols}
        mun_hist =  mun_hist.rename(columns=rename_map)
        # mun_hist has muncode and year
        aef =  aef.merge(
            mun_hist[['muncode', 'year'] + mun_hist_cols],
            on=['muncode', 'year'], how='left'
        )
        # Fill NaN mun features with 0
        for c in mun_hist_cols:
            aef[c] =  aef[c].fillna(0.0)
        print(f"  Mun hist features merged: {len(mun_hist_cols)}")

    # Agland auxiliary features
    agland =  pd.read_csv(agland_path)
    agland['irrig_share']  =  (agland['siap_irrig_area'] /
                                agland['siap_agland_area']).replace(
                                    [np.inf, -np.inf], np.nan).fillna(0.0)
    agland['log_adc_area'] =  np.log1p(agland['adc_area'])
    agland['agland_share'] =  (agland['siap_agland_area'] /
                                agland['adc_area']).replace(
                                    [np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
    agland =  agland[['adcid', 'irrig_share', 'log_adc_area', 'agland_share',
                       'siap_agland_area']]

    aef =  aef.merge(agland, on='adcid', how='left')
    aef['irrig_share']      =  aef['irrig_share'].fillna(0.0)
    aef['log_adc_area']     =  aef['log_adc_area'].fillna(aef['log_adc_area'].median())
    aef['agland_share']     =  aef['agland_share'].fillna(0.0)
    aef['siap_agland_area'] =  aef['siap_agland_area'].fillna(0.0)

    # RF baseline (trained on mean embeddings)
    rf_baseline =  pd.read_parquet(
        os.path.join(pred_dir, "adc_alpha_earth_preds_maize.parquet"))
    aef =  aef.merge(rf_baseline[['adcid', 'year', 'yield_pred']].rename(
        columns={'yield_pred': 'yield_pred_rf'}), on=['adcid', 'year'], how='left')
    rf_median            =  aef['yield_pred_rf'].median()
    aef['yield_pred_rf'] =  aef['yield_pred_rf'].fillna(rf_median)

    # SIAP municipal yields
    siap_szn =  pd.read_stata(os.path.join(siap_dir,
                "siap_ag_prod_estimation_by_season.dta"))
    siap_szn['muncode'] =  siap_szn['muncode'].apply(lambda x: str(int(x)).zfill(5))
    siap_szn['yield']   =  siap_szn['q'] / siap_szn['ha_planted']
    siap_maize =  siap_szn[
        (siap_szn['name'] == 'Maize') &
        (siap_szn['growing_season'] == 'Spring-Summer') &
        (siap_szn['year'] >= 2017)
    ][['muncode', 'year', 'yield']].copy()

    print(f"  AEF merged: {len(aef):,}")
    print(f"  SIAP mun-years: {len(siap_maize):,}")
    print(f"  Data load time: {time.time()-t0:.1f}s")

    ### ---------------------------------------------------------------- ###
    ### Build feature set
    ### ---------------------------------------------------------------- ###

    aux_cols       =  ['irrig_share', 'log_adc_area', 'agland_share']
    all_feat_cols  =  hist_cols + aux_cols + mun_hist_cols

    print(f"\n--- Feature set ---")
    print(f"  ADC histogram:  {len(hist_cols)}")
    print(f"  Auxiliary:      {len(aux_cols)}")
    print(f"  Mun histogram:  {len(mun_hist_cols)}")
    print(f"  Total features: {len(all_feat_cols)}")

    ### ---------------------------------------------------------------- ###
    ### Build mun-year index
    ### ---------------------------------------------------------------- ###

    aef_munyears   =  set(zip(aef['muncode'], aef['year']))
    siap_munyears  =  set(zip(siap_maize['muncode'], siap_maize['year']))
    valid_munyears =  sorted(aef_munyears & siap_munyears)
    siap_dict      =  dict(zip(zip(siap_maize['muncode'], siap_maize['year']),
                                siap_maize['yield']))

    aef =  aef.reset_index(drop=True)
    grouped    =  aef.groupby(['muncode', 'year']).indices
    valid_keys =  [k for k in valid_munyears if k in grouped]
    print(f"  Valid mun-years: {len(valid_keys):,}")

    ### ---------------------------------------------------------------- ###
    ### Extract arrays
    ### ---------------------------------------------------------------- ###

    all_features =  aef[all_feat_cols].to_numpy().astype(np.float32)
    nan_mask     =  np.isnan(all_features)
    if nan_mask.any():
        col_means =  np.nanmean(all_features, axis=0)
        for j in range(all_features.shape[1]):
            all_features[nan_mask[:, j], j] =  col_means[j]
        print(f"  Filled {nan_mask.sum()} NaN values")

    all_weights  =  aef['siap_agland_area'].to_numpy().astype(np.float32)
    all_weights  =  np.where(all_weights > 0, all_weights, 1.0)
    all_rf_preds =  aef['yield_pred_rf'].to_numpy().astype(np.float32)
    input_dim    =  len(all_feat_cols)

    ### ---------------------------------------------------------------- ###
    ### Train/val split
    ### ---------------------------------------------------------------- ###

    np.random.seed(args.seed)
    unique_muns  =  sorted(set(k[0] for k in valid_keys))
    np.random.shuffle(unique_muns)
    n_train_muns =  int(0.8 * len(unique_muns))
    train_muns   =  set(unique_muns[:n_train_muns])
    val_muns     =  set(unique_muns[n_train_muns:])
    train_keys   =  [k for k in valid_keys if k[0] in train_muns]
    val_keys     =  [k for k in valid_keys if k[0] in val_muns]

    print(f"\n  Train: {len(train_keys):,} mun-years ({len(train_muns)} muns)")
    print(f"  Val:   {len(val_keys):,} mun-years ({len(val_muns)} muns)")

    train_row_idxs  =  np.concatenate([np.asarray(grouped[k]) for k in train_keys])
    scaler          =  StandardScaler()
    scaler.fit(all_features[train_row_idxs])
    features_scaled =  scaler.transform(all_features).astype(np.float32)

    ### ---------------------------------------------------------------- ###
    ### Load INEGI 2022 census for evaluation
    ### ---------------------------------------------------------------- ###

    ca2022_path =  os.path.join(inegi_dir,
                   "LM2304-CA22-2025-09-29-superficie_ENTREGA",
                   "adc_land_use_ca22_adc07.dta")
    adc_gt =  pd.read_stata(ca2022_path)
    adc_gt =  adc_gt[adc_gt['name'] == 'Maize']
    adc_gt =  adc_gt[['adc', 'muncode', 'yield']].copy()

    def evaluate_vs_inegi(preds_arr, label=""):
        nn_2022         =  aef[aef['year'] == 2022][['adcid']].copy()
        nn_2022['pred'] =  preds_arr[aef[aef['year'] == 2022].index]
        nn_2022['adc']  =  nn_2022['adcid'].str.replace('-', '', regex=False)
        eval_df =  adc_gt.merge(nn_2022[['adc', 'pred']], on='adc', how='left')
        return compute_all_metrics(eval_df, 'yield', 'pred', group_col='muncode')

    ### ================================================================ ###
    ### Ensemble training (Config B, 5 seeds)
    ### ================================================================ ###

    cfg =  CFG.copy()

    print("\n" + "=" * 70)
    print(f"  Ensemble Training ({input_dim} features, {len(ENSEMBLE_SEEDS)} seeds)")
    print(f"  Config: hidden={cfg['hidden']}, blocks={cfg['blocks']}, "
          f"dropout={cfg['dropout']}, max_res={cfg['max_residual']}")
    print("=" * 70)

    t_ens_start    =  time.time()
    ensemble_preds =  []

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n  Seed {seed} ({i+1}/{len(ENSEMBLE_SEEDS)})")

        np.random.seed(seed)
        torch.manual_seed(seed)

        model =  ResidualMLP(input_dim, cfg['hidden'], cfg['blocks'],
                              dropout=cfg['dropout']).to(device)

        _, val_mse, stop_ep =  train_model(
            model, train_keys, val_keys, grouped, features_scaled, all_weights,
            all_rf_preds, siap_dict, cfg, device, args.epochs, verbose=True)

        print(f"    Val MSE: {val_mse:.4f}, stopped at epoch {stop_ep}")

        # Retrain on all data
        print(f"    Retraining on all data ({stop_ep} epochs)...")
        all_row_idxs  =  np.concatenate([np.asarray(grouped[k]) for k in valid_keys])
        scaler_full   =  StandardScaler()
        scaler_full.fit(all_features[all_row_idxs])
        features_full =  scaler_full.transform(all_features).astype(np.float32)

        final_model =  ResidualMLP(input_dim, cfg['hidden'], cfg['blocks'],
                                    dropout=cfg['dropout']).to(device)
        np.random.seed(seed)
        torch.manual_seed(seed)

        optimizer =  torch.optim.Adam(final_model.parameters(), lr=cfg['lr'],
                                       weight_decay=cfg['weight_decay'])
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5, min_lr=1e-6)

        all_valid_keys =  list(valid_keys)
        for ep in range(stop_ep):
            final_model.train()
            np.random.shuffle(all_valid_keys)
            epoch_losses =  []
            for batch_start in range(0, len(all_valid_keys), cfg['batch_size']):
                batch_keys =  all_valid_keys[batch_start:batch_start + cfg['batch_size']]
                X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                    batch_keys, grouped, features_full, all_weights,
                    all_rf_preds, siap_dict, max_adcs=cfg['max_adcs'])
                X_flat, W_flat, RF_flat =  X_flat.to(device), W_flat.to(device), RF_flat.to(device)
                mun_idx, Y =  mun_idx.to(device), Y.to(device)
                if X_flat.shape[0] < 2: continue
                optimizer.zero_grad()
                residuals =  torch.tanh(final_model(X_flat)) * cfg['max_residual']
                adc_pred  =  (RF_flat + residuals).clamp(min=0)
                mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx, len(batch_keys))
                mse_loss  =  nn.functional.mse_loss(mun_pred, Y)
                loss      =  mse_loss + cfg['residual_lambda'] * (residuals ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(mse_loss.item())
            avg_loss =  np.mean(epoch_losses)
            scheduler.step(avg_loss)
            if (ep + 1) % 50 == 0:
                print(f"      Retrain epoch {ep+1:3d}: loss={avg_loss:.4f}")

        preds =  generate_adc_predictions(final_model, features_full, all_rf_preds,
                                           cfg['max_residual'], device)
        ensemble_preds.append(preds)

        inegi_m =  evaluate_vs_inegi(preds)
        print(f"    INEGI R²={inegi_m['R2']:.3f}, Btw={inegi_m['Between_R2']:.3f}, "
              f"Wtn={inegi_m['Within_R2']:.3f}")

    ensemble_avg =  np.mean(ensemble_preds, axis=0)
    t_ens =  time.time() - t_ens_start

    ### ================================================================ ###
    ### Results
    ### ================================================================ ###

    print("\n" + "=" * 70)
    print("  Results: ADC-level vs INEGI 2022")
    print("=" * 70)

    # Histogram ensemble
    hist_m =  evaluate_vs_inegi(ensemble_avg)
    feat_label =  f"Hist Agg-NN ({input_dim} feat)"
    print(f"\n  {feat_label}:")
    print(f"    R²={hist_m['R2']:.3f}, Between={hist_m['Between_R2']:.3f}, "
          f"Within={hist_m['Within_R2']:.3f}, RMSE={hist_m['RMSE']:.3f}, "
          f"N={hist_m['N']:,}")

    # RF baseline
    rf_m =  evaluate_vs_inegi(all_rf_preds)
    print(f"\n  AEF RF baseline:")
    print(f"    R²={rf_m['R2']:.3f}, Between={rf_m['Between_R2']:.3f}, "
          f"Within={rf_m['Within_R2']:.3f}, RMSE={rf_m['RMSE']:.3f}")

    # Load base Agg-NN predictions for comparison
    base_mlp_path =  os.path.join(pred_dir, "adc_mlp_yield_preds.csv")
    if os.path.exists(base_mlp_path):
        base_mlp =  pd.read_csv(base_mlp_path)
        base_mlp =  base_mlp[base_mlp['year'] == 2022]
        base_mlp['adc'] =  base_mlp['adcid'].str.replace('-', '', regex=False)
        base_eval =  adc_gt.merge(base_mlp[['adc', 'pred_yield']], on='adc', how='left')
        base_m =  compute_all_metrics(base_eval, 'yield', 'pred_yield', group_col='muncode')
        print(f"\n  Base Agg-NN (67 feat, mean embeds):")
        print(f"    R²={base_m['R2']:.3f}, Between={base_m['Between_R2']:.3f}, "
              f"Within={base_m['Within_R2']:.3f}, RMSE={base_m['RMSE']:.3f}")

    print(f"\n  Ensemble training time: {t_ens/60:.1f} min")
    print(f"  Total runtime: {(time.time()-t0)/60:.1f} min")

    ### Summary table
    print(f"\n{'Model':<40s} {'R²':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
    print("-" * 70)
    rows =  [("AEF RF (baseline)", rf_m), (feat_label, hist_m)]
    if os.path.exists(base_mlp_path):
        rows.insert(1, ("Agg-NN base (67 feat, mean)", base_m))
    for name, m in rows:
        print(f"{name:<40s} {m['R2']:>6.3f} {m['Between_R2']:>6.3f} "
              f"{m['Within_R2']:>7.3f} {m['RMSE']:>6.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
