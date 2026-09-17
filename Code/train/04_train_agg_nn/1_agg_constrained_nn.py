"""
1_agg_constrained_nn.py
=====================
Phase 2 & 3: Aggregation-Constrained Neural Network for within-mun R².

Trains a PyTorch model that predicts RESIDUAL ADJUSTMENTS on top of the
existing RF baseline predictions. ADC-level residuals are bounded via tanh,
aggregated to municipality level via area-weighted mean, and loss is computed
against SIAP municipal yields.

Architecture:
  - ResidualMLP: 128-dim, 2 residual blocks, dropout=0.2
  - Input: 67 features per ADC (64 AEF + irrig_share + log_adc_area + agland_share)
  - Output: tanh-bounded residual, clipped to max_residual (default ±5 t/ha)
  - Final prediction: RF_pred + residual, clipped to [0, 25]
  - Loss: MSE(mun_pred, siap_yield) + lambda * mean(residual²)

Phase 3 extensions (activated via --phase3 flag):
  - Embedding deviation: ADC_embed - mun_mean_embed (64 dims)
  - Temporal delta: embed[year] - embed[year-1] (64 dims)

Usage:
  conda activate ML_env
  python3 1_agg_constrained_nn.py --epochs 300
  python3 1_agg_constrained_nn.py --phase3 --epochs 300

Author: Jay Sayre
Date: 2026-02-22
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
### Model Definition (reused from NN.ipynb)
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


### ------------------------------------------------------------------ ###
### Evaluation metrics
### ------------------------------------------------------------------ ###

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


### ------------------------------------------------------------------ ###
### Efficient flat-batch training functions
### ------------------------------------------------------------------ ###

def build_flat_batch(keys, grouped, features, weights, rf_preds, siap_dict,
                     max_adcs=200):
    """Build a flat batch: concatenate all ADCs from selected mun-years.

    Returns:
        X_flat:   (N_total_adcs, n_feat) tensor
        W_flat:   (N_total_adcs,) tensor of area weights
        RF_flat:  (N_total_adcs,) tensor of RF baseline predictions
        mun_idx:  (N_total_adcs,) tensor mapping each ADC to its mun-year index
        Y:        (n_mun_years,) tensor of SIAP yields
    """
    all_X   =  []
    all_W   =  []
    all_RF  =  []
    all_idx =  []
    all_Y   =  []

    for i, key in enumerate(keys):
        idxs =  np.asarray(grouped[key])
        if len(idxs) > max_adcs:
            idxs =  np.random.choice(idxs, max_adcs, replace=False)
        all_X.append(features[idxs])
        all_W.append(weights[idxs])
        all_RF.append(rf_preds[idxs])
        all_idx.append(np.full(len(idxs), i, dtype=np.int64))
        all_Y.append(siap_dict[key])

    X_flat  =  torch.from_numpy(np.concatenate(all_X)).float()
    W_flat  =  torch.from_numpy(np.concatenate(all_W)).float()
    RF_flat =  torch.from_numpy(np.concatenate(all_RF)).float()
    mun_idx =  torch.from_numpy(np.concatenate(all_idx)).long()
    Y       =  torch.tensor(all_Y, dtype=torch.float32)

    return X_flat, W_flat, RF_flat, mun_idx, Y


def aggregate_predictions(preds, weights, mun_idx, n_muns):
    """Area-weighted mean aggregation using scatter operations."""
    w_preds    =  preds * weights
    sum_wpreds =  torch.zeros(n_muns, device=preds.device)
    sum_wpreds.scatter_add_(0, mun_idx, w_preds)
    sum_w =  torch.zeros(n_muns, device=preds.device)
    sum_w.scatter_add_(0, mun_idx, weights)
    return sum_wpreds / sum_w.clamp(min=1e-8)


def compute_var_penalty(preds, mun_idx, n_muns):
    """Compute mean within-mun prediction variance (for encouragement)."""
    sum_preds =  torch.zeros(n_muns, device=preds.device)
    sum_preds.scatter_add_(0, mun_idx, preds)
    counts =  torch.zeros(n_muns, device=preds.device)
    counts.scatter_add_(0, mun_idx, torch.ones_like(preds))
    mean_pred =  sum_preds / counts.clamp(min=1)
    dev       =  preds - mean_pred[mun_idx]
    sum_dev2  =  torch.zeros(n_muns, device=preds.device)
    sum_dev2.scatter_add_(0, mun_idx, dev ** 2)
    var_per_mun =  sum_dev2 / counts.clamp(min=1)
    return var_per_mun.mean()


### ------------------------------------------------------------------ ###
### Training loop
### ------------------------------------------------------------------ ###

def train_model(model, train_keys, val_keys, grouped, features, weights,
                rf_preds, siap_dict, args, device):
    """Train with early stopping. Returns best model state dict and stop epoch."""

    optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)

    best_val_loss =  float('inf')
    best_state    =  None
    patience      =  40
    wait          =  0
    stopped_epoch =  0

    for epoch in range(args.epochs):
        t_epoch =  time.time()

        # --- Train ---
        model.train()
        shuffled_keys =  list(train_keys)
        np.random.shuffle(shuffled_keys)
        train_losses =  []

        for batch_start in range(0, len(shuffled_keys), args.batch_size):
            batch_keys =  shuffled_keys[batch_start:batch_start + args.batch_size]
            X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                batch_keys, grouped, features, weights, rf_preds,
                siap_dict, max_adcs=args.max_adcs)

            X_flat  =  X_flat.to(device)
            W_flat  =  W_flat.to(device)
            RF_flat =  RF_flat.to(device)
            mun_idx =  mun_idx.to(device)
            Y       =  Y.to(device)

            if X_flat.shape[0] < 2:
                continue

            optimizer.zero_grad()
            residuals =  torch.tanh(model(X_flat)) * args.max_residual
            adc_pred  =  (RF_flat + residuals).clamp(min=0)
            mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx,
                                                len(batch_keys))
            mse_loss  =  nn.functional.mse_loss(mun_pred, Y)

            loss =  mse_loss + args.residual_lambda * (residuals ** 2).mean()
            if args.var_lambda > 0:
                loss =  loss - args.var_lambda * compute_var_penalty(
                    adc_pred, mun_idx, len(batch_keys))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(mse_loss.item())

        # --- Validate ---
        model.eval()
        val_losses =  []
        with torch.no_grad():
            for batch_start in range(0, len(val_keys), args.batch_size):
                batch_keys =  val_keys[batch_start:batch_start + args.batch_size]
                X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                    batch_keys, grouped, features, weights, rf_preds,
                    siap_dict, max_adcs=10000)

                X_flat  =  X_flat.to(device)
                W_flat  =  W_flat.to(device)
                RF_flat =  RF_flat.to(device)
                mun_idx =  mun_idx.to(device)
                Y       =  Y.to(device)

                if X_flat.shape[0] < 2:
                    continue

                residuals =  torch.tanh(model(X_flat)) * args.max_residual
                adc_pred  =  (RF_flat + residuals).clamp(min=0)
                mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx,
                                                    len(batch_keys))
                val_losses.append(nn.functional.mse_loss(mun_pred, Y).item())

        avg_train  =  np.mean(train_losses)
        avg_val    =  np.mean(val_losses) if val_losses else float('inf')
        scheduler.step(avg_val)
        current_lr =  optimizer.param_groups[0]['lr']

        if avg_val < best_val_loss:
            best_val_loss =  avg_val
            best_state    =  {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait   =  0
            marker =  " *"
        else:
            wait  += 1
            marker =  ""

        epoch_time =  time.time() - t_epoch
        if (epoch + 1) % 5 == 0 or wait == 0:
            print(f"  Epoch {epoch+1:3d}: train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={current_lr:.1e}  [{epoch_time:.1f}s]{marker}")

        stopped_epoch =  epoch + 1
        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val MSE: {best_val_loss:.4f}")
    return best_state, stopped_epoch


### ------------------------------------------------------------------ ###
### Main
### ------------------------------------------------------------------ ###

def main():
    parser =  argparse.ArgumentParser(description="Aggregation-Constrained NN")
    parser.add_argument('--phase3', action='store_true',
                        help='Enable Phase 3 features (embed deviation + temporal delta)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Max training epochs (default: 300)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--n_blocks', type=int, default=2,
                        help='Number of residual blocks (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size in mun-years (default: 128)')
    parser.add_argument('--var_lambda', type=float, default=0.0,
                        help='Variance encouragement weight (default: 0.0)')
    parser.add_argument('--max_adcs', type=int, default=200,
                        help='Max ADCs per municipality in training (default: 200)')
    parser.add_argument('--max_residual', type=float, default=5.0,
                        help='Max residual adjustment in t/ha (default: 5.0)')
    parser.add_argument('--residual_lambda', type=float, default=0.01,
                        help='L2 penalty on residuals (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--inner_es', action='store_true',
                        help='Hold 10%% of the TRAINING municipalities for early '
                             'stopping (validation municipalities never influence '
                             'any training choice), refit on the full training set '
                             'for the selected epoch count, write '
                             '*_inner_es.parquet, and exit (no full-data retrain).')
    parser.add_argument('--seed', type=int, default=42)
    args =  parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device      =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phase_label =  "Phase 2+3" if args.phase3 else "Phase 2"

    print("=" * 70)
    print(f"  Aggregation-Constrained NN ({phase_label}) — Residual Mode")
    print(f"  Device: {device}")
    print(f"  Config: hidden={args.hidden_dim}, blocks={args.n_blocks}, "
          f"lr={args.lr}, batch={args.batch_size}")
    print(f"  Residual: max={args.max_residual}, lambda={args.residual_lambda}, "
          f"var_lambda={args.var_lambda}")
    print("=" * 70)

    ### ---------------------------------------------------------------- ###
    ### Load data
    ### ---------------------------------------------------------------- ###

    print("\n--- Loading data ---")
    t0 =  time.time()

    # ADC-level AEF embeddings
    aef =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
    aef['muncode'] =  aef['adcid'].str[:5]
    feat_cols =  [f"A{str(i).zfill(2)}" for i in range(64)]

    # ADC auxiliary features (agland)
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

    # Merge
    aef =  aef.merge(agland, on='adcid', how='left')
    aef['irrig_share']      =  aef['irrig_share'].fillna(0.0)
    aef['log_adc_area']     =  aef['log_adc_area'].fillna(aef['log_adc_area'].median())
    aef['agland_share']     =  aef['agland_share'].fillna(0.0)
    aef['siap_agland_area'] =  aef['siap_agland_area'].fillna(0.0)

    # RF baseline predictions
    rf_baseline =  pd.read_parquet(
        os.path.join(pred_dir, "adc_alpha_earth_preds_maize.parquet"))
    aef =  aef.merge(rf_baseline[['adcid', 'year', 'yield_pred']].rename(
        columns={'yield_pred': 'yield_pred_rf'}), on=['adcid', 'year'], how='left')
    rf_median          =  aef['yield_pred_rf'].median()
    aef['yield_pred_rf'] =  aef['yield_pred_rf'].fillna(rf_median)
    n_rf_matched       =  (aef['yield_pred_rf'] != rf_median).sum()
    print(f"  AEF ADC embeddings: {len(aef):,}")
    print(f"  RF baseline merged: {n_rf_matched:,} matched, "
          f"{(aef['yield_pred_rf'] == rf_median).sum():,} filled with median")

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

    print(f"  SIAP maize mun-years: {len(siap_maize):,}")
    print(f"  Data load time: {time.time()-t0:.1f}s")

    ### ---------------------------------------------------------------- ###
    ### Build features
    ### ---------------------------------------------------------------- ###

    print("\n--- Building features ---")

    aux_cols      =  ['irrig_share', 'log_adc_area', 'agland_share']
    all_feat_cols =  feat_cols + aux_cols  # 67 features

    if args.phase3:
        print("  Computing Phase 3 features...")
        t1 =  time.time()

        # Embedding deviation: ADC_embed - mun_mean_embed
        mun_year_means =  aef.groupby(['muncode', 'year'])[feat_cols].transform('mean')
        dev_cols =  [f'{c}_dev' for c in feat_cols]
        for i, c in enumerate(feat_cols):
            aef[dev_cols[i]] =  aef[c] - mun_year_means[c]

        # Temporal delta: embed[year] - embed[year-1]
        aef =  aef.sort_values(['adcid', 'year'])
        delta_cols =  [f'{c}_delta' for c in feat_cols]
        for i, c in enumerate(feat_cols):
            aef[delta_cols[i]] =  aef.groupby('adcid')[c].diff().fillna(0.0)

        all_feat_cols =  feat_cols + aux_cols + dev_cols + delta_cols
        print(f"  Phase 3 features: {len(all_feat_cols)} "
              f"(64 embed + 3 aux + 64 dev + 64 delta)  [{time.time()-t1:.1f}s]")
    else:
        print(f"  Phase 2 features: {len(all_feat_cols)} (64 embed + 3 aux)")

    ### ---------------------------------------------------------------- ###
    ### Build mun-year index
    ### ---------------------------------------------------------------- ###

    aef_munyears   =  set(zip(aef['muncode'], aef['year']))
    siap_munyears  =  set(zip(siap_maize['muncode'], siap_maize['year']))
    valid_munyears =  sorted(aef_munyears & siap_munyears)

    siap_dict =  dict(zip(zip(siap_maize['muncode'], siap_maize['year']),
                          siap_maize['yield']))

    aef =  aef.reset_index(drop=True)
    grouped    =  aef.groupby(['muncode', 'year']).indices
    valid_keys =  [k for k in valid_munyears if k in grouped]
    print(f"  Valid mun-years for training: {len(valid_keys):,}")

    ### ---------------------------------------------------------------- ###
    ### Extract and clean feature arrays
    ### ---------------------------------------------------------------- ###

    print("\n--- Extracting feature arrays ---")
    t0 =  time.time()

    all_features =  aef[all_feat_cols].to_numpy().astype(np.float32)
    # Fill any NaN in features (32 ADC-years have missing embeddings)
    nan_mask =  np.isnan(all_features)
    if nan_mask.any():
        col_means =  np.nanmean(all_features, axis=0)
        for j in range(all_features.shape[1]):
            all_features[nan_mask[:, j], j] =  col_means[j]
        print(f"  Filled {nan_mask.sum()} NaN values in features")

    all_weights =  aef['siap_agland_area'].to_numpy().astype(np.float32)
    all_weights =  np.where(all_weights > 0, all_weights, 1.0)

    all_rf_preds =  aef['yield_pred_rf'].to_numpy().astype(np.float32)

    print(f"  Feature array: {all_features.shape}")
    print(f"  RF baseline: mean={all_rf_preds.mean():.2f}, "
          f"std={all_rf_preds.std():.2f}")
    print(f"  Extract time: {time.time()-t0:.1f}s")

    ### ---------------------------------------------------------------- ###
    ### Train/val split at municipality level (80/20)
    ### ---------------------------------------------------------------- ###

    unique_muns  =  sorted(set(k[0] for k in valid_keys))
    np.random.shuffle(unique_muns)
    n_train_muns =  int(0.8 * len(unique_muns))
    train_muns   =  set(unique_muns[:n_train_muns])
    val_muns     =  set(unique_muns[n_train_muns:])

    train_keys =  [k for k in valid_keys if k[0] in train_muns]
    val_keys   =  [k for k in valid_keys if k[0] in val_muns]

    print(f"\n  Train: {len(train_keys):,} mun-years ({len(train_muns)} muns)")
    print(f"  Val:   {len(val_keys):,} mun-years ({len(val_muns)} muns)")

    # Fit scaler on training ADCs
    train_row_idxs =  np.concatenate([np.asarray(grouped[k]) for k in train_keys])
    scaler =  StandardScaler()
    scaler.fit(all_features[train_row_idxs])
    features_scaled =  scaler.transform(all_features).astype(np.float32)
    print(f"  Scaler fit on {len(train_row_idxs):,} training ADC-year obs")

    ### ---------------------------------------------------------------- ###
    ### Inner-early-stopping mode (clean held-out protocol, 2026-08-28)
    ### ---------------------------------------------------------------- ###
    # The dev model early-stops on val_muns -- the same municipalities Table 1
    # scores -- so its stopping epoch is selected on the evaluation set. This
    # mode removes that asymmetry: early stopping uses a 10% slice of the
    # TRAINING municipalities, the model is then refit on the full training set
    # for the selected epoch count, and the validation municipalities never
    # influence any choice.
    if args.inner_es:
        input_dim =  len(all_feat_cols)
        rng_in =  np.random.default_rng(args.seed + 1)
        tm =  sorted(train_muns)
        rng_in.shuffle(tm)
        n_in =  int(0.9 * len(tm))
        inner_train_muns =  set(tm[:n_in])
        inner_stop_muns  =  set(tm[n_in:])
        inner_train_keys =  [k for k in train_keys if k[0] in inner_train_muns]
        inner_stop_keys  =  [k for k in train_keys if k[0] in inner_stop_muns]
        print(f"\n--- Inner-ES training: {len(inner_train_keys):,} train / "
              f"{len(inner_stop_keys):,} stop mun-years ---")
        model =  ResidualMLP(input_dim, args.hidden_dim, args.n_blocks,
                             dropout=0.2).to(device)
        _, stopped_epoch =  train_model(
            model, inner_train_keys, inner_stop_keys, grouped, features_scaled,
            all_weights, all_rf_preds, siap_dict, args, device)
        print(f"\n--- Refit on all {len(train_keys):,} training mun-years "
              f"for {stopped_epoch} epochs ---")
        refit =  ResidualMLP(input_dim, args.hidden_dim, args.n_blocks,
                             dropout=0.2).to(device)
        optimizer =  torch.optim.Adam(refit.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5, min_lr=1e-6)
        tk =  list(train_keys)
        for ep in range(stopped_epoch):
            refit.train()
            np.random.shuffle(tk)
            epoch_losses =  []
            for batch_start in range(0, len(tk), args.batch_size):
                batch_keys =  tk[batch_start:batch_start + args.batch_size]
                X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                    batch_keys, grouped, features_scaled, all_weights,
                    all_rf_preds, siap_dict, max_adcs=args.max_adcs)
                X_flat  =  X_flat.to(device);  W_flat  =  W_flat.to(device)
                RF_flat =  RF_flat.to(device); mun_idx =  mun_idx.to(device)
                Y       =  Y.to(device)
                if X_flat.shape[0] < 2:
                    continue
                optimizer.zero_grad()
                residuals =  torch.tanh(refit(X_flat)) * args.max_residual
                adc_pred  =  (RF_flat + residuals).clamp(min=0)
                mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx,
                                                   len(batch_keys))
                mse_loss  =  nn.functional.mse_loss(mun_pred, Y)
                loss =  mse_loss + args.residual_lambda * (residuals ** 2).mean()
                if args.var_lambda > 0:
                    loss =  loss - args.var_lambda * compute_var_penalty(
                        adc_pred, mun_idx, len(batch_keys))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(refit.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(mse_loss.item())
            scheduler.step(np.mean(epoch_losses))
            if (ep + 1) % 25 == 0:
                print(f"  Epoch {ep+1:3d}: loss={np.mean(epoch_losses):.4f}")
        refit.eval()
        chunk_size =  50000
        all_preds  =  np.zeros(len(features_scaled))
        with torch.no_grad():
            for start in range(0, len(features_scaled), chunk_size):
                end       =  min(start + chunk_size, len(features_scaled))
                X_chunk   =  torch.from_numpy(features_scaled[start:end]).to(device)
                RF_chunk  =  torch.from_numpy(all_rf_preds[start:end]).to(device)
                residuals =  torch.tanh(refit(X_chunk)) * args.max_residual
                all_preds[start:end] =  (RF_chunk + residuals).clamp(
                    min=0, max=25).cpu().numpy()
        aef['yield_pred_agg_nn'] =  all_preds
        suffix =  "phase3" if args.phase3 else "phase2"
        os.makedirs(pred_dir, exist_ok=True)
        out_path =  os.path.join(
            pred_dir, f"adc_agg_nn_preds_maize_{suffix}_inner_es.parquet")
        aef[['adcid', 'year', 'yield_pred_agg_nn']].to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")
        return

    ### ---------------------------------------------------------------- ###
    ### Train model
    ### ---------------------------------------------------------------- ###

    print("\n--- Training (dev) ---")

    input_dim =  len(all_feat_cols)
    model     =  ResidualMLP(input_dim, args.hidden_dim, args.n_blocks,
                              dropout=0.2).to(device)

    best_state, stopped_epoch =  train_model(
        model, train_keys, val_keys, grouped, features_scaled, all_weights,
        all_rf_preds, siap_dict, args, device)

    ### ---------------------------------------------------------------- ###
    ### Generate ADC-level predictions (dev model)
    ### ---------------------------------------------------------------- ###

    print("\n--- Generating ADC-level predictions (dev model) ---")
    t0 =  time.time()

    chunk_size =  50000
    all_preds  =  np.zeros(len(features_scaled))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features_scaled), chunk_size):
            end       =  min(start + chunk_size, len(features_scaled))
            X_chunk   =  torch.from_numpy(features_scaled[start:end]).to(device)
            RF_chunk  =  torch.from_numpy(all_rf_preds[start:end]).to(device)
            residuals =  torch.tanh(model(X_chunk)) * args.max_residual
            adc_pred  =  (RF_chunk + residuals).clamp(min=0, max=25)
            all_preds[start:end] =  adc_pred.cpu().numpy()

    aef['yield_pred_agg_nn'] =  all_preds
    residuals_arr =  all_preds - all_rf_preds
    print(f"  Predictions: min={all_preds.min():.2f}, max={all_preds.max():.2f}, "
          f"mean={all_preds.mean():.2f}, std={all_preds.std():.2f}")
    print(f"  Residuals: mean={residuals_arr.mean():.3f}, "
          f"std={residuals_arr.std():.3f}, "
          f"range=[{residuals_arr.min():.2f}, {residuals_arr.max():.2f}]")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Save dev predictions
    suffix =  "phase3" if args.phase3 else "phase2"
    os.makedirs(pred_dir, exist_ok=True)
    out_path =  os.path.join(pred_dir, f"adc_agg_nn_preds_maize_{suffix}.parquet")
    aef[['adcid', 'year', 'yield_pred_agg_nn']].to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    ### ---------------------------------------------------------------- ###
    ### Retrain on all data
    ### ---------------------------------------------------------------- ###

    print("\n--- Retraining on all data ---")
    t0 =  time.time()

    all_row_idxs =  np.concatenate([np.asarray(grouped[k]) for k in valid_keys])
    scaler_full  =  StandardScaler()
    scaler_full.fit(all_features[all_row_idxs])
    features_full =  scaler_full.transform(all_features).astype(np.float32)

    final_model =  ResidualMLP(input_dim, args.hidden_dim, args.n_blocks,
                                dropout=0.2).to(device)
    optimizer   =  torch.optim.Adam(final_model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    scheduler   =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)

    all_valid_keys =  list(valid_keys)
    for ep in range(stopped_epoch):
        final_model.train()
        np.random.shuffle(all_valid_keys)
        epoch_losses =  []

        for batch_start in range(0, len(all_valid_keys), args.batch_size):
            batch_keys =  all_valid_keys[batch_start:batch_start + args.batch_size]
            X_flat, W_flat, RF_flat, mun_idx, Y =  build_flat_batch(
                batch_keys, grouped, features_full, all_weights,
                all_rf_preds, siap_dict, max_adcs=args.max_adcs)

            X_flat  =  X_flat.to(device)
            W_flat  =  W_flat.to(device)
            RF_flat =  RF_flat.to(device)
            mun_idx =  mun_idx.to(device)
            Y       =  Y.to(device)

            if X_flat.shape[0] < 2:
                continue

            optimizer.zero_grad()
            residuals =  torch.tanh(final_model(X_flat)) * args.max_residual
            adc_pred  =  (RF_flat + residuals).clamp(min=0)
            mun_pred  =  aggregate_predictions(adc_pred, W_flat, mun_idx,
                                                len(batch_keys))
            mse_loss  =  nn.functional.mse_loss(mun_pred, Y)

            loss =  mse_loss + args.residual_lambda * (residuals ** 2).mean()
            if args.var_lambda > 0:
                loss =  loss - args.var_lambda * compute_var_penalty(
                    adc_pred, mun_idx, len(batch_keys))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(mse_loss.item())

        avg_loss =  np.mean(epoch_losses)
        scheduler.step(avg_loss)
        if (ep + 1) % 25 == 0:
            print(f"  Epoch {ep+1:3d}: loss={avg_loss:.4f}")

    print(f"  Final retrain time: {time.time()-t0:.1f}s")

    # Generate final predictions
    final_model.eval()
    final_preds =  np.zeros(len(features_full))
    with torch.no_grad():
        for start in range(0, len(features_full), chunk_size):
            end       =  min(start + chunk_size, len(features_full))
            X_chunk   =  torch.from_numpy(features_full[start:end]).to(device)
            RF_chunk  =  torch.from_numpy(all_rf_preds[start:end]).to(device)
            residuals =  torch.tanh(final_model(X_chunk)) * args.max_residual
            adc_pred  =  (RF_chunk + residuals).clamp(min=0, max=25)
            final_preds[start:end] =  adc_pred.cpu().numpy()

    aef['yield_pred_agg_nn_final'] =  final_preds

    out_path_final =  os.path.join(pred_dir,
                      f"adc_agg_nn_preds_maize_{suffix}_final.parquet")
    out_df_final =  aef[['adcid', 'year', 'yield_pred_agg_nn_final']].rename(
        columns={'yield_pred_agg_nn_final': 'yield_pred_agg_nn'})
    out_df_final.to_parquet(out_path_final, index=False)
    print(f"  Saved: {out_path_final}")

    ### ---------------------------------------------------------------- ###
    ### Evaluate vs INEGI 2022 census
    ### ---------------------------------------------------------------- ###

    print("\n" + "=" * 70)
    print(f"  Evaluation vs INEGI 2022 Census ({phase_label})")
    print("=" * 70)

    ca2022_path =  os.path.join(inegi_dir,
                   "LM2304-CA22-2025-09-29-superficie_ENTREGA",
                   "adc_land_use_ca22_adc07.dta")

    adc_gt =  pd.read_stata(ca2022_path)
    adc_gt =  adc_gt[adc_gt['name'] == 'Maize']
    adc_gt =  adc_gt[['adc', 'muncode', 'land_input', 'vol_output',
                       'yield', 'share_irrig']].copy()

    # Get 2022 dev predictions
    nn_dev_2022        =  aef[aef['year'] == 2022][['adcid', 'yield_pred_agg_nn']].copy()
    nn_dev_2022['adc'] =  nn_dev_2022['adcid'].str.replace('-', '', regex=False)

    # Get 2022 final predictions
    nn_fin_2022        =  aef[aef['year'] == 2022][['adcid', 'yield_pred_agg_nn_final']].copy()
    nn_fin_2022['adc'] =  nn_fin_2022['adcid'].str.replace('-', '', regex=False)

    eval_df =  adc_gt.merge(nn_dev_2022[['adc', 'yield_pred_agg_nn']], on='adc', how='left')
    eval_df =  eval_df.merge(nn_fin_2022[['adc', 'yield_pred_agg_nn_final']], on='adc', how='left')

    # Load RF baseline
    rf_path  =  os.path.join(pred_dir, "adc_alpha_earth_preds_maize.parquet")
    rf_preds_all =  pd.read_parquet(rf_path)
    rf_2022  =  rf_preds_all[rf_preds_all['year'] == 2022].copy()
    rf_2022['adc'] =  rf_2022['adcid'].str.replace('-', '', regex=False)
    eval_df  =  eval_df.merge(rf_2022[['adc', 'yield_pred']].rename(
                columns={'yield_pred': 'yield_pred_rf'}), on='adc', how='left')

    # Load irrigation adjustment
    irrig_path =  os.path.join(pred_dir,
                  "adc_alpha_earth_preds_maize_irrig_adj.parquet")
    if os.path.exists(irrig_path):
        irrig_preds =  pd.read_parquet(irrig_path)
        irrig_2022  =  irrig_preds[irrig_preds['year'] == 2022].copy()
        irrig_2022['adc'] =  irrig_2022['adcid'].str.replace('-', '', regex=False)
        eval_df =  eval_df.merge(
            irrig_2022[['adc', 'yield_pred_irrig_adj_recentered']],
            on='adc', how='left')

    # SIAP baseline
    siap_2022 =  siap_maize[siap_maize['year'] == 2022][['muncode', 'yield']].rename(
        columns={'yield': 'yield_siap'})
    eval_df =  eval_df.merge(siap_2022, on='muncode', how='left')

    # Combined model: Agg-NN between-mun + irrigation within-mun
    # Uses Agg-NN mun-level predictions (better between-R²) with
    # irrigation adjustment for within-mun variation (proven Within-R²)
    beta_hat =  1.56  # From Phase 1 CA2007 FE regression
    agland_eval =  pd.read_csv(agland_path)
    agland_eval['irrig_share'] =  (agland_eval['siap_irrig_area'] /
                                    agland_eval['siap_agland_area']).replace(
                                        [np.inf, -np.inf], np.nan).fillna(0.0)
    agland_eval['adc'] =  agland_eval['adcid'].str.replace('-', '', regex=False)
    eval_df =  eval_df.merge(agland_eval[['adc', 'irrig_share']].rename(
        columns={'irrig_share': 'irrig_share_siap'}), on='adc', how='left')
    eval_df['irrig_share_siap'] =  eval_df['irrig_share_siap'].fillna(0.0)

    mun_irrig_mean =  eval_df.groupby('muncode')['irrig_share_siap'].transform('mean')
    eval_df['irrig_dev'] =  eval_df['irrig_share_siap'] - mun_irrig_mean

    # Agg-NN dev mun mean + irrigation adjustment
    mun_nn_mean =  eval_df.groupby('muncode')['yield_pred_agg_nn'].transform('mean')
    eval_df['yield_pred_nn_irrig'] =  (
        mun_nn_mean + beta_hat * eval_df['irrig_dev']).clip(lower=0)

    # Also combine RF mun mean + Agg-NN between-mun correction + irrig within
    # (for ADCs without NN predictions, use RF + irrig)
    eval_df['yield_pred_nn_irrig'] =  eval_df['yield_pred_nn_irrig'].fillna(
        eval_df.get('yield_pred_irrig_adj_recentered', pd.Series(dtype=float)))

    print(f"\n  Evaluation data: {len(eval_df):,} ADCs")

    models =  {
        'yield_pred_rf':             'AEF RF (baseline)',
        'yield_pred_agg_nn':         f'Agg-NN dev ({phase_label})',
        'yield_pred_agg_nn_final':   f'Agg-NN final ({phase_label})',
        'yield_pred_nn_irrig':       f'Agg-NN + Irrig Adj',
        'yield_siap':                'SIAP Mun. Avg.',
    }
    if 'yield_pred_irrig_adj_recentered' in eval_df.columns:
        models['yield_pred_irrig_adj_recentered'] =  'AEF RF + Irrig Adj'

    print(f"\n{'Model':40s} {'N':>7s} {'R²':>7s} {'Between':>8s} {'Within':>8s} {'RMSE':>7s}")
    print("-" * 80)
    for pred_col, name in models.items():
        if pred_col not in eval_df.columns:
            continue
        m =  compute_all_metrics(eval_df, 'yield', pred_col, group_col='muncode')
        print(f"  {name:38s} {m['N']:>7,} {m['R2']:>7.3f} {m['Between_R2']:>8.3f} "
              f"{m['Within_R2']:>8.3f} {m['RMSE']:>7.3f}")

    # Within-mun correlation with census share_irrig
    print("\n  Within-mun correlation with census share_irrig:")
    for pred_col, name in models.items():
        if pred_col not in eval_df.columns:
            continue
        sub =  eval_df[['muncode', pred_col, 'share_irrig']].dropna()
        if len(sub) < 10:
            continue
        grp_means =  sub.groupby('muncode')[[pred_col, 'share_irrig']].transform('mean')
        p_dev =  sub[pred_col] - grp_means[pred_col]
        s_dev =  sub['share_irrig'] - grp_means['share_irrig']
        if p_dev.std() > 0 and s_dev.std() > 0:
            corr =  np.corrcoef(p_dev, s_dev)[0, 1]
            print(f"    {name:38s}  r = {corr:.4f}")

    # Save combined predictions (Agg-NN mun-level + irrigation within-mun)
    nn_all_2022     =  aef[aef['year'] == 2022][['adcid', 'muncode',
                         'yield_pred_agg_nn', 'yield_pred_rf']].copy()
    nn_all_2022     =  nn_all_2022.merge(agland[['adcid', 'irrig_share']], on='adcid', how='left')
    nn_all_2022['irrig_share'] =  nn_all_2022['irrig_share'].fillna(0.0)
    mun_irrig_all   =  nn_all_2022.groupby('muncode')['irrig_share'].transform('mean')
    mun_nn_all      =  nn_all_2022.groupby('muncode')['yield_pred_agg_nn'].transform('mean')
    nn_all_2022['yield_pred_nn_irrig'] =  (
        mun_nn_all + beta_hat * (nn_all_2022['irrig_share'] - mun_irrig_all)
    ).clip(lower=0)

    # Save for all years
    nn_combined =  aef[['adcid', 'year', 'muncode', 'yield_pred_agg_nn']].copy()
    nn_combined =  nn_combined.merge(agland[['adcid', 'irrig_share']], on='adcid', how='left')
    nn_combined['irrig_share'] =  nn_combined['irrig_share'].fillna(0.0)
    mun_irrig_c =  nn_combined.groupby(['muncode', 'year'])['irrig_share'].transform('mean')
    mun_nn_c    =  nn_combined.groupby(['muncode', 'year'])['yield_pred_agg_nn'].transform('mean')
    nn_combined['yield_pred_nn_irrig'] =  (
        mun_nn_c + beta_hat * (nn_combined['irrig_share'] - mun_irrig_c)
    ).clip(lower=0)

    out_combined =  os.path.join(pred_dir,
                    f"adc_agg_nn_irrig_preds_maize_{suffix}.parquet")
    nn_combined[['adcid', 'year', 'yield_pred_agg_nn', 'yield_pred_nn_irrig',
                  'irrig_share']].to_parquet(out_combined, index=False)
    print(f"\n  Saved combined: {out_combined}")

    print("\nDone.")


if __name__ == '__main__':
    main()
