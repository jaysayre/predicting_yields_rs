"""
ndvi_nn_temporal.py — Temporal CNN for monthly histogram-based yield prediction
Phase 3: Architecture designed for (32 bins × 8 months × 6 indices) tensors
from the redesigned GEE extraction (ls_monthly_hists.py).

Architecture:
  1. Spectral branch: 1D Conv along bin dimension → extract spectral features per time step
  2. Temporal branch: 1D Conv + attention along month dimension → capture growth dynamics
  3. Auxiliary branch: Dense layers for tabular features (coordinates, elevation, state)
  4. Fusion: Concatenate all branches → Dense → linear output

Usage:
  python ndvi_nn_temporal.py <holdout_year>
  python ndvi_nn_temporal.py 2020
"""

import os
import sys
import pickle
from ast import literal_eval

import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

holdout_year =  int(sys.argv[1])
bins         =  32
n_months     =  8
n_indices    =  6
seed         =  1364
batch_size   =  64
max_epochs   =  200
patience     =  25
lr           =  0.001
weight_decay =  1e-4

torch.manual_seed(seed)
np.random.seed(seed)

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Temporal Attention ────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Soft attention over time steps. Learns which months matter most
    for yield prediction.

    Input:  (B, T, D) — batch × time steps × features
    Output: (B, D) — attention-weighted sum over time
    """

    def __init__(self, dim):
        super().__init__()
        self.attn =  nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, T, D)
        weights =  self.attn(x)           # (B, T, 1)
        weights =  F.softmax(weights, dim=1)
        out     =  (x * weights).sum(dim=1)  # (B, D)
        return out, weights.squeeze(-1)


# ── Spectral Feature Extractor ────────────────────────────

class SpectralBlock(nn.Module):
    """
    Extract features from the bin dimension for each (month, index) pair.
    Input:  (B, bins, n_months, n_indices)
    Output: (B, n_months, feat_dim)

    Treats (n_months × n_indices) as the batch dimension,
    applies 1D convolutions along the bin axis.
    """

    def __init__(self, n_indices, feat_dim=64):
        super().__init__()
        self.conv =  nn.Sequential(
            nn.Conv1d(n_indices, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pool across bins → single value per filter
        )
        self.proj =  nn.Linear(64, feat_dim)

    def forward(self, x):
        # x: (B, bins, months, indices)
        B, bins, M, I =  x.shape

        # Reshape to process each month independently: (B*M, I, bins)
        x =  x.permute(0, 2, 3, 1)      # (B, M, I, bins)
        x =  x.reshape(B * M, I, bins)   # (B*M, I, bins) — channels=indices, length=bins

        x =  self.conv(x)                # (B*M, 64, 1)
        x =  x.squeeze(-1)              # (B*M, 64)
        x =  self.proj(x)               # (B*M, feat_dim)
        x =  x.reshape(B, M, -1)        # (B, M, feat_dim)
        return x


# ── Temporal Feature Extractor ────────────────────────────

class TemporalBlock(nn.Module):
    """
    Process the temporal sequence of spectral features.
    Input:  (B, n_months, feat_dim)
    Output: (B, out_dim) via temporal conv + attention
    """

    def __init__(self, feat_dim, out_dim=64):
        super().__init__()
        # Temporal convolutions (1D along month axis)
        self.conv =  nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.attention =  TemporalAttention(out_dim)

    def forward(self, x):
        # x: (B, M, feat_dim)
        x =  x.permute(0, 2, 1)    # (B, feat_dim, M) for Conv1d
        x =  self.conv(x)           # (B, out_dim, M)
        x =  x.permute(0, 2, 1)    # (B, M, out_dim)
        x, attn_weights =  self.attention(x)  # (B, out_dim)
        return x, attn_weights


# ── 2D CNN alternative ────────────────────────────────────

class Conv2DBlock(nn.Module):
    """
    Alternative: treat (bins × months) as a 2D spatial input.
    Uses rectangular kernels (wider along bins, narrower along time).
    Input:  (B, n_indices, bins, months) — channels = indices
    Output: (B, out_dim)
    """

    def __init__(self, n_indices, out_dim=128):
        super().__init__()
        self.features =  nn.Sequential(
            nn.Conv2d(n_indices, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((2, 1)),  # Downsample bins only (keep all months)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d((2, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj =  nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (B, I, bins, months)
        x =  self.features(x)     # (B, 128, 1, 1)
        x =  x.squeeze(-1).squeeze(-1)
        x =  self.proj(x)
        return x


# ── Full Hybrid Model ────────────────────────────────────

class TemporalHistogramCNN(nn.Module):
    """
    Hybrid model combining:
      - Spectral + temporal CNN for histogram tensor (32 × 8 × 6)
      - Dense branch for auxiliary tabular features
      - Optional percentile features

    Args:
        n_indices: number of spectral indices (6)
        n_months: number of monthly time steps (8)
        bins: number of histogram bins (32)
        n_aux: number of auxiliary tabular features (0 to disable)
        n_pctl: number of percentile features (0 to disable)
        use_2d: if True, use Conv2D approach instead of spectral+temporal
        dropout: dropout rate
    """

    def __init__(self, n_indices=6, n_months=8, bins=32,
                 n_aux=0, n_pctl=0, use_2d=False, dropout=0.2):
        super().__init__()
        self.use_2d  =  use_2d
        self.n_aux   =  n_aux
        self.n_pctl  =  n_pctl

        if use_2d:
            self.hist_branch =  Conv2DBlock(n_indices, out_dim=128)
            hist_dim =  128
        else:
            self.spectral =  SpectralBlock(n_indices, feat_dim=64)
            self.temporal =  TemporalBlock(feat_dim=64, out_dim=64)
            hist_dim =  64

        # Percentile branch
        if n_pctl > 0:
            pctl_flat =  n_months * n_indices * 5  # 5 percentiles
            self.pctl_branch =  nn.Sequential(
                nn.Linear(pctl_flat, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            pctl_dim =  32
        else:
            pctl_dim =  0

        # Auxiliary feature branch
        if n_aux > 0:
            self.aux_branch =  nn.Sequential(
                nn.Linear(n_aux, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            aux_dim =  16
        else:
            aux_dim =  0

        # Fusion head
        total_dim =  hist_dim + pctl_dim + aux_dim
        self.head =  nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, hist_tensor, pctl_features=None, aux_features=None):
        """
        Args:
            hist_tensor:   (B, bins, months, indices)
            pctl_features: (B, n_pctl_flat) or None
            aux_features:  (B, n_aux) or None
        """
        if self.use_2d:
            # Reshape to (B, indices, bins, months) for Conv2d
            x =  hist_tensor.permute(0, 3, 1, 2)
            hist_out =  self.hist_branch(x)
            attn_weights =  None
        else:
            spectral_out =  self.spectral(hist_tensor)           # (B, M, 64)
            hist_out, attn_weights =  self.temporal(spectral_out)  # (B, 64)

        parts =  [hist_out]

        if self.n_pctl > 0 and pctl_features is not None:
            parts.append(self.pctl_branch(pctl_features))

        if self.n_aux > 0 and aux_features is not None:
            parts.append(self.aux_branch(aux_features))

        combined =  torch.cat(parts, dim=1)
        out      =  self.head(combined).squeeze(-1)
        return out, attn_weights


# ── Dataset ───────────────────────────────────────────────

class MonthlyHistDataset(Dataset):
    """
    Dataset for monthly histogram tensors.
    Applies log1p transform and per-channel standardization.
    """

    def __init__(self, df, channel_stats=None, include_pctl=True, include_aux=False):
        """
        Args:
            df: DataFrame with 'hist_tensor' (bins, months, indices) arrays,
                'yield' column, optional 'percentiles' and aux columns
            channel_stats: dict with 'mean'/'std' arrays for standardization
            include_pctl: whether to include percentile features
            include_aux: whether to include auxiliary features
        """
        self.tensors =  df['hist_tensor'].tolist()
        self.yields  =  df['yield'].values.astype(np.float32)

        # Percentiles
        self.include_pctl =  include_pctl
        if include_pctl and 'percentiles' in df.columns:
            self.pctls =  [p.ravel().astype(np.float32) for p in df['percentiles'].tolist()]
        else:
            self.pctls         =  None
            self.include_pctl  =  False

        # Auxiliary features
        self.include_aux =  include_aux
        aux_cols         =  [c for c in ['lat', 'lon', 'elevation', 'state_id', 'area_ha',
                                          'irrig_frac'] if c in df.columns]
        if include_aux and len(aux_cols) > 0:
            self.aux  =  df[aux_cols].values.astype(np.float32)
            self.n_aux =  len(aux_cols)
        else:
            self.aux          =  None
            self.include_aux  =  False
            self.n_aux        =  0

        # Channel stats for histogram normalization
        if channel_stats is None:
            all_t          =  np.stack(self.tensors)         # (N, bins, months, indices)
            all_log        =  np.log1p(all_t)
            self.chan_mean  =  all_log.mean(axis=(0, 1, 2))  # (indices,)
            self.chan_std   =  all_log.std(axis=(0, 1, 2)) + 1e-8
        else:
            self.chan_mean =  channel_stats['mean']
            self.chan_std  =  channel_stats['std']

    def get_channel_stats(self):
        return {'mean': self.chan_mean, 'std': self.chan_std}

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        tensor =  self.tensors[idx].copy()  # (bins, months, indices)

        # Log1p + standardize
        tensor =  np.log1p(tensor)
        tensor =  (tensor - self.chan_mean) / self.chan_std
        tensor =  tensor.astype(np.float32)

        result =  {'hist': torch.from_numpy(tensor),
                    'yield': torch.tensor(self.yields[idx], dtype=torch.float32)}

        if self.include_pctl and self.pctls is not None:
            result['pctl'] =  torch.from_numpy(self.pctls[idx])

        if self.include_aux and self.aux is not None:
            result['aux'] =  torch.from_numpy(self.aux[idx])

        return result


# ── Training loop ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, has_pctl, has_aux):
    model.train()
    total_loss =  0.0
    n          =  0
    for batch in loader:
        X    =  batch['hist'].to(device)
        y    =  batch['yield'].to(device)
        pctl =  batch.get('pctl', None)
        aux  =  batch.get('aux', None)
        if pctl is not None:
            pctl =  pctl.to(device)
        if aux is not None:
            aux =  aux.to(device)

        optimizer.zero_grad()
        pred, _ =  model(X, pctl if has_pctl else None, aux if has_aux else None)
        loss =  criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def evaluate(model, loader, criterion, device, has_pctl, has_aux):
    model.eval()
    total_loss =  0.0
    n          =  0
    all_preds  =  []
    all_true   =  []
    all_attn   =  []
    with torch.no_grad():
        for batch in loader:
            X    =  batch['hist'].to(device)
            y    =  batch['yield'].to(device)
            pctl =  batch.get('pctl', None)
            aux  =  batch.get('aux', None)
            if pctl is not None:
                pctl =  pctl.to(device)
            if aux is not None:
                aux =  aux.to(device)

            pred, attn =  model(X, pctl if has_pctl else None, aux if has_aux else None)
            loss =  criterion(pred, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            all_preds.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
            if attn is not None:
                all_attn.append(attn.cpu().numpy())

    all_preds =  np.concatenate(all_preds)
    all_true  =  np.concatenate(all_true)
    if len(all_attn) > 0:
        all_attn =  np.concatenate(all_attn)
    mse =  total_loss / n
    return mse, all_preds, all_true, all_attn


# ── Main ──────────────────────────────────────────────────

def add_zeros(x, n):
    x =  str(x)
    while len(x) < n:
        x =  "0" + x
    return x


print(f"Loading data...")

# Load preprocessed monthly histograms
hist_path =  "../Data/monthly_hists_clean/monthly_hists_all.pkl"
hist_df   =  pd.read_pickle(hist_path)

# Load yields
mun_maize_yields            =  load_muni_yields()   # canonical SIAP Maize/Spring-Summer

hist_df =  pd.merge(hist_df, mun_maize_yields, on=['muncode', 'year'])

print(f"Total observations: {len(hist_df)}")

# Leave-one-year-out split
train_df =  hist_df.loc[hist_df['year'] != holdout_year].copy()
val_df   =  hist_df.loc[hist_df['year'] == holdout_year].copy()

print(f"Holdout year: {holdout_year}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Standardize yields
yield_mean =  train_df['yield'].mean()
yield_std  =  train_df['yield'].std()
print(f"Train yield mean: {yield_mean:.2f}, std: {yield_std:.2f}")

scale_df =  pd.DataFrame({'mean': [yield_mean], 'sd': [yield_std]})
scale_df.to_csv("../processed_data/nn_temporal_rescale.csv", index=False)

train_df.loc[:, 'yield'] =  (train_df['yield'] - yield_mean) / yield_std
val_df.loc[:, 'yield']   =  (val_df['yield'] - yield_mean) / yield_std

# Check for auxiliary features
has_pctl =  'percentiles' in train_df.columns
has_aux  =  any(c in train_df.columns for c in ['lat', 'lon', 'elevation'])

print(f"Percentile features: {has_pctl}")
print(f"Auxiliary features: {has_aux}")

# Build datasets
train_dataset =  MonthlyHistDataset(train_df, include_pctl=has_pctl, include_aux=has_aux)
chan_stats     =  train_dataset.get_channel_stats()
val_dataset   =  MonthlyHistDataset(val_df, channel_stats=chan_stats,
                                     include_pctl=has_pctl, include_aux=has_aux)

def collate_fn(batch):
    """Custom collate to handle optional fields."""
    result =  {
        'hist':  torch.stack([b['hist'] for b in batch]),
        'yield': torch.stack([b['yield'] for b in batch]),
    }
    if 'pctl' in batch[0]:
        result['pctl'] =  torch.stack([b['pctl'] for b in batch])
    if 'aux' in batch[0]:
        result['aux'] =  torch.stack([b['aux'] for b in batch])
    return result

train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, collate_fn=collate_fn)
val_loader   =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_fn)

# Determine aux dimension
n_aux  =  train_dataset.n_aux if has_aux else 0
n_pctl =  n_months * n_indices * 5 if has_pctl else 0

# Build model — try both approaches
model =  TemporalHistogramCNN(
    n_indices=n_indices,
    n_months=n_months,
    bins=bins,
    n_aux=n_aux,
    n_pctl=n_pctl,
    use_2d=False,   # Spectral+temporal with attention
    dropout=0.2,
).to(device)

criterion =  nn.MSELoss()
optimizer =  torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# Validation baseline
val_mss =  ((val_df['yield'] - val_df['yield'].mean()) ** 2).mean()
print(f"\nVal MSS (predict mean): {val_mss:.4f}")

# Training loop
best_val_loss     =  float('inf')
epochs_no_improve =  0
best_model_path   =  f"../processed_data/temporal_nn_year{holdout_year}.pt"

for epoch in range(max_epochs):
    train_loss =  train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, has_pctl, has_aux)
    val_loss, val_preds, val_true, val_attn =  evaluate(
        model, val_loader, criterion, device, has_pctl, has_aux
    )
    scheduler.step()

    current_lr =  optimizer.param_groups[0]['lr']

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}  train_mse={train_loss:.4f}  "
              f"val_mse={val_loss:.4f}  lr={current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss     =  val_loss
        epochs_no_improve =  0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Load best model and evaluate
model.load_state_dict(torch.load(best_model_path, weights_only=True))
val_loss, val_preds, val_true, val_attn =  evaluate(
    model, val_loader, criterion, device, has_pctl, has_aux
)

# R^2
ss_res =  np.sum((val_true - val_preds) ** 2)
ss_tot =  np.sum((val_true - val_true.mean()) ** 2)
r2     =  1 - ss_res / ss_tot

print(f"\n{'='*50}")
print(f"Holdout year {holdout_year}")
print(f"Best val MSE: {best_val_loss:.4f}")
print(f"Val R²: {r2:.4f}")
print(f"Val MSS: {val_mss:.4f}")
print(f"{'='*50}")

# Attention weights summary
if val_attn is not None and len(val_attn) > 0:
    mean_attn =  val_attn.mean(axis=0)
    print(f"\nMean attention weights per month:")
    for m in range(n_months):
        print(f"  Month {m}: {mean_attn[m]:.3f}")

# Save predictions
val_out             =  val_df[['muncode', 'year']].copy()
val_out['pred']     =  val_preds * yield_std + yield_mean
val_out['actual']   =  val_true * yield_std + yield_mean
val_out['pred_std'] =  val_preds
val_out.to_csv(f"../processed_data/temporal_nn_preds_year{holdout_year}.csv", index=False)
print(f"Predictions saved.")
