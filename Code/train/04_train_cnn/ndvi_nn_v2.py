"""
ndvi_nn_v2.py — Improved CNN for histogram-based yield prediction
Phase 1 improvements over ndvi_nn_redux.py:
  1a. Architecture: 3x3 kernels, residual blocks, dropout 0.2, lighter (32→64→128)
  1b. Training: LR 0.001 + cosine annealing, weight decay 1e-4, patience 20, batch 64
  1c. Input: log1p-transform histograms, per-channel standardization

Supports two histogram formats:
  - 3-channel: separate ndvi_hist/gcvi_hist/ndti_hist columns → (32,32,3)
  - 1-channel: single 'hist' column (1024-element NDVI 2D histogram) → (32,32,1)

Usage:
  python ndvi_nn_v2.py <holdout_year>
  python ndvi_nn_v2.py 2020

Run from Maize_prediction/ directory.
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
from torch.utils.data import Dataset, DataLoader

holdout_year =  int(sys.argv[1])
bins         =  32
seed         =  1364
batch_size   =  64
max_epochs   =  200
patience     =  20
lr           =  0.001
weight_decay =  1e-4

torch.manual_seed(seed)
np.random.seed(seed)

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Residual CNN ──────────────────────────────────────────

class ResBlock(nn.Module):
    """Two 3x3 convs with batch norm, ReLU, and skip connection."""

    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride =  2 if downsample else 1
        self.conv1 =  nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1   =  nn.BatchNorm2d(out_ch)
        self.conv2 =  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2   =  nn.BatchNorm2d(out_ch)
        self.relu  =  nn.ReLU(inplace=True)

        # Skip connection: match dimensions if needed
        if in_ch != out_ch or downsample:
            self.skip =  nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip =  nn.Identity()

    def forward(self, x):
        identity =  self.skip(x)
        out =  self.relu(self.bn1(self.conv1(x)))
        out =  self.bn2(self.conv2(out))
        out =  self.relu(out + identity)
        return out


class HistogramCNN(nn.Module):
    """
    Residual CNN for 32x32x3 spectral-index histograms.
    Architecture: 3 residual blocks (32→64→128) with avg-pool downsampling,
    global average pool, dropout 0.2, linear output.
    """

    def __init__(self, in_channels=3, dropout=0.2):
        super().__init__()

        self.block1 =  ResBlock(in_channels, 32, downsample=False)  # 32x32→32x32
        self.pool1  =  nn.AvgPool2d(2)                              # →16x16

        self.block2 =  ResBlock(32, 64, downsample=False)           # 16x16→16x16
        self.pool2  =  nn.AvgPool2d(2)                              # →8x8

        self.block3 =  ResBlock(64, 128, downsample=False)          # 8x8→8x8
        self.pool3  =  nn.AvgPool2d(2)                              # →4x4

        self.gap     =  nn.AdaptiveAvgPool2d(1)  # →1x1x128
        self.dropout =  nn.Dropout(dropout)
        self.fc      =  nn.Linear(128, 1)

    def forward(self, x):
        x =  self.pool1(self.block1(x))
        x =  self.pool2(self.block2(x))
        x =  self.pool3(self.block3(x))
        x =  self.gap(x).squeeze(-1).squeeze(-1)  # (B, 128)
        x =  self.dropout(x)
        x =  self.fc(x)  # Linear output for regression
        return x.squeeze(-1)


# ── ADC pixel subsampling augmentation ────────────────────

def return_adc_sample_mhist(munhist, small_n):
    """Subsample histogram to simulate ADC-level pixel counts."""
    total      =  munhist.sum()
    flat       =  munhist.ravel()
    probs      =  flat / total
    choices    =  np.random.choice(len(flat), size=small_n, p=probs)
    resampled  =  np.zeros_like(flat)
    for c in choices:
        resampled[c] += 1
    return resampled.reshape(munhist.shape) / small_n


# ── Dataset ───────────────────────────────────────────────

class HistDataset(Dataset):
    """
    Dataset for histogram-based yield prediction.
    Applies log1p transform and per-channel standardization.
    Optionally applies ADC subsampling augmentation during training.
    """

    def __init__(self, df, channel_stats=None, augment=False):
        """
        Args:
            df: DataFrame with 'hist' (32x32x3 arrays), 'yield', 'n_pixel' columns
            channel_stats: dict with 'mean' and 'std' arrays (3,) for standardization.
                           If None, computed from this dataset.
            augment: whether to apply ADC subsampling augmentation
        """
        self.hists    =  df['hist'].tolist()
        self.yields   =  df['yield'].values.astype(np.float32)
        self.augment  =  augment

        if augment and 'n_pixel' in df.columns:
            self.n_pixels =  df['n_pixel'].tolist()
        else:
            self.n_pixels =  None

        # Compute or store channel statistics for log1p-transformed histograms
        if channel_stats is None:
            all_hists        =  np.stack(self.hists)                  # (N, 32, 32, 3)
            all_log          =  np.log1p(all_hists)
            self.chan_mean    =  all_log.mean(axis=(0, 1, 2))         # (3,)
            self.chan_std     =  all_log.std(axis=(0, 1, 2)) + 1e-8   # (3,)
        else:
            self.chan_mean =  channel_stats['mean']
            self.chan_std  =  channel_stats['std']

    def get_channel_stats(self):
        return {'mean': self.chan_mean, 'std': self.chan_std}

    def __len__(self):
        return len(self.hists)

    def __getitem__(self, idx):
        hist =  self.hists[idx].copy()  # (32, 32, 3)

        # ADC subsampling augmentation (training only)
        if self.augment and self.n_pixels is not None:
            pixel_list =  self.n_pixels[idx]
            small_n    =  int(np.random.choice(pixel_list, 1)[0])
            if small_n > 0:
                channels =  []
                for c in range(hist.shape[2]):
                    ch =  return_adc_sample_mhist(hist[:, :, c], small_n)
                    channels.append(ch)
                hist =  np.stack(channels, axis=-1)

        # Log1p transform
        hist =  np.log1p(hist)

        # Per-channel standardization
        hist =  (hist - self.chan_mean) / self.chan_std

        # Convert to channels-first (3, 32, 32) for PyTorch
        hist =  hist.transpose(2, 0, 1).astype(np.float32)

        y =  self.yields[idx]
        return torch.from_numpy(hist), torch.tensor(y, dtype=torch.float32)


# ── Training loop ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss =  0.0
    n          =  0
    for X, y in loader:
        X, y =  X.to(device), y.to(device)
        optimizer.zero_grad()
        pred =  model(X)
        loss =  criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss =  0.0
    n          =  0
    all_preds  =  []
    all_true   =  []
    with torch.no_grad():
        for X, y in loader:
            X, y =  X.to(device), y.to(device)
            pred =  model(X)
            loss =  criterion(pred, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            all_preds.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
    all_preds =  np.concatenate(all_preds)
    all_true  =  np.concatenate(all_true)
    mse       =  total_loss / n
    return mse, all_preds, all_true


# ── Main ──────────────────────────────────────────────────

def add_zeros(x, n):
    x =  str(x)
    while len(x) < n:
        x =  "0" + x
    return x


print(f"Loading data...")

# ── Resolve paths (run from Maize_prediction/ or detect data dir) ────
data_dir =  os.environ.get('DATA_DIR', os.getcwd())
if not os.path.isdir(os.path.join(data_dir, 'Data')):
    # Try parent of script directory
    data_dir =  os.path.expanduser("~/Dropbox/Projects/Maize_prediction")
print(f"Data dir: {data_dir}")

hist_3ch_dir     =  os.path.join(data_dir, "Data", "muni_vi_hists_0.2_1.0_0.0_12.0_0.0_0.6_32_max")
hist_1ch_dir     =  os.path.join(data_dir, "Data", "muni_ndvi_hist_0.2_1.0_32_max")
adc_counts_path  =  os.path.join(data_dir, "Data", "mun_adc_pixel_counts.csv")
out_dir          =  os.path.join(data_dir, "Data", "predictions")
os.makedirs(out_dir, exist_ok=True)

mun_maize_yields            =  load_muni_yields()   # canonical SIAP Maize/Spring-Summer

# ── Load histograms: prefer 3-channel, fall back to 1-channel ────
if os.path.isdir(hist_3ch_dir):
    print("Loading 3-channel histograms (NDVI/GCVI/NDTI)...")
    hist_fs =  [f for f in os.listdir(hist_3ch_dir) if f.endswith(".pkl")]
    hist_df =  pd.concat([pd.read_pickle(os.path.join(hist_3ch_dir, f)) for f in hist_fs])
    hist_df['hist'] =  hist_df.apply(lambda x: np.concatenate(
        [np.reshape(x['ndvi_hist'], (bins, bins, 1)),
         np.reshape(x['gcvi_hist'], (bins, bins, 1)),
         np.reshape(x['ndti_hist'], (bins, bins, 1))],
        axis=-1
    ), axis=1)
    n_channels =  3
elif os.path.isdir(hist_1ch_dir):
    print("Loading 1-channel histograms (NDVI only)...")
    hist_fs =  [f for f in os.listdir(hist_1ch_dir) if f.endswith(".pkl")]
    hist_df =  pd.concat([pd.read_pickle(os.path.join(hist_1ch_dir, f)) for f in hist_fs])
    # Reshape 1024-element flat histogram → (32, 32, 1)
    hist_df['hist'] =  hist_df['hist'].apply(
        lambda x: np.reshape(x, (bins, bins, 1))
    )
    # Construct muncode from CVE_ENT + CVE_MUN
    hist_df['CVE_ENT_str'] =  hist_df['CVE_ENT'].astype(str).str.zfill(2)
    hist_df['CVE_MUN_str'] =  hist_df['CVE_MUN'].astype(str).str.zfill(3)
    hist_df['muncode']     =  (hist_df['CVE_ENT_str'] + hist_df['CVE_MUN_str']).astype(int)
    n_channels =  1
else:
    raise FileNotFoundError(f"No histogram directory found at {hist_3ch_dir} or {hist_1ch_dir}")

print(f"  Loaded {len(hist_df)} histograms, {n_channels} channel(s)")

hist_df =  pd.merge(hist_df, mun_maize_yields, on=['muncode', 'year'])
print(f"  After yield merge: {len(hist_df)} observations")

# ── ADC pixel counts (optional augmentation) ─────────────
has_adc =  os.path.isfile(adc_counts_path)
if has_adc:
    print("Loading ADC pixel counts for augmentation...")
    adc_pixel_counts            =  pd.read_csv(adc_counts_path)
    adc_pixel_counts            =  adc_pixel_counts.loc[adc_pixel_counts['year'] == 2007]
    adc_pixel_counts['muncode'] =  adc_pixel_counts['muncode'].apply(int)
    adc_pixel_counts['n_pixel'] =  adc_pixel_counts['n_pixel'].apply(literal_eval)
    hist_df =  pd.merge(hist_df, adc_pixel_counts[['muncode', 'n_pixel']], on='muncode')
else:
    print("ADC pixel counts not found — skipping subsampling augmentation")

# ── Fold assignments ──────────────────────────────────────
np.random.seed(seed)
muncodes         =  hist_df[['muncode']].drop_duplicates()
muncodes['fold'] =  np.random.choice(range(0, 5), len(muncodes.index), replace=True)
hist_df          =  pd.merge(hist_df, muncodes, on='muncode')

# ── Leave-one-year-out split ──────────────────────────────
train_df =  hist_df.loc[hist_df['year'] != holdout_year].copy()
val_df   =  hist_df.loc[hist_df['year'] == holdout_year].copy()

if len(val_df) == 0:
    print(f"No data for holdout year {holdout_year}. Available years: {sorted(hist_df['year'].unique())}")
    sys.exit(1)

print(f"Holdout year: {holdout_year}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}")
print(f"Train yield mean: {train_df['yield'].mean():.2f}, std: {train_df['yield'].std():.2f}")

# ── Standardize yields ────────────────────────────────────
yield_mean =  train_df['yield'].mean()
yield_std  =  train_df['yield'].std()

scale_df =  pd.DataFrame({'mean': [yield_mean], 'sd': [yield_std]})
scale_df.to_csv(os.path.join(out_dir, f"nn_v2_rescale_year{holdout_year}.csv"), index=False)

train_df.loc[:, 'yield'] =  (train_df['yield'] - yield_mean) / yield_std
val_df.loc[:, 'yield']   =  (val_df['yield'] - yield_mean) / yield_std

# ── Build datasets ────────────────────────────────────────
use_augment   =  has_adc
train_dataset =  HistDataset(train_df, augment=use_augment)
chan_stats     =  train_dataset.get_channel_stats()
val_dataset   =  HistDataset(val_df, channel_stats=chan_stats, augment=False)

train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, drop_last=False)
val_loader   =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0)

# ── Build model ───────────────────────────────────────────
model     =  HistogramCNN(in_channels=n_channels, dropout=0.2).to(device)
criterion =  nn.MSELoss()
optimizer =  torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# Validation baseline (predict mean = 0 since yields are standardized)
val_mss =  ((val_df['yield'] - val_df['yield'].mean()) ** 2).mean()
print(f"\nVal MSS (predict mean): {val_mss:.4f}")

# ── Training loop ─────────────────────────────────────────
best_val_loss     =  float('inf')
epochs_no_improve =  0
best_model_path   =  os.path.join(out_dir, f"ndvi_v2_b{bins}_year{holdout_year}.pt")

for epoch in range(max_epochs):
    train_loss =  train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_true =  evaluate(model, val_loader, criterion, device)
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

# ── Evaluate best model ───────────────────────────────────
model.load_state_dict(torch.load(best_model_path, weights_only=True))
val_loss, val_preds, val_true =  evaluate(model, val_loader, criterion, device)

# Compute R^2 in standardized space
ss_res =  np.sum((val_true - val_preds) ** 2)
ss_tot =  np.sum((val_true - val_true.mean()) ** 2)
r2     =  1 - ss_res / ss_tot

print(f"\n{'='*50}")
print(f"Holdout year {holdout_year}")
print(f"Best val MSE: {best_val_loss:.4f}")
print(f"Val R²: {r2:.4f}")
print(f"Val MSS: {val_mss:.4f}")
print(f"{'='*50}")

# Save predictions (rescaled to original units)
val_out             =  val_df[['muncode', 'year']].copy()
val_out['pred']     =  val_preds * yield_std + yield_mean
val_out['actual']   =  val_true * yield_std + yield_mean
val_out['pred_std'] =  val_preds
preds_path =  os.path.join(out_dir, f"ndvi_v2_preds_year{holdout_year}.csv")
val_out.to_csv(preds_path, index=False)
print(f"Predictions saved to {preds_path}")
