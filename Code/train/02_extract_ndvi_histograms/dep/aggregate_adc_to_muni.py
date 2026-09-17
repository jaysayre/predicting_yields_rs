"""
aggregate_adc_to_muni.py — Build municipality harmonic-feature parquets by
aggregating ADC-level exports (from ls_harmonic_features.py --level adc).

Because the municipality geometry is the dissolved union of its ADCs and the
harmonic fit is per-pixel, the muni histogram is EXACTLY the sum of its ADCs'
count-histograms, and the muni coefficient is the pixel-count-weighted mean of
its ADCs' coefficients. So we get identical muni features to a direct muni
extraction, at ADC-level cost — no separate (expensive) muni run.

Reads the raw ADC CSVs (keeps histogram COUNTS, not normalized), groups by
muncode (= adcid[:5]) and year, sums counts, normalizes, and weight-averages
coefficients. Outputs the same five parquets as clean_harmonic_features.py
--level muni.

Usage:
  ~/miniforge3/envs/geo_env/bin/python aggregate_adc_to_muni.py \
      --csv_dir <downloaded_adc_csvs> [--out_dir ...] [--bins 16] [--harmonics 3]
"""
import os
import glob
import argparse

import numpy  as np
import pandas as pd

parser =  argparse.ArgumentParser()
parser.add_argument('--csv_dir', required=True, help='dir of ADC harmfeat CSVs')
parser.add_argument('--out_dir', default=None)
parser.add_argument('--bins', type=int, default=16)
parser.add_argument('--harmonics', type=int, default=3)
args =  parser.parse_args()

BINS =  args.bins
NSQ  =  BINS * BINS
COEF =  ['h_const'] + sum([[f'h_cos{k}', f'h_sin{k}'] for k in range(1, args.harmonics + 1)], [])
HISTS =  ['h3_f_p12', 'h3_f_p23', 'h3_f_p13', 'h3_q_p12', 'h3_q_p23', 'h3_q_p13',
          'h2_f_p12', 'h2_q_p12']

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
out_dir  =  args.out_dir or os.path.join(data_dir, "harmonic_features")
os.makedirs(out_dir, exist_ok=True)


def parse_counts(cell):
    """GEE fixedHistogram string -> raw count vector (len NSQ). Zeros if empty."""
    if pd.isna(cell) or cell in ('', None):
        return np.zeros(NSQ, dtype=np.float64)
    s =  str(cell).replace('[[', '[').replace(']]', ']').strip('][')
    counts =  []
    for entry in s.split('], ['):
        parts =  entry.split(', ')
        if len(parts) >= 2:
            try:
                counts.append(float(parts[1]))
            except ValueError:
                counts.append(0.0)
    a =  np.zeros(NSQ, dtype=np.float64)
    a[:min(len(counts), NSQ)] =  counts[:NSQ]
    return a


files =  sorted(glob.glob(os.path.join(args.csv_dir, "harmfeat_adc_*.csv")))
print(f"{len(files)} ADC CSVs")
if not files:
    raise SystemExit("no harmfeat_adc_*.csv found")

# accumulate per (muncode, year): summed counts per hist, weighted coef sums, pixel weight
count_acc =  {}      # (mun, yr) -> {hist: np.array(NSQ)}
coef_acc  =  {}      # (mun, yr) -> np.array(len COEF)  (pixel-weighted sum)
wsum      =  {}      # (mun, yr) -> total pixels

for f in files:
    df =  pd.read_csv(f)
    df['muncode'] =  df['adcid'].astype(str).str[:5]
    for _, r in df.iterrows():
        key =  (r['muncode'], int(r['gs_year']))
        cnts =  {h: parse_counts(r.get(h)) for h in HISTS}
        w    =  cnts['h3_f_p12'].sum()                 # valid pixel count for this ADC
        if w <= 0:
            continue
        if key not in count_acc:
            count_acc[key] =  {h: np.zeros(NSQ) for h in HISTS}
            coef_acc[key]  =  np.zeros(len(COEF)); wsum[key] = 0.0
        for h in HISTS:
            count_acc[key][h] +=  cnts[h]
        cvals =  np.array([r.get(c, np.nan) for c in COEF], dtype=np.float64)
        if np.all(np.isfinite(cvals)):
            coef_acc[key] +=  cvals * w
            wsum[key]     +=  w

keys =  sorted(count_acc.keys())
print(f"aggregated to {len(keys):,} municipality-years")

# ── coefficients ─────────────────────────────────────────
coef_rows =  []
for (mun, yr) in keys:
    if wsum[(mun, yr)] > 0:
        row =  {'muncode': mun, 'year': yr}
        cm  =  coef_acc[(mun, yr)] / wsum[(mun, yr)]
        row.update({c: cm[i] for i, c in enumerate(COEF)})
        coef_rows.append(row)
pd.DataFrame(coef_rows).to_parquet(os.path.join(out_dir, "muni_harmonic_coefs.parquet"), index=False)
print(f"  wrote muni_harmonic_coefs.parquet ({len(coef_rows)}, {2+len(COEF)})")

# ── histograms (sum counts -> normalize per period block) ─
def write_hist(hcols, prefix, name):
    rows =  []
    for (mun, yr) in keys:
        rec =  {'muncode': mun, 'year': yr}
        ok  =  True
        for h in hcols:
            c =  count_acc[(mun, yr)][h]; tot = c.sum()
            if tot <= 0:
                ok =  False; break
            p =  h.split('_')[-1]
            frac =  (c / tot).astype(np.float32)
            for j in range(NSQ):
                rec[f"{prefix}_{p}_b{j:03d}"] =  frac[j]
        if ok:
            rows.append(rec)
    df =  pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, name), index=False)
    print(f"  wrote {name} ({len(df)}, {df.shape[1]})")

write_hist(['h3_f_p12', 'h3_f_p23', 'h3_f_p13'], 'h3f', "muni_h3_fixed.parquet")
write_hist(['h3_q_p12', 'h3_q_p23', 'h3_q_p13'], 'h3q', "muni_h3_quantile.parquet")
write_hist(['h2_f_p12'], 'h2f', "muni_h2_fixed.parquet")
write_hist(['h2_q_p12'], 'h2q', "muni_h2_quantile.parquet")
print("done")
