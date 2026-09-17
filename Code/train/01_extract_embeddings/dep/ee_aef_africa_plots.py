"""
ee_aef_africa_plots.py
======================
Extract AEF embedding features for GROW-Africa plot locations via GEE.

Uses a 500m buffer around each plot to capture distributional features
(multiple pixels). Extracts three feature types:
  1. Mean embeddings:       64 features  (A00-A63)
  2. Percentile + stdDev:  384 features  (64 dims x 6 stats)
  3. Binned histograms:    512 features  (64 dims x 8 bins)

Batches by country x year to stay within GEE payload limits.

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate ML_env
  python3 ee_aef_africa_plots.py
  python3 ee_aef_africa_plots.py --type mean
  python3 ee_aef_africa_plots.py --type binned --batch_size 2000

Author: Jay Sayre
"""

import os, sys, time, argparse

import ee
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# Configuration
# ============================================================
BUFFER_M    =  500     # meters around each plot
SCALE       =  10
N_BINS      =  8
BIN_MIN     =  -0.8
BIN_MAX     =  0.8
BATCH_SIZE  =  3000    # plots per GEE task

# Drive folders
FOLDER_MEAN    =  'aef_africa_plots_mean'
FOLDER_HIST    =  'aef_africa_plots_hist'
FOLDER_BINNED  =  'aef_africa_plots_binned'
# ============================================================

BIN_WIDTH  =  (BIN_MAX - BIN_MIN) / N_BINS

home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
grow_dir    =  os.path.join(proj_dir, "Data", "GROW_Africa")

feat_names  =  [f"A{d:02d}" for d in range(64)]

# Column names
mean_selectors  =  feat_names + ['plot_id', 'year']

pct_cols  =  []
for d in feat_names:
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"{d}{s}")
pct_selectors  =  pct_cols + ['plot_id', 'year']

bin_cols  =  [f"{d}_b{b}" for d in feat_names for b in range(N_BINS)]
bin_selectors  =  bin_cols + ['plot_id', 'year']


def build_indicator_image(img):
    """512-band indicator image for binned histograms."""
    bands  =  []
    for dim_name in feat_names:
        band  =  img.select(dim_name)
        for b in range(N_BINS):
            lo  =  BIN_MIN + b * BIN_WIDTH
            hi  =  lo + BIN_WIDTH
            if b < N_BINS - 1:
                indicator  =  band.gte(lo).And(band.lt(hi))
            else:
                indicator  =  band.gte(lo).And(band.lte(hi))
            bands.append(indicator.rename(f"{dim_name}_b{b}"))
    return ee.Image(bands).toFloat()


def submit_task(results, desc, folder, selectors):
    """Submit GEE export with queue-full retry."""
    task  =  ee.batch.Export.table.toDrive(
        collection=results, description=desc,
        folder=folder, selectors=selectors
    )
    for attempt in range(10):
        try:
            task.start()
            return True
        except ee.ee_exception.EEException as e:
            msg  =  str(e).lower()
            if 'too many tasks' in msg:
                wait  =  300 * (attempt + 1)
                print(f"  queue full, wait {wait // 60}m...",
                      end='', flush=True)
                time.sleep(wait)
            elif 'payload size' in msg:
                print(f"  PAYLOAD TOO LARGE")
                return False
            else:
                print(f"  ERROR: {e}")
                return False
    return False


def main():
    parser  =  argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'mean', 'hist', 'binned'])
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--start_country', type=str, default=None)
    args  =  parser.parse_args()

    do_mean    =  args.type in ('all', 'mean')
    do_hist    =  args.type in ('all', 'hist')
    do_binned  =  args.type in ('all', 'binned')

    ee.Authenticate()
    ee.Initialize(project='avocadoyieldsdeforestation')

    alpha_earth  =  ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    print(f"Buffer: {BUFFER_M}m, Scale: {SCALE}m, Batch: {args.batch_size}")
    print(f"Features: mean={do_mean}, hist={do_hist}, binned={do_binned}")

    # -- Load plot data ----------------------------------------
    plots  =  pd.read_parquet(os.path.join(grow_dir, "grow_africa_maize_plots.parquet"))
    print(f"\nLoaded {len(plots):,} plots across {plots['country'].nunique()} countries")

    # Build EE features with 500m buffer
    plots['eeGeom']  =  plots.apply(
        lambda r: ee.Geometry.Point([r['lon'], r['lat']]).buffer(BUFFER_M),
        axis=1
    )
    plots['eeFeature']  =  plots.apply(
        lambda r: ee.Feature(r['eeGeom'], {
            'plot_id': int(r['plot_id']),
            'year': int(r['year'])
        }), axis=1
    )

    counts  =  {'mean': 0, 'hist': 0, 'binned': 0}

    for country in sorted(plots['country'].unique()):
        if args.start_country and country < args.start_country:
            continue

        c_plots  =  plots[plots['country'] == country]

        for year in sorted(c_plots['year'].unique()):
            yp  =  c_plots[c_plots['year'] == year]
            n_batches  =  (len(yp) + args.batch_size - 1) // args.batch_size

            for b in range(n_batches):
                batch  =  yp.iloc[b * args.batch_size : (b + 1) * args.batch_size]
                blabel =  f"_b{b}" if n_batches > 1 else ""

                coll  =  ee.FeatureCollection(batch['eeFeature'].tolist())
                img   =  alpha_earth.filterDate(
                    f'{year}-01-01', f'{year}-12-31'
                ).mean()
                # No agland mask — plots are known agricultural locations

                # -- Mean --
                if do_mean:
                    desc  =  f'aef_plots_mean_{country}_{year}{blabel}'
                    print(f"  {desc} ({len(batch)} plots)...",
                          end='', flush=True)
                    results  =  img.reduceRegions(
                        collection=coll, reducer=ee.Reducer.mean(),
                        scale=SCALE
                    )
                    if submit_task(results, desc, FOLDER_MEAN, mean_selectors):
                        counts['mean'] += 1
                        print(f" #{counts['mean']}")
                    else:
                        print()

                # -- Percentile + stdDev --
                if do_hist:
                    desc  =  f'aef_plots_hist_{country}_{year}{blabel}'
                    print(f"  {desc}...", end='', flush=True)
                    reducer  =  (ee.Reducer.percentile([10, 25, 50, 75, 90])
                                 .combine(ee.Reducer.stdDev(),
                                          sharedInputs=True))
                    results  =  img.reduceRegions(
                        collection=coll, reducer=reducer, scale=SCALE
                    )
                    if submit_task(results, desc, FOLDER_HIST, pct_selectors):
                        counts['hist'] += 1
                        print(f" #{counts['hist']}")
                    else:
                        print()

                # -- Binned histogram --
                if do_binned:
                    desc  =  f'aef_plots_binned_{country}_{year}{blabel}'
                    print(f"  {desc}...", end='', flush=True)
                    indicator_img  =  build_indicator_image(img)
                    results  =  indicator_img.reduceRegions(
                        collection=coll, reducer=ee.Reducer.mean(),
                        scale=SCALE
                    )
                    if submit_task(results, desc, FOLDER_BINNED, bin_selectors):
                        counts['binned'] += 1
                        print(f" #{counts['binned']}")
                    else:
                        print()

    print(f"\n{'=' * 55}")
    print(f"  Tasks: mean={counts['mean']}, hist={counts['hist']}, "
          f"binned={counts['binned']}")
    print(f"  Total: {sum(counts.values())}")
    print(f"  Monitor: https://code.earthengine.google.com/tasks")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()
