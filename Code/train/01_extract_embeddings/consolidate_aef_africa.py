"""
consolidate_aef_africa.py
=========================
Consolidate GEE-exported CSVs into parquet files for downstream modeling.

Run after downloading CSVs from the three Google Drive folders:
  aef_africa_adm2_mean/         -> aef_africa_adm2_mean.parquet
  aef_africa_adm2_hist/         -> aef_africa_adm2_hist.parquet
  aef_africa_adm2_binned_hist/  -> aef_africa_adm2_binned_hist.parquet

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 consolidate_aef_africa.py --drive_dir ~/Google_Drive_Downloads

Author: Jay Sayre
"""

import os, glob, sys, argparse
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)


# -- Directories -----------------------------------------------
home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
hvstat_dir  =  os.path.join(proj_dir, "Data", "HarvestStat_Africa")


def consolidate_folder(folder_path, out_name, id_cols=('fnid', 'year')):
    """Read all CSVs in folder, stack, deduplicate, save parquet."""
    csvs  =  sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csvs:
        print(f"  SKIP: no CSVs in {folder_path}")
        return None

    print(f"  Reading {len(csvs)} CSVs from {os.path.basename(folder_path)}...")
    dfs  =  []
    for f in csvs:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"    WARNING: skipping {os.path.basename(f)}: {e}")

    df  =  pd.concat(dfs, ignore_index=True)

    # Deduplicate (re-runs may produce overlapping exports)
    before  =  len(df)
    df  =  df.drop_duplicates(subset=list(id_cols))
    if len(df) < before:
        print(f"    Dropped {before - len(df):,} duplicate rows")

    # Ensure year is integer
    if 'year' in df.columns:
        df['year']  =  df['year'].astype(int)

    out_path  =  os.path.join(hvstat_dir, out_name)
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"    {len(df):,} rows, {df['fnid'].nunique()} regions, "
          f"{df['year'].nunique()} years")
    return df


def main():
    parser  =  argparse.ArgumentParser()
    parser.add_argument('--drive_dir', type=str, required=True,
                        help='Path to Google Drive download directory')
    args  =  parser.parse_args()

    drive  =  args.drive_dir

    print("Consolidating AEF Africa features...\n")

    # Mean embeddings (64 features)
    mean_dir  =  os.path.join(drive, 'aef_africa_adm2_mean')
    consolidate_folder(mean_dir, 'aef_africa_adm2_mean.parquet')

    # Percentile + stdDev (384 features)
    hist_dir  =  os.path.join(drive, 'aef_africa_adm2_hist')
    consolidate_folder(hist_dir, 'aef_africa_adm2_hist.parquet')

    # Binned histograms (512 features)
    bin_dir  =  os.path.join(drive, 'aef_africa_adm2_binned_hist')
    consolidate_folder(bin_dir, 'aef_africa_adm2_binned_hist.parquet')

    print("\nDone.")


if __name__ == '__main__':
    main()
