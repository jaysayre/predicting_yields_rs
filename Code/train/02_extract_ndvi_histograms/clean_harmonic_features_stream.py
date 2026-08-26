"""
clean_harmonic_features_stream.py — Streaming/parallel version of
`clean_harmonic_features.py`, for the full 2017-2024 ADC panel.

WHY THIS EXISTS (do not "simplify" back to the in-memory version):
The original reads every CSV into one DataFrame (`pd.concat`) before parsing.
At full scale that is ~74 GB of histogram *strings* in RAM, plus ~20 GB for the
parsed dict and ~7 GB for the widest matrix — >100 GB against 41 GB available,
i.e. a guaranteed OOM. Measured: one 500-row CSV is 15.4 MB in memory, x4,784.

This version instead parses ONE CSV at a time in a worker pool and appends each
batch to open `pyarrow.ParquetWriter`s, so peak memory is O(one batch) rather
than O(whole panel). Outputs are byte-for-byte the same schema as the original.

Outputs (same five as the original, written to Data/harmonic_features/):
  <level>_harmonic_coefs.parquet      7 coefficient columns
  <level>_h3_fixed.parquet            768 cols (p12|p23|p13 x 256)
  <level>_h3_quantile.parquet         768 cols
  <level>_h2_fixed.parquet            256 cols
  <level>_h2_quantile.parquet         256 cols

Usage:
  ~/miniforge3/envs/geo_env/bin/python clean_harmonic_features_stream.py \
      --csv_dir <downloaded_csvs> --level {muni|adc|cimmyt} [--workers 24]
"""
import os
import glob
import argparse
import multiprocessing as mp

import numpy         as np
import pandas        as pd
import pyarrow       as pa
import pyarrow.parquet as pq

parser =  argparse.ArgumentParser()
parser.add_argument('--csv_dir', required=True, help='dir of GEE-exported CSVs')
parser.add_argument('--out_dir', default=None, help='defaults to Data/harmonic_features')
parser.add_argument('--level', choices=['muni', 'adc', 'cimmyt'], required=True)
parser.add_argument('--bins', type=int, default=16)
parser.add_argument('--harmonics', type=int, default=3)
parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 4))
parser.add_argument('--compression', default='zstd')
args =  parser.parse_args()

BINS =  args.bins
NSQ  =  BINS * BINS
ID   =  {'muni': 'muncode', 'adc': 'adcid', 'cimmyt': 'plot_id'}[args.level]
COEF =  ['h_const'] + sum([[f'h_cos{k}', f'h_sin{k}'] for k in range(1, args.harmonics + 1)], [])

home_dir =  os.path.expanduser("~")
data_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction", "Data")
out_dir  =  args.out_dir or os.path.join(data_dir, "harmonic_features")
os.makedirs(out_dir, exist_ok=True)

# name -> (hist source columns, column prefix)  — mirrors the original's build()
GROUPS =  {
    'h3_fixed':    (['h3_f_p12', 'h3_f_p23', 'h3_f_p13'], 'h3f'),
    'h3_quantile': (['h3_q_p12', 'h3_q_p23', 'h3_q_p13'], 'h3q'),
    'h2_fixed':    (['h2_f_p12'],                          'h2f'),
    'h2_quantile': (['h2_q_p12'],                          'h2q'),
}
HIST_COLS =  [c for cols, _ in GROUPS.values() for c in cols]


def parse_hist(cell):
    """GEE fixedHistogram string -> normalized count vector (len NSQ), or None."""
    if pd.isna(cell) or cell in ('', None):
        return None
    s =  str(cell).replace('[[', '[').replace(']]', ']').strip('][')
    counts =  []
    for entry in s.split('], ['):
        parts =  entry.split(', ')
        if len(parts) >= 2:
            try:
                counts.append(float(parts[1]))
            except ValueError:
                pass
    if not counts:
        return None
    a =  np.zeros(NSQ, dtype=np.float32)
    a[:len(counts)] =  counts[:NSQ]
    tot =  a.sum()
    return a / tot if tot > 0 else None


def process_file(path):
    """Parse one CSV -> {group: (ids, years, matrix)} + coefs. Runs in a worker."""
    try:
        df =  pd.read_csv(path)
    except Exception as e:
        return path, None, f"read failed: {e}"
    if ID not in df.columns:
        return path, None, f"no {ID} column"
    df[ID] =  df[ID].astype(str)
    df =  df.rename(columns={'gs_year': 'year'})

    parsed =  {h: df[h].apply(parse_hist) if h in df.columns else pd.Series([None] * len(df))
               for h in HIST_COLS}

    out =  {}
    for gname, (hcols, prefix) in GROUPS.items():
        ok =  np.ones(len(df), dtype=bool)
        for h in hcols:
            ok &=  parsed[h].notna().values
        if not ok.any():
            out[gname] =  None
            continue
        blocks =  []
        for h in hcols:
            arr =  parsed[h][ok]
            blocks.append(np.vstack([a for a in arr]))
        out[gname] =  (df.loc[ok, ID].to_numpy(), df.loc[ok, 'year'].to_numpy(),
                       np.hstack(blocks))

    cf =  df.dropna(subset=[c for c in COEF if c in df.columns])
    coefs =  (cf[ID].to_numpy(), cf['year'].to_numpy(),
              cf[COEF].to_numpy(dtype=np.float64)) if len(cf) else None
    return path, (out, coefs), None


def group_cols(gname):
    hcols, prefix =  GROUPS[gname]
    return [f"{prefix}_{h.split('_')[-1]}_b{j:03d}" for h in hcols for j in range(NSQ)]


def main():
    files =  sorted(glob.glob(os.path.join(args.csv_dir, f"harmfeat_{args.level}_*.csv")))
    print(f"{len(files)} CSVs for level={args.level}; workers={args.workers}", flush=True)
    if not files:
        raise SystemExit("no harmfeat_*.csv found")

    schemas =  {g: pa.schema([(ID, pa.string()), ('year', pa.int32())] +
                             [(c, pa.float32()) for c in group_cols(g)]) for g in GROUPS}
    schemas['coefs'] =  pa.schema([(ID, pa.string()), ('year', pa.int32())] +
                                  [(c, pa.float64()) for c in COEF])

    paths =  {g: os.path.join(out_dir, f"{args.level}_{g}.parquet") for g in GROUPS}
    paths['coefs'] =  os.path.join(out_dir, f"{args.level}_harmonic_coefs.parquet")
    writers =  {k: pq.ParquetWriter(p, schemas[k], compression=args.compression)
                for k, p in paths.items()}

    nrows =  {k: 0 for k in writers}
    bad   =  []
    try:
        with mp.Pool(args.workers) as pool:
            for i, (path, res, err) in enumerate(pool.imap_unordered(process_file, files,
                                                                     chunksize=1), 1):
                if err:
                    bad.append((os.path.basename(path), err)); continue
                out, coefs =  res
                for g in GROUPS:
                    if out[g] is None:
                        continue
                    ids, yrs, mat =  out[g]
                    arrays =  [pa.array(ids, pa.string()),
                               pa.array(yrs.astype(np.int32), pa.int32())]
                    arrays += [pa.array(mat[:, j], pa.float32()) for j in range(mat.shape[1])]
                    writers[g].write_table(pa.Table.from_arrays(arrays, schema=schemas[g]))
                    nrows[g] +=  len(ids)
                if coefs is not None:
                    ids, yrs, mat =  coefs
                    arrays =  [pa.array(ids, pa.string()),
                               pa.array(yrs.astype(np.int32), pa.int32())]
                    arrays += [pa.array(mat[:, j], pa.float64()) for j in range(mat.shape[1])]
                    writers['coefs'].write_table(pa.Table.from_arrays(arrays,
                                                                      schema=schemas['coefs']))
                    nrows['coefs'] +=  len(ids)
                if i % 200 == 0:
                    print(f"  {i}/{len(files)} files; rows so far "
                          f"{ {k: f'{v:,}' for k, v in nrows.items()} }", flush=True)
    finally:
        for w in writers.values():
            w.close()

    print("\nwrote:")
    for k, p in paths.items():
        print(f"  {os.path.basename(p):<38} {nrows[k]:>10,} rows  "
              f"{os.path.getsize(p)/1e9:.2f} GB")
    if bad:
        print(f"\n{len(bad)} problem files (first 10): {bad[:10]}")
    print("done")


if __name__ == '__main__':
    main()
