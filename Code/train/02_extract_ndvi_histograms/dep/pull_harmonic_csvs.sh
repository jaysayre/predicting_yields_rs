#!/usr/bin/env bash
# pull_harmonic_csvs.sh — Download completed harmonic-feature CSV exports from
# Google Drive to local disk, robustly and resume-safely.
#
# WHY THIS EXISTS (do not "simplify" back to `rclone copy`):
#   GEE Export.table.toDrive scatters exports across THOUSANDS of duplicate-named
#   Drive folders (all literally `adc_harmonic_features_16b_h3`). `rclone copy`/
#   `sync` hits "Duplicate directory found in source - ignoring" and silently
#   keeps only ONE folder -> ~99.9% data loss. So we pull BY FILE-ID instead,
#   driven by the authoritative list of SUCCEEDED EE tasks.
#
# Method:
#   1. authoritative filenames  = EE listOperations, state==SUCCEEDED,
#                                 desc startswith harmfeat_  ->  {desc}.csv
#   2. name->id map             = streaming `rclone lsf ... --format 'pi'`
#                                 (NO --fast-list: it blanks the name field;
#                                  NOT lsjson: multi-MB blob corrupts on parse)
#   3. download                 = `rclone backend copyid gdrive: <id> <dest>/ ...`
#                                 (top-level `copyid` absent in this rclone; the
#                                  drive-backend one takes many id/path pairs)
#   4. rate-limit safe          = -P 2, --tpslimit 6, gentle pacer, resume-safe
#                                 (re-run backfills anything rateLimitExceeded
#                                  dropped; already-local files are skipped)
#
# Usage:
#   ./pull_harmonic_csvs.sh                 # pull adc + cimmyt
#   ./pull_harmonic_csvs.sh adc             # just one level
#   PROJECTS="ds421reproducibilityproject ee-sayrejay poppydetection" \
#     REMOTE=gdrive ./pull_harmonic_csvs.sh
set -euo pipefail

PY=${PY:-~/miniforge3/envs/geo_env/bin/python}
REMOTE=${REMOTE:-gdrive}
# MUST list every project any batch was submitted to. A batch that succeeded only
# on a project missing from this list is silently absent from the authoritative
# filename set in step 1 -> never fetched, and the "N/N local" check still passes
# because N is itself computed from the short list. (Bit us once: yield-predict
# won the last 5 state-31 batches and the pull reported a clean 4779/4779.)
PROJECTS=${PROJECTS:-"ds421reproducibilityproject ee-sayrejay poppydetection yield-predict"}
LEVELS=${*:-"adc cimmyt"}
DST=${DST:-~/Dropbox/Projects/Maize_prediction/Data/harmonic_features/csvs}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ── 1. authoritative SUCCEEDED filenames from EE ─────────────────────────────
echo "[1/4] listing SUCCEEDED harmfeat tasks across: $PROJECTS"
"$PY" - "$WORK" $PROJECTS <<'PYEOF'
import ee, sys, warnings; warnings.filterwarnings('ignore')
work, projects = sys.argv[1], sys.argv[2:]
adc, cim = set(), set()
for p in projects:
    ee.Initialize(project=p)
    for o in ee.data.listOperations():
        md = o.get('metadata', {}); d = md.get('description', '')
        if md.get('state') == 'SUCCEEDED' and d.startswith('harmfeat_'):
            (cim if d.startswith('harmfeat_cimmyt_') else adc).add(d + '.csv')
open(f"{work}/expected_adc.txt", "w").write("\n".join(sorted(adc)))
open(f"{work}/expected_cimmyt.txt", "w").write("\n".join(sorted(cim)))
print(f"      adc={len(adc)} cimmyt={len(cim)}")
PYEOF

# ── 2. name->id map (streaming lsf; no --fast-list, no lsjson) ────────────────
echo "[2/4] building name->id map (traverses all duplicate folders; minutes)"
rclone lsf "$REMOTE": -R --include 'harmfeat_adc_*.csv' --include 'harmfeat_cimmyt_*.csv' \
    --format 'pi' --separator '|' 2>/dev/null \
  | awk -F'|' '{n=split($1,a,"/"); nm=a[n]; if(nm!="" && !(nm in s)){s[nm]=1; print nm"\t"$2}}' \
  > "$WORK/name2id.tsv"
echo "      mapped $(wc -l < "$WORK/name2id.tsv") unique names"

# ── 3+4. intersect + download by ID, resume-safe, rate-limit gentle ──────────
for kind in $LEVELS; do
  dest="$DST/$kind"; mkdir -p "$dest"
  # emit id + dest-dir pairs only for expected files not already local
  "$PY" - "$WORK" "$kind" "$dest" <<'PYEOF' > "$WORK/dl_$kind.args"
import sys, os
work, kind, dest = sys.argv[1], sys.argv[2], sys.argv[3]
m = {}
for line in open(f"{work}/name2id.tsv"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2 and p[1]: m[p[0]] = p[1]
exp = [x.strip() for x in open(f"{work}/expected_{kind}.txt") if x.strip()]
miss = [n for n in exp if n not in m]
if miss: sys.stderr.write(f"  WARNING {kind}: {len(miss)} expected files have no Drive ID (sample {miss[:3]})\n")
for n in exp:
    if n in m and not os.path.exists(os.path.join(dest, n)):
        print(m[n]); print(dest + "/")
PYEOF
  npair=$(( $(wc -l < "$WORK/dl_$kind.args") / 2 ))
  echo "[3/4] $kind: $npair to fetch -> $dest"
  if [ "$npair" -gt 0 ]; then
    xargs -P 2 -n 40 rclone backend copyid "$REMOTE": \
      --tpslimit 6 --drive-pacer-min-sleep 100ms --drive-pacer-burst 1 \
      --retries 5 --low-level-retries 10 < "$WORK/dl_$kind.args" 2>/dev/null || true
  fi
  have=$(find "$dest" -name "harmfeat_${kind}_*.csv" | wc -l)
  want=$(grep -c . "$WORK/expected_$kind.txt")
  echo "[4/4] $kind: $have / $want local  (re-run to backfill any rate-limited stragglers)"
done
echo "done."
