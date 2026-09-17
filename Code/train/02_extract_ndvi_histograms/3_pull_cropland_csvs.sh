#!/usr/bin/env bash
# 3_pull_cropland_csvs.sh — download completed ADC cropland-feature CSV exports
# (2_ls_cropland_features.py; desc `cropfeat<MTAG>_adc_*`) from Google Drive.
#
# Same file-ID method as pull_harmonic_csvs.sh (see its header): GEE toDrive
# scatters exports across thousands of duplicate-named folders, so `rclone copy`
# silently keeps one -> pull BY FILE-ID driven by the SUCCEEDED EE task list.
#
# Usage:
#   TAG=crop_aefn2 PROJECTS="<gcp-project-id>" ./3_pull_cropland_csvs.sh   # pull the aefn2 run
#   TAG=crop_aefn2 PROJECTS="<gcp-project-id> ..." ./3_pull_cropland_csvs.sh
set -euo pipefail

PY=${PY:-~/miniforge3/envs/geo_env/bin/python}
REMOTE=${REMOTE:-gdrive}
TAG=${TAG:-crop_aefn2}                       # desc/folder suffix (MTAG without leading _)
PREFIX="cropfeat_${TAG}_adc_"                # EE description prefix to match
PROJECTS=${PROJECTS:?set PROJECTS to the space-separated Earth Engine project ids the exports were submitted from}
DST=${DST:-~/Dropbox/Projects/Maize_prediction/Data/cropland_features/csvs_${TAG}}
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$DST"

# ── 1. authoritative SUCCEEDED filenames from EE ─────────────────────────────
echo "[1/4] listing SUCCEEDED $PREFIX* tasks across: $PROJECTS"
"$PY" - "$WORK" "$PREFIX" $PROJECTS <<'PYEOF'
import ee, sys, warnings; warnings.filterwarnings('ignore')
work, prefix, projects = sys.argv[1], sys.argv[2], sys.argv[3:]
exp = set()
for p in projects:
    ee.Initialize(project=p)
    for o in ee.data.listOperations():
        md = o.get('metadata', {}); d = md.get('description', '')
        if md.get('state') == 'SUCCEEDED' and d.startswith(prefix):
            exp.add(d + '.csv')
open(f"{work}/expected.txt", "w").write("\n".join(sorted(exp)))
print(f"      expected={len(exp)}")
PYEOF

# ── 2. name->id map (streaming lsf; no --fast-list, no lsjson) ────────────────
echo "[2/4] building name->id map (traverses duplicate folders; a few min)"
rclone lsf "$REMOTE": -R --include "${PREFIX}*.csv" \
    --format 'pi' --separator '|' 2>/dev/null \
  | awk -F'|' '{n=split($1,a,"/"); nm=a[n]; if(nm!="" && !(nm in s)){s[nm]=1; print nm"\t"$2}}' \
  > "$WORK/name2id.tsv"
echo "      mapped $(wc -l < "$WORK/name2id.tsv") unique names"

# ── 3+4. intersect + download by ID, resume-safe ────────────────────────────
"$PY" - "$WORK" "$DST" <<'PYEOF' > "$WORK/dl.args"
import sys, os
work, dest = sys.argv[1], sys.argv[2]
m = {}
for line in open(f"{work}/name2id.tsv"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2 and p[1]: m[p[0]] = p[1]
exp = [x.strip() for x in open(f"{work}/expected.txt") if x.strip()]
miss = [n for n in exp if n not in m]
if miss: sys.stderr.write(f"  WARNING: {len(miss)} expected files have no Drive ID (sample {miss[:3]})\n")
for n in exp:
    if n in m and not os.path.exists(os.path.join(dest, n)):
        print(m[n]); print(dest + "/")
PYEOF
npair=$(( $(wc -l < "$WORK/dl.args") / 2 ))
echo "[3/4] $npair to fetch -> $DST"
if [ "$npair" -gt 0 ]; then
  xargs -P 2 -n 40 rclone backend copyid "$REMOTE": \
    --tpslimit 6 --drive-pacer-min-sleep 100ms --drive-pacer-burst 1 \
    --retries 5 --low-level-retries 10 < "$WORK/dl.args" 2>/dev/null || true
fi
have=$(find "$DST" -name "${PREFIX}*.csv" | wc -l)
want=$(grep -c . "$WORK/expected.txt")
echo "[4/4] $have / $want local  (re-run to backfill rate-limited stragglers)"
