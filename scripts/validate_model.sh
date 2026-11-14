#!/usr/bin/env bash
set -euo pipefail
# Usage: validate_model.sh <cli> <model.bin> <samples_lang_dir> <lang> [f16.bin] [vad.bin] [threads]
CLI=${1:?} MODEL=${2:?} SAMPLES=${3:?} LANG_CODE=${4:?}
F16=${5:-} VAD=${6:-} THREADS=${7:-4}

[[ -f "$SAMPLES/expected-transcripts.txt" ]] \
  || { echo "FATAL: missing $SAMPLES/expected-transcripts.txt"; exit 1; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/quant" "$WORK/f16"

# Flags mirror the JNI decode config
run() {
  local model=$1 wav=$2 out=$3
  local extra=()
  if [[ -n "$VAD" ]]; then extra=(--vad -vm "$VAD"); fi
  "$CLI" -m "$model" -f "$wav" -t "$THREADS" -l "$LANG_CODE" \
    --no-timestamps --no-prints --best-of 1 --temperature 0.0 \
    ${extra[@]+"${extra[@]}"} -otxt -of "$out" >/dev/null 2>"$out.log" \
    || { echo "FATAL: inference failed on $wav"; cat "$out.log"; return 1; }
}

while IFS=$'\t' read -r wav _ || [[ -n "${wav:-}" ]]; do
  if [[ -z "${wav// }" || "$wav" == \#* ]]; then continue; fi
  run "$MODEL" "$SAMPLES/$wav" "$WORK/quant/${wav%.wav}"
  if [[ -n "$F16" ]]; then run "$F16" "$SAMPLES/$wav" "$WORK/f16/${wav%.wav}"; fi
done < "$SAMPLES/expected-transcripts.txt"

HAS_F16=$([[ -n "$F16" ]] && echo 1 || echo 0) python3 - \
  "$SAMPLES/expected-transcripts.txt" "$WORK/quant" "$WORK/f16" <<'PY'
import os, re, sys, pathlib
exp_file, qdir, fdir = sys.argv[1], sys.argv[2], sys.argv[3]
if os.environ.get("HAS_F16") != "1":
    fdir = None

norm = lambda s: re.sub(r"[^\w\s']", "", s.lower()).split()

def wer(ref, hyp):
    d = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    return d[-1][-1], max(len(ref), 1)

qe = qn = fe = fn = 0
fail = []
for line in pathlib.Path(exp_file).read_text().splitlines():
    if not line.strip() or line.startswith("#"): continue
    wav, _, expected = line.partition("\t")
    exp = expected.strip()
    q = pathlib.Path(f"{qdir}/{wav[:-4]}.txt")
    if not q.exists():
        fail.append(f"{wav}: no transcript produced"); continue
    qt = q.read_text().strip()
    if not exp or exp == "<silence>":
        if qt: fail.append(f"{wav}: silence produced {qt!r} (must be empty)")
        print(f"[silence] {wav}: {'OK' if not qt else 'FAIL'}")
        continue
    e, n = wer(norm(exp), norm(qt)); qe += e; qn += n
    print(f"[quant] {wav}: {qt!r} ({e}/{n})")
    if fdir:
        f = pathlib.Path(f"{fdir}/{wav[:-4]}.txt")
        if not f.exists():
            fail.append(f"{wav}: no f16 transcript produced"); continue
        e2, n2 = wer(norm(exp), norm(f.read_text().strip())); fe += e2; fn += n2

qwer = qe / (qn or 1)
print(f"quant WER = {qwer:.3f}")
if fdir and fn:
    fwer = fe / fn
    print(f"f16 WER   = {fwer:.3f}  delta = {qwer - fwer:+.3f}")
    if qwer - fwer > 0.05:
        fail.append(f"quant regressed >5pts vs f16 ({qwer:.3f} vs {fwer:.3f})")
if fail:
    print("\n".join("FAIL: " + f for f in fail)); sys.exit(1)
print("validation OK")
PY
