#!/usr/bin/env bash
set -euo pipefail
# Usage: convert_and_quantize.sh <hf_dir> <openai_dir> <wcpp_dir> <out_dir> <out_file> <quant>
HF_DIR=${1:?} OPENAI_DIR=${2:?} WCPP=${3:?} OUT_DIR=${4:?} OUT_FILE=${5:?} QUANT=${6:?}
BUILD="$WCPP/build"

for f in config.json vocab.json added_tokens.json; do
  [[ -s "$HF_DIR/$f" ]] || { echo "FATAL: missing $HF_DIR/$f"; exit 1; }
done

pick() { for c in "$@"; do [[ -x "$c" ]] && { echo "$c"; return 0; }; done; return 1; }
QUANTIZE=$(pick "$BUILD/bin/whisper-quantize" "$BUILD/bin/quantize") \
  || { echo "FATAL: quantize binary not found"; exit 1; }

mkdir -p "$OUT_DIR" "$(dirname "$OUT_FILE")"
rm -f "$OUT_DIR/ggml-model.bin" "$OUT_DIR/ggml-model-f32.bin"

# EXACTLY three positional args. A 4th of any value forces f32.
python3 "$WCPP/models/convert-h5-to-ggml.py" "$HF_DIR" "$OPENAI_DIR" "$OUT_DIR"

F16="$OUT_DIR/ggml-model.bin"
[[ -s "$F16" ]] || { echo "FATAL: expected $F16"; exit 1; }

"$QUANTIZE" "$F16" "$OUT_FILE" "$QUANT"
[[ -s "$OUT_FILE" ]] || { echo "FATAL: quantization produced no output"; exit 1; }

printf 'f16_bytes=%s\nquant_bytes=%s\n' \
  "$(stat -c%s "$F16")" "$(stat -c%s "$OUT_FILE")" | tee "$OUT_DIR/SIZES.txt"
