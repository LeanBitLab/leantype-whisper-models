Save as `guide.md`. This consolidates every correction from the review chain  -  including the `project(`-anchored version match, the functional `models.json` override path, and the fail-closed license gate ahead of the build.

---

# LeanType Voice  -  Whisper Model Build & Integration Plan (v1)

**Repo:** `LeanBitLab/leantype-whisper-models` · **Consumer:** LeanType Android IME (`:voice` process)
**Status:** v1 scope locked; commit-ready pending the first dry dispatch.

## 1. Purpose & scope

Convert, quantize, validate, and release compact Whisper checkpoints for offline voice typing in LeanType. Building rather than mirroring `ggml-org` binaries buys verifiable provenance (F-Droid) and lets us ship fine-tuned/regional checkpoints upstream doesn't distribute.

**v1 goals:** 100% local inference · on-demand download · bounded peak RAM · short-utterance accuracy · redistributable licensing · pinned, auditable builds.

**Non-goals for v1:** streaming partials, translation, diarization, GPU/NPU backends, bundled models of any size.

## 2. Verified facts & traps

Each confirmed against upstream source. Do not "simplify" these.

| Area             | Fact                                                                                                                      | Consequence                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Converter argv   | `convert-h5-to-ggml.py dir_model path-to-whisper-repo dir-output [use-f32]`                                               | Pass **exactly 3** args; a 4th of *any* value forces f32 (`len(sys.argv) > 4`). |
| Converter output | Always `<dir-out>/ggml-model.bin`                                                                                         | Literal path; no `find \| head`.                                                |
| Converter inputs | Needs `config.json`, `vocab.json`, `added_tokens.json`; weights via `WhisperForConditionalGeneration.from_pretrained()`   | safetensors works; **transformers version is a build input**.                   |
| Container        | Magic `0x67676d6c` (`ggml`), bare hparam header, **no version field**. Not GGUF.                                          | `min_runtime_version` is structural.                                            |
| Memory           | `whisper_context_params` has **no `use_mmap`**; weights load via `whisper_model_loader` into anon buffers                 | ~400–500 MB `Private_Dirty` → zram → LMK. Isolation + idle unload mandatory.    |
| Sampling         | Default strategy `GREEDY`; `beam_search.beam_size` ignored under it; **`greedy.best_of` defaults to 5**                   | Set `best_of = 1`.                                                              |
| Fallback         | `temperature_inc` defaults `0.2`                                                                                          | Set `0.0` to kill the retry ladder.                                             |
| VAD model        | `ggml-silero-v5.1.2.bin` / `v6.2.0.bin`, **885 kB each** (`ggml-org/whisper-vad`)                                         | Bundle in `assets/`; pin one + hash.                                            |
| VAD APIs         | Integrated: `wparams.vad` + `vad_model_path` (**filesystem path**). Standalone: `whisper_vad_init_with_params(loader, …)` | Extract asset for the integrated path.                                          |
| VAD state        | LSTM; `whisper_vad_reset_state()`                                                                                         | Reset between utterances.                                                       |
| CMake            | No `CMAKE_BUILD_TYPE` ⇒ `-O0`; `--config Release` no-op on Makefiles; `GGML_NATIVE` defaults ON                           | Set both explicitly.                                                            |
| Actions          | `github.event.inputs.*` are **strings**; `"false"` is truthy                                                              | Compare `== 'true'`.                                                            |
| Actions          | `workflow_dispatch` caps at **10 inputs**                                                                                 | v1 uses 9; policy lives in `models.json`.                                       |
| pip              | PEP 440: `2.5.1+cpu` > `2.5.1`, and `==2.5.1` matches **both**                                                            | Pin `torch==2.5.1+cpu`.                                                         |
| pip              | `pip freeze` omits index directives                                                                                       | Prepend `--extra-index-url` to the lock.                                        |
| bash             | `while` inherits its last body command's status; trailing `[[ … ]] && cmd` returns 1                                      | `set -e` aborts at `done`. Use explicit `if`.                                   |
| License          | HF `license:` may be a string **or list**                                                                                 | Never `matched[0]`; fail closed on multi-license.                               |
| CMake version    | `cmake_minimum_required(VERSION x.y.z)` can precede `project(...)`                                                        | Anchor the match to `project(`.                                                 |
| Roadmap          | master ships `include/parakeet.h` + `src/parakeet.cpp`                                                                    | Non-30 s-window A/B is a v1.1 experiment in the *same* runtime.                 |

## 3. Repository layout

```
leantype-whisper-models/
├── .editorconfig
├── .gitattributes
├── .github/workflows/build-whisper-model.yml
├── bench/<model-id>.json           # on-device results; gates publishing
├── catalog.json                    # consumed by LeanType; entries emitted by CI
├── models.json                     # per-model license overrides
├── docs/MODEL_CARD_TEMPLATE.md
├── samples/
│   ├── README.md                   # per-clip provenance + license
│   └── en/
│       ├── sample-en-clean.wav
│       ├── sample-en-commands.wav
│       ├── sample-en-noisy.wav
│       ├── sample-short-silence.wav
│       └── expected-transcripts.txt
├── scripts/
│   ├── convert_and_quantize.sh
│   ├── validate_model.sh
│   └── benchmark_wer.py            # nightly / local
├── requirements.txt
├── requirements.lock.txt           # promoted from dry-run pip freeze
├── LICENSE                         # MIT (scripts only)
├── THIRD_PARTY_NOTICES.md
├── RELEASE_CHECKLIST.md
└── guide.md
```

`.gitattributes`  -  enforced at checkout; `.editorconfig` alone is advisory:

```gitattributes
samples/**/expected-transcripts.txt text eol=lf
*.wav binary
*.bin binary
```

`.editorconfig`:

```ini
root = true

[samples/**/expected-transcripts.txt]
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = false
```

`samples/en/expected-transcripts.txt`  -  tab-separated; silence is an explicit sentinel so no invisible whitespace carries meaning:

```text
# filename.wav<TAB>expected transcript (or <silence>)
sample-en-clean.wav    hello world
sample-en-commands.wav    new line send message
sample-en-noisy.wav    on my way
sample-short-silence.wav    <silence>
```

## 4. Licensing policy

Scripts/docs: MIT. Anything derived from HeliBoard/LeanType: GPLv3. Weights: **per-checkpoint, verified, fail-closed.**

- Auto-approve allowlist: `mit`, `apache-2.0`, `bsd-3-clause`, `cc0-1.0`.
- Anything else  -  a *list* containing an allowlisted value, `other`, CC-BY  -  halts unless `models.json` carries an explicit override with attribution.
- **Non-commercial is a hard floor**: any license containing `-nc` is rejected even with an override.
- License is read from the **pinned snapshot's** `README.md` frontmatter, *before* the build.

`models.json`:

```json
{
  "overrides": {
    "someorg/indic-whisper-small@<40-hex-revision>": {
      "license": "cc-by-4.0",
      "attribution": "Indic Whisper by <author>, licensed CC-BY-4.0.",
      "approved_by": "<maintainer>",
      "rationale": "Attribution shipped in MODEL_CARD.md and in-app About screen."
    }
  }
}
```

## 5. Build pipeline

### `requirements.txt`

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
transformers==4.46.3
huggingface_hub==0.26.2
numpy==2.1.3
safetensors==0.4.5
pyyaml==6.0.2
```

PyYAML is pinned explicitly rather than relied on transitively  -  it backs a licensing gate. Refresh pins once against your `whisper.cpp`, then freeze. After the dry run, promote `pip-freeze.txt` → `requirements.lock.txt` **with `--extra-index-url` as line 1**.

### `scripts/convert_and_quantize.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# usage: convert_and_quantize.sh <hf_dir> <openai_dir> <wcpp_dir> <out_dir> <out_file> <quant>
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
python "$WCPP/models/convert-h5-to-ggml.py" "$HF_DIR" "$OPENAI_DIR" "$OUT_DIR"

F16="$OUT_DIR/ggml-model.bin"
[[ -s "$F16" ]] || { echo "FATAL: expected $F16"; exit 1; }

"$QUANTIZE" "$F16" "$OUT_FILE" "$QUANT"
[[ -s "$OUT_FILE" ]] || { echo "FATAL: quantization produced no output"; exit 1; }

printf 'f16_bytes=%s\nquant_bytes=%s\n' \
  "$(stat -c%s "$F16")" "$(stat -c%s "$OUT_FILE")" | tee "$OUT_DIR/SIZES.txt"
```

### `scripts/validate_model.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# usage: validate_model.sh <cli> <model.bin> <samples_lang_dir> <lang> [f16.bin] [vad.bin] [threads]
CLI=${1:?} MODEL=${2:?} SAMPLES=${3:?} LANG_CODE=${4:?}
F16=${5:-} VAD=${6:-} THREADS=${7:-4}

[[ -f "$SAMPLES/expected-transcripts.txt" ]] \
  || { echo "FATAL: missing $SAMPLES/expected-transcripts.txt"; exit 1; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/quant" "$WORK/f16"

# Flags mirror the JNI decode config. Confirm against `$CLI --help` on your pinned commit.
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

HAS_F16=$([[ -n "$F16" ]] && echo 1 || echo 0) python - \
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
```

The 5-point tolerance is loose by design for ~40 words; the nightly gate uses +1.0 absolute.

### `.github/workflows/build-whisper-model.yml`

```yaml
name: Build & Release LeanType Whisper Model

on:
  workflow_dispatch:
    inputs:
      model_name:      { description: "HF repo", required: true, default: "openai/whisper-small.en" }
      model_revision:  { description: "HF commit SHA (40 hex)", required: true }
      quant_type:      { description: "q5_1 | q5_0 | q8_0 | q4_0", required: true, default: "q5_1" }
      output_prefix:   { description: "Output prefix", required: true, default: "leantype-whisper-small.en" }
      model_version:   { description: "Catalog version (positive integer)", required: true, default: "1" }
      languages:       { description: "Comma-separated ISO codes; first used for CI", required: true, default: "en" }
      whisper_cpp_ref: { description: "whisper.cpp commit SHA (40 hex)", required: true }
      openai_ref:      { description: "openai/whisper commit SHA (40 hex)", required: true }
      publish_release: { description: "Publish GitHub Release", type: boolean, required: true, default: false }

concurrency:
  group: build-${{ inputs.output_prefix }}-${{ inputs.quant_type }}
  cancel-in-progress: false

permissions: { contents: read }

jobs:
  build:
    runs-on: ubuntu-24.04
    timeout-minutes: 30
    env:
      MODEL_NAME:      ${{ inputs.model_name }}
      MODEL_REVISION:  ${{ inputs.model_revision }}
      QUANT_TYPE:      ${{ inputs.quant_type }}
      OUTPUT_PREFIX:   ${{ inputs.output_prefix }}
      MODEL_VERSION:   ${{ inputs.model_version }}
      MODEL_LANGS:     ${{ inputs.languages }}
      WCPP_REF:        ${{ inputs.whisper_cpp_ref }}
      OPENAI_REF:      ${{ inputs.openai_ref }}
      VAD_REVISION:    ${{ vars.VAD_REVISION }}
      VAD_SHA256:      ${{ vars.VAD_SHA256 }}
      HF_HUB_DISABLE_TELEMETRY: "1"
    steps:
      - uses: actions/checkout@v4

      - name: Validate inputs and repo vars
        run: |
          set -euo pipefail
          [[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "model_revision must be 40-hex"; exit 1; }
          [[ "$WCPP_REF"       =~ ^[0-9a-f]{40}$ ]] || { echo "whisper_cpp_ref must be 40-hex"; exit 1; }
          [[ "$OPENAI_REF"     =~ ^[0-9a-f]{40}$ ]] || { echo "openai_ref must be 40-hex"; exit 1; }
          [[ "$OUTPUT_PREFIX"  =~ ^[a-z0-9._-]+$ ]] || { echo "bad output_prefix"; exit 1; }
          [[ "$QUANT_TYPE"     =~ ^q[458]_[01]$  ]] || { echo "bad quant_type"; exit 1; }
          [[ "$MODEL_VERSION"  =~ ^[1-9][0-9]*$  ]] || { echo "bad model_version"; exit 1; }
          [[ "$MODEL_LANGS"    =~ ^[a-z]{2,3}(,[a-z]{2,3})*$ ]] || { echo "bad languages"; exit 1; }
          [[ "${VAD_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "set VAD_REVISION repo var"; exit 1; }
          [[ "${VAD_SHA256:-}"   =~ ^[0-9a-f]{64}$ ]] || { echo "set VAD_SHA256 repo var"; exit 1; }

      - uses: actions/checkout@v4
        with: { repository: ggml-org/whisper.cpp, ref: "${{ inputs.whisper_cpp_ref }}", path: whisper.cpp }
      - uses: actions/checkout@v4
        with: { repository: openai/whisper, ref: "${{ inputs.openai_ref }}", path: openai-whisper }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }

      - name: Install CPU Python deps
        run: |
          set -euo pipefail
          python -m pip install -r requirements.txt
          python -c "import torch; assert torch.version.cuda is None, torch.__version__; print(torch.__version__)"
          pip freeze > pip-freeze.txt

      - name: Download checkpoint + license gate
        run: |
          set -euo pipefail
          mkdir -p ./meta
          python - <<'PY'
          import json, os, pathlib, yaml
          from huggingface_hub import snapshot_download
          repo, rev = os.environ["MODEL_NAME"], os.environ["MODEL_REVISION"]
          snapshot_download(
              repo_id=repo, revision=rev, local_dir="./hf_model",
              allow_patterns=["README.md","config.json","generation_config.json","vocab.json",
                              "added_tokens.json","tokenizer*.json","special_tokens_map.json",
                              "preprocessor_config.json","model.safetensors","pytorch_model.bin"],
          )
          p = pathlib.Path("./hf_model/README.md")
          text = p.read_text(encoding="utf-8") if p.exists() else ""
          parts = text.split("---", 2)
          meta = yaml.safe_load(parts[1]) if text.startswith("---") and len(parts) >= 3 else {}
          meta = meta if isinstance(meta, dict) else {}
          raw = meta.get("license") or "unknown"
          lics = [raw.lower()] if isinstance(raw, str) else [str(x).lower() for x in raw]

          ALLOW = {"mit","apache-2.0","bsd-3-clause","cc0-1.0"}
          attribution = ""
          if len(lics) == 1 and lics[0] in ALLOW:
              lic = lics[0]
          else:
              ov = json.loads(pathlib.Path("models.json").read_text()).get("overrides", {}) \
                   if pathlib.Path("models.json").exists() else {}
              entry = ov.get(f"{repo}@{rev}")
              if not entry:
                  raise SystemExit(f"FATAL: license {lics!r} requires an explicit models.json "
                                   f"override for {repo}@{rev}")
              lic = str(entry["license"]).lower()
              attribution = entry.get("attribution", "").strip()
              if "-nc" in lic:
                  raise SystemExit(f"FATAL: non-commercial license {lic!r} is never redistributable")
              if lic.startswith("cc-by") and not attribution:
                  raise SystemExit("FATAL: CC-BY override requires attribution text")
              print(f"license override applied by {entry.get('approved_by','?')}: {entry.get('rationale','')}")

          pathlib.Path("./meta/license.txt").write_text(lic + "\n")
          pathlib.Path("./meta/attribution.txt").write_text(attribution + "\n")
          print("license:", lic)
          PY

      - name: Build whisper.cpp
        run: |
          set -euo pipefail
          cmake -S ./whisper.cpp -B ./whisper.cpp/build \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
            -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DGGML_NATIVE=OFF
          cmake --build ./whisper.cpp/build -j"$(nproc)"

      - name: Convert + quantize
        run: |
          set -euo pipefail
          chmod +x scripts/*.sh
          ./scripts/convert_and_quantize.sh ./hf_model ./openai-whisper ./whisper.cpp \
            ./converted "./release/${OUTPUT_PREFIX}-${QUANT_TYPE}.bin" "$QUANT_TYPE"

      - name: Fetch pinned VAD model
        run: |
          set -euo pipefail
          curl -fsSL -o ./ggml-silero-v5.1.2.bin \
            "https://huggingface.co/ggml-org/whisper-vad/resolve/${VAD_REVISION}/ggml-silero-v5.1.2.bin"
          echo "${VAD_SHA256}  ./ggml-silero-v5.1.2.bin" | sha256sum -c -

      - name: Validate
        run: |
          set -euo pipefail
          CLI=./whisper.cpp/build/bin/whisper-cli
          [[ -x "$CLI" ]] || CLI=./whisper.cpp/build/bin/main
          PRIMARY_LANG="${MODEL_LANGS%%,*}"
          SAMPLES_DIR="./samples/$PRIMARY_LANG"
          [[ -f "$SAMPLES_DIR/expected-transcripts.txt" ]] || { echo "FATAL: no sample set for $PRIMARY_LANG"; exit 1; }
          ./scripts/validate_model.sh "$CLI" \
            "./release/${OUTPUT_PREFIX}-${QUANT_TYPE}.bin" "$SAMPLES_DIR" "$PRIMARY_LANG" \
            ./converted/ggml-model.bin ./ggml-silero-v5.1.2.bin 4

      - name: Metadata, checksums, catalog entry, model card
        run: |
          set -euo pipefail
          WCPP_VER=$(awk '
            /^[[:space:]]*project[[:space:]]*\(/ && match($0, /VERSION[[:space:]]+[0-9]+\.[0-9]+\.[0-9]+/) {
              v = substr($0, RSTART, RLENGTH); sub(/^VERSION[[:space:]]+/, "", v); print v; exit
            }' whisper.cpp/CMakeLists.txt)
          [[ -n "$WCPP_VER" ]] || { echo "FATAL: could not determine whisper.cpp version"; exit 1; }
          WCPP_COMMIT=$(git -C whisper.cpp rev-parse HEAD)
          OPENAI_COMMIT=$(git -C openai-whisper rev-parse HEAD)

          cd release
          sha256sum ./*.bin > SHA256SUMS.txt
          {
            echo "base_model=$MODEL_NAME"
            echo "model_revision=$MODEL_REVISION"
            echo "quant_type=$QUANT_TYPE"
            echo "model_version=$MODEL_VERSION"
            echo "languages=$MODEL_LANGS"
            echo "license=$(cat ../meta/license.txt)"
            echo "whisper_cpp_version=$WCPP_VER"
            echo "min_runtime_version=$WCPP_VER"
            echo "whisper_cpp_commit=$WCPP_COMMIT"
            echo "openai_whisper_commit=$OPENAI_COMMIT"
            echo "vad_model=ggml-silero-v5.1.2.bin"
            echo "vad_revision=$VAD_REVISION"
            echo "vad_sha256=$VAD_SHA256"
            cat ../converted/SIZES.txt
            echo "build_date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "--- pip freeze ---"; cat ../pip-freeze.txt
            echo "--- compiler ---"; cc --version | head -n1
          } > BUILD_INFO.txt

          WCPP_VER="$WCPP_VER" WCPP_COMMIT="$WCPP_COMMIT" python - <<'PY'
          import hashlib, json, os, pathlib, textwrap
          prefix, quant, ver = os.environ["OUTPUT_PREFIX"], os.environ["QUANT_TYPE"], os.environ["MODEL_VERSION"]
          repo = os.environ["GITHUB_REPOSITORY"]
          mid, name = f"{prefix}-{quant}", f"{prefix}-{quant}.bin"
          data = pathlib.Path(name).read_bytes()
          sha = hashlib.sha256(data).hexdigest()
          lic = pathlib.Path("../meta/license.txt").read_text().strip()
          attr = pathlib.Path("../meta/attribution.txt").read_text().strip()
          wcv, wcc = os.environ["WCPP_VER"], os.environ["WCPP_COMMIT"]

          entry = {
            "id": mid, "version": int(ver), "file": name,
            "url": f"https://github.com/{repo}/releases/download/{mid}-v{ver}/{name}",
            "sha256": sha, "size_bytes": len(data),
            "base_model": os.environ["MODEL_NAME"],
            "model_revision": os.environ["MODEL_REVISION"],
            "quant_type": quant,
            "languages": os.environ["MODEL_LANGS"].split(","),
            "license": lic, "attribution": attr,
            "whisper_cpp_commit": wcc,
            "min_runtime_version": wcv,
          }
          pathlib.Path("catalog-entry.json").write_text(json.dumps(entry, indent=2) + "\n")

          v = pathlib.Path("../ggml-silero-v5.1.2.bin").read_bytes()
          pathlib.Path("vad-entry.json").write_text(json.dumps({
            "file": "ggml-silero-v5.1.2.bin",
            "revision": os.environ["VAD_REVISION"],
            "sha256": hashlib.sha256(v).hexdigest(),
            "size_bytes": len(v), "bundled": True,
          }, indent=2) + "\n")

          bpath = pathlib.Path(f"../bench/{mid}.json")
          b = json.loads(bpath.read_text()) if bpath.exists() else {}
          g = lambda k: b.get(k, "PENDING (device bench)")
          card = textwrap.dedent(f"""\
          # Model Card: `{name}`

          - **Model ID:** `{mid}` (v{ver})
          - **Base Model:** `{entry['base_model']}` (`{entry['model_revision']}`)
          - **Quantization:** `{quant}`
          - **Size:** {len(data):,} bytes ({len(data)/(1024*1024):.1f} MiB)
          - **SHA-256:** `{sha}`
          - **Languages:** {", ".join(entry['languages'])}
          - **License:** `{lic}` (verified from the pinned upstream model card)
          - **Built with whisper.cpp:** `{wcv}` (`{wcc}`)
          - **Minimum Runtime Version:** `{wcv}`
          - **Intended Use:** Offline voice typing in LeanType (`:voice` isolated process).

          ## Attribution
          {attr or "Upstream OpenAI Whisper weights (MIT); see THIRD_PARTY_NOTICES.md."}

          ## Device Benchmarks
          - **Peak Private_Dirty RAM:** {g('peak_private_dirty_kb')} kB
          - **Cold Load Time:** {g('cold_load_ms')} ms
          - **RTF:** {g('rtf')}
          - **Device / SoC / threads:** {g('device')} / {g('soc')} / {g('threads')}
          """)
          pathlib.Path("MODEL_CARD.md").write_text(card)
          PY

      - uses: actions/upload-artifact@v4
        with:
          name: ${{ inputs.output_prefix }}-${{ inputs.quant_type }}
          path: |
            release/*.bin
            release/SHA256SUMS.txt
            release/BUILD_INFO.txt
            release/catalog-entry.json
            release/vad-entry.json
            release/MODEL_CARD.md

  publish:
    needs: build
    if: ${{ github.event.inputs.publish_release == 'true' }}
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    permissions: { contents: write }
    steps:
      - uses: actions/checkout@v4
      - name: Require device benchmarks
        run: |
          set -euo pipefail
          B="bench/${{ inputs.output_prefix }}-${{ inputs.quant_type }}.json"
          [[ -f "$B" ]] || { echo "FATAL: $B missing  -  run the on-device bench first"; exit 1; }
      - uses: actions/download-artifact@v4
        with: { name: "${{ inputs.output_prefix }}-${{ inputs.quant_type }}", path: release }
      - name: Verify artifact checksums
        run: |
          set -euo pipefail
          cd release && sha256sum -c SHA256SUMS.txt
      - uses: softprops/action-gh-release@v2   # pin to a commit SHA before first public release
        with:
          tag_name: ${{ inputs.output_prefix }}-${{ inputs.quant_type }}-v${{ inputs.model_version }}
          name: "${{ inputs.output_prefix }}-${{ inputs.quant_type }} (v${{ inputs.model_version }})"
          body_path: release/BUILD_INFO.txt
          make_latest: false
          files: |
            release/*.bin
            release/SHA256SUMS.txt
            release/BUILD_INFO.txt
            release/catalog-entry.json
            release/vad-entry.json
            release/MODEL_CARD.md
            THIRD_PARTY_NOTICES.md
```

## 6. Catalog contract

`catalog.json` is the single source of truth: CI emits entries, the app consumes them, this guide documents them. Never hand-maintain sizes or hashes.

```json
{
  "schema": 1,
  "vad": { "file": "ggml-silero-v5.1.2.bin", "sha256": "<64-hex>", "size_bytes": 0, "bundled": true },
  "models": []
}
```

Merge `catalog-entry.json` and `vad-entry.json` by PR after each dry run.

App-side rules:

- Reject entries with `size_bytes == 0` or placeholder hashes.
- **SHA-256 is pinned in the APK** (compiled-in catalog or signed manifest). Fetching `SHA256SUMS.txt` from the same release detects corruption, not tampering.
- Refuse to load when `min_runtime_version` > the bundled runtime  -  the file carries no format version to negotiate with.
- Pin the app's `whisper.cpp` submodule to the commit recorded in the entry.

## 7. Android runtime architecture

**Delivery.** No bundled Whisper model. Download → `filesDir/models/<id>.bin.partial` → verify hash + exact size → atomic rename. WorkManager retries, unmetered by default, size + license shown before download. Exclude models from backup. Provide SAF import (copy + verify). A **dedicated offline/F-Droid flavor** with the downloader stripped is what yields zero `INTERNET` permission  -  an import option alone does not, since permissions are app-wide.

**Process.** `android:process=":voice"` bound service, `exported=false`, `foregroundServiceType="microphone"` + `FOREGROUND_SERVICE_MICROPHONE` (Android 14+), started while the IME is visible. **`AudioRecord` lives in `:voice`**  -  only text crosses Binder. Budget ~20–40 MB extra baseline for the second process.

**Memory.** Weights are anonymous memory (no mmap) → zram → LMK. Unload on IME hide plus a short idle timer behind one `BuildConfig` constant, **set from a `/proc/<pid>/smaps_rollup` reading**. Warm up with a tiny dummy inference on mic-tap.

**Decode config**  -  identical in JNI and `validate_model.sh`:

```cpp
auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
wparams.greedy.best_of  = 1;      // NOT beam_search.beam_size (ignored under GREEDY)
wparams.temperature     = 0.0f;
wparams.temperature_inc = 0.0f;   // disables the fallback ladder
wparams.no_context      = true;
wparams.single_segment  = true;
wparams.no_timestamps   = true;
wparams.suppress_blank  = true;
wparams.suppress_nst    = true;
wparams.max_tokens      = 96;
wparams.n_threads       = 4;      // tune per SoC
wparams.language        = "en";   // from the catalog entry
wparams.vad             = true;
wparams.vad_model_path  = vadPath;
```

**VAD.** Bundle the 885 kB blob in `assets/`; extract once to `noBackupFilesDir` (`.partial` → verify SHA-256 → rename; re-extract when the catalog hash changes; not `cacheDir`, which the OS can evict mid-session). For endpointing during capture use the standalone streaming API  -  `whisper_vad_detect_speech_no_reset()` plus **`whisper_vad_reset_state()` between utterances**. Feed fixed 512-sample frames at 16 kHz (verify against your pinned commit). Knobs: `min_silence_duration_ms` ≈ 700–900 ms endpoint, `max_speech_duration_s` hard cap, `speech_pad_ms` protects the first phoneme, `min_speech_duration_ms` rejects taps/coughs. Gate silence on `whisper_full_get_segment_no_speech_prob()`.

**Latency.** Whisper pads every input to a 30 s mel window, so short dictation costs nearly as much as long. `audio_ctx` scaling is a possible encoder speedup  -  measure the WER cost before shipping.

## 8. Quality gates

| Gate              | When                                | Criteria                                                                 |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------ |
| 4-clip regression | every build                         | transcripts present, quant WER ≤ f16 + 5 pts, silence empty, VAD enabled |
| 200-clip WER      | nightly + dispatch                  | quant ≤ f16 + 1.0 absolute; per-language table archived                  |
| Device bench      | before publish                      | `bench/<id>.json` present; fills the model card                          |
| Reproducibility   | on `whisper_cpp_ref` bump / monthly | build twice, diff SHA-256                                                |

Eval sets: LibriSpeech dev-clean, Common Voice / FLEURS per language, a **short-utterance set** ("yes", "ok", "call mom"), and silence/noise. `whisper-small` multilingual is weak on several Indic languages  -  publish per-language WER and pick defaults per language. `distil-*` is English-only; `large-v3-turbo` (~809M params) is not viable on 4 GB devices.

## 9. Local build (mirrors CI)

```bash
git clone https://github.com/ggml-org/whisper.cpp && git -C whisper.cpp checkout "$WCPP_REF"
git clone https://github.com/openai/whisper openai-whisper && git -C openai-whisper checkout "$OPENAI_REF"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-small.en', revision='$MODEL_REVISION', local_dir='./hf_model')"
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DGGML_NATIVE=OFF
cmake --build whisper.cpp/build -j
./scripts/convert_and_quantize.sh ./hf_model ./openai-whisper ./whisper.cpp \
  ./converted ./release/leantype-whisper-small.en-q5_1.bin q5_1
./scripts/validate_model.sh ./whisper.cpp/build/bin/whisper-cli \
  ./release/leantype-whisper-small.en-q5_1.bin ./samples/en en \
  ./converted/ggml-model.bin ./ggml-silero-v5.1.2.bin 4
```

## 10. Release checklist

**Provenance & licensing**  -  base repo + revision recorded · `whisper.cpp` commit pinned · license verified fail-closed (override with attribution if needed) · `THIRD_PARTY_NOTICES.md` + `MODEL_CARD.md` + weights license in assets · in-app attribution updated.

**Build & quality**  -  Release build, `GGML_NATIVE=OFF` · quant rationale documented · checksums verified against the published asset · 4-clip gate green incl. silence · nightly WER within tolerance.

**Android**  -  download + hash verify end-to-end · SAF import tested · `:voice` isolation and idle release verified · tested on a 4 GB and a mid-range ARM64 device · `bench/<id>.json` committed.

## 11. First-run runbook

1. Move WAVs into `samples/en/`; add `samples/README.md` with per-clip provenance/license (LibriSpeech needs CC-BY attribution).
2. Set repo vars `VAD_REVISION` (40-hex) and `VAD_SHA256` (64-hex).
3. Commit all files above; dispatch with **`publish_release: false`**.
4. Lock in from the artifact: `SIZES.txt` → size table · `pip-freeze.txt` → `requirements.lock.txt` (prepend `--extra-index-url`) · `catalog-entry.json` + `vad-entry.json` → `catalog.json` · derived `whisper_cpp_version`.
5. On device, read `/proc/<pid>/smaps_rollup` in `:voice` after load → set the idle-unload constant → write `bench/<id>.json`.
6. Pin `actions/*@v4` and `softprops/action-gh-release@v2` to commit SHAs; dispatch with `publish_release: true`.

**Verify on run #1:** `whisper-cli --help` flag names on your pinned commit · `transformers==4.46.3` loads the checkpoint · the `quantize` binary name · whether the CLI exits non-zero when VAD finds no speech (if so, special-case the `<silence>` row rather than treating it as inference failure).

## 12. Size reference (approximate  -  replace with `SIZES.txt`)

| Model            | f16      | q5_1     | q8_0     |
| ---------------- | -------- | -------- | -------- |
| tiny.en          | ~75 MiB  | ~31 MiB  | ~42 MiB  |
| base.en          | ~142 MiB | ~57 MiB  | ~78 MiB  |
| small.en / small | ~466 MiB | ~181 MiB | ~252 MiB |

`.en` and multilingual variants of a size class are architecturally identical  -  near-identical file sizes. q5_1 is ~9% larger than q5_0, not ~5%. Don't claim a quality ordering until measured.

## 13. Deferred (v1.1+)

- **Quant throughput benchmark on real ARM**  -  ggml has repacked GEMM paths for some types (notably q4_0/q8_0) on ARMv8.2+ dotprod/i8mm; q5_1 may use slower generic kernels. Choose the default on size **and** WER **and** tokens/s.
- **Parakeet A/B** via `include/parakeet.h` in the same runtime  -  no second engine, no ONNX Runtime.
- **Indic fine-tune evaluation** with `samples/<lang>/` sets.
- `pip --require-hashes` and per-build double-build diffing.
- Build provenance attestation (`actions/attest-build-provenance`).
