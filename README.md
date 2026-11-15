# LeanType Whisper Models

[![Latest release](https://img.shields.io/github/v/release/LeanBitLab/leantype-whisper-models?style=flat-square&color=7C4DFF&labelColor=161b22)](https://github.com/LeanBitLab/leantype-whisper-models/releases/latest)
[![License: MIT](https://img.shields.io/badge/Tooling-MIT-7C4DFF?style=flat-square&labelColor=161b22)](LICENSE)
[![Models: Apache-2.0](https://img.shields.io/badge/Model%20weights-Apache--2.0-7C4DFF?style=flat-square&labelColor=161b22)](THIRD_PARTY_NOTICES.md)

Official **Whisper speech-to-text models** for offline voice typing in [LeanType](https://github.com/LeanBitLab/LeanType), built, quantized (Q5_1) and verified by an auditable pipeline. They are standard `whisper.cpp` GGML files, so they also work in any other app that runs GGML Whisper models.

This repository is the build and release pipeline, and the source of the model catalog that LeanType and the [LeanType Voice Plugin](https://github.com/LeanBitLab/LeanType-Voice-Plugin) read.

## Available models (v1.0.0)

All six models are quantized to **Q5_1** and use the Apache-2.0 license of the upstream OpenAI checkpoints. English-only models are a little more accurate for English; multilingual models support 99 languages with automatic language detection.

| Model | File | Languages | Size | RAM (peak) | Cold load | RTF | Download |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tiny** (English) | `leantype-whisper-tiny.en-q5_1.bin` | English | 30.7 MiB | ~128 MB | 310 ms | 0.16 | [Download](https://github.com/LeanBitLab/leantype-whisper-models/releases/download/v1.0.0/leantype-whisper-tiny.en-q5_1.bin) |
| **Tiny** (Multilingual) | `leantype-whisper-tiny-q5_1.bin` | Multilingual (99 languages) | 30.7 MiB | ~132 MB | 320 ms | 0.18 | [Download](https://github.com/LeanBitLab/leantype-whisper-models/releases/download/v1.0.0/leantype-whisper-tiny-q5_1.bin) |
| **Base** (English) | `leantype-whisper-base.en-q5_1.bin` | English | 57.0 MiB | ~215 MB | 520 ms | 0.25 | [Download](https://github.com/LeanBitLab/leantype-whisper-models/releases/download/v1.0.0/leantype-whisper-base.en-q5_1.bin) |
| **Base** (Multilingual) | `leantype-whisper-base-q5_1.bin` | Multilingual (99 languages) | 56.9 MiB | ~220 MB | 550 ms | 0.28 | [Download](https://github.com/LeanBitLab/leantype-whisper-models/releases/download/v1.0.0/leantype-whisper-base-q5_1.bin) |
| **Small** (English) | `leantype-whisper-small.en-q5_1.bin` | English | 181.3 MiB | ~435 MB | 1150 ms | 0.42 | [Download](https://github.com/LeanBitLab/leantype-whisper-models/releases/download/v1.0.0/leantype-whisper-small.en-q5_1.bin) |
| **Small** (Multilingual) | `leantype-whisper-small-q5_1.bin` | Multilingual (99 languages) | 181.3 MiB | ~440 MB | 1200 ms | 0.45 | [Download](https://github.com/LeanBitLab/leantype-whisper-models/releases/download/v1.0.0/leantype-whisper-small-q5_1.bin) |

RAM, cold load time and real-time factor (RTF, lower is faster) are measured on a **POCO M2 Pro** (Snapdragon 720G, 4 threads) and recorded in [`bench/`](bench/). They gate every release. Your device will differ; newer phones are faster.

Choosing a model:
- **Tiny**: lowest memory and fastest start, for entry-level phones and quick notes.
- **Base**: the recommended balance of accuracy and speed for daily dictation.
- **Small**: the highest accuracy for complex vocabulary and accents, with a larger download and more memory.

## Use in LeanType

1. Install the [LeanType Voice Plugin](https://github.com/LeanBitLab/LeanType-Voice-Plugin/releases/latest).
2. In LeanType open **Settings → Plugins → Voice Input → Whisper Models**.
3. **Standard flavor**: tap a model to download it. **Offline flavor**: download the `.bin` here in your browser, then tap **Import** on **Custom GGML Model**.

LeanType checks the size and SHA-256 of every download against [`catalog.json`](catalog.json).

## Verify a download

```bash
# download a model and SHA256SUMS.txt from the release into one folder, then:
sha256sum --ignore-missing -c SHA256SUMS.txt
```

| File | SHA-256 |
| :--- | :--- |
| `leantype-whisper-tiny.en-q5_1.bin` | `cc393fd5d2055d00bed6b3a67f1f3393c88e731ded80bdeea2912985179b1c9b` |
| `leantype-whisper-tiny-q5_1.bin` | `2fc2aa8e079b64dcab60dfd0d7bea8b9e18235ebc5137cb5ba0280adf6473b06` |
| `leantype-whisper-base.en-q5_1.bin` | `121e76ad8f95c9b7dcfe0295527423d2523beee7b50d9679782fc907a7f0b3de` |
| `leantype-whisper-base-q5_1.bin` | `36a6bf6281261fec331fcc65235e0f466d855ea4dae1bbbfb28ed4c0cf72d8c0` |
| `leantype-whisper-small.en-q5_1.bin` | `29800b2e0d75721e8a0dd04f9dc5a7054661cf866e8578e0d980128f62de3ebd` |
| `leantype-whisper-small-q5_1.bin` | `a91fdf80f97a66b22dabe567d2be9f195285ea6de2e9b2e73f80eb5a6aa627d0` |

## Why build the models here

Upstream binaries on Hugging Face are opaque files without an auditable build. Building them here gives:
- **Verifiable provenance:** pinned commits for `whisper.cpp`, `openai/whisper` and the Hugging Face model revisions, recorded per model in `catalog.json`.
- **Fail-closed licensing:** the upstream model card license is checked before the build, and non-commercial (`-nc`) licenses are always rejected.
- **Quality gates:** automated inference tests on every build with four clips (real speech clean, short and noisy, plus silence): the clips must contain audio, the word error rate must stay at or below 30% and within 5 points of the f16 model, and silence must decode to nothing.
- **Device gating:** a recorded on-device benchmark is required before a release can be published.
- **Integrity:** SHA-256 checksums and exact byte sizes are published in `catalog.json` and `SHA256SUMS.txt`.

Each catalog entry lists the base model and revision, quantization, languages, license, the `whisper.cpp` commit it was built with, and `min_runtime_version`. The files are GGML containers (not GGUF) with no format version field, so `min_runtime_version` is what tells a runtime whether it can load them.

## Catalog contract

[`catalog.json`](catalog.json) is the contract with LeanType and the Voice Plugin: it lists the models above plus the bundled Silero VAD model used to skip silence. See [guide.md](guide.md) for the full specification of the pipeline, the catalog and the quality gates.

## Repository layout

```
leantype-whisper-models/
├── .github/workflows/
│   ├── build-whisper-model.yml          # build, validate and release workflow
│   └── sync-upstream-whisper.yml        # scheduled watcher for new upstream whisper.cpp releases
├── bench/<model-id>.json                # on-device benchmark results (gate releases)
├── catalog.json                         # single source of truth consumed by LeanType
├── models.json                          # per-model license overrides and attributions
├── docs/MODEL_CARD_TEMPLATE.md
├── samples/                             # speech clips and expected transcripts for the quality gates
├── scripts/                             # convert, quantize, validate, benchmark and sample scripts
├── requirements.txt / requirements.lock.txt
├── SHA256SUMS.txt
├── THIRD_PARTY_NOTICES.md
├── RELEASE_CHECKLIST.md
└── guide.md                             # technical specification
```

## Build a model locally

```bash
# 1. Clone dependencies
git clone https://github.com/ggml-org/whisper.cpp
git clone https://github.com/openai/whisper openai-whisper

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download checkpoint (e.g. whisper-small.en)
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-small.en', local_dir='./hf_model')"

# 4. Build whisper.cpp quantize and cli tools
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DGGML_NATIVE=OFF
cmake --build whisper.cpp/build -j"$(nproc)"

# 5. Convert and quantize
./scripts/convert_and_quantize.sh ./hf_model ./openai-whisper ./whisper.cpp \
  ./converted ./release/leantype-whisper-small.en-q5_1.bin q5_1

# 6. Validate
./scripts/validate_model.sh ./whisper.cpp/build/bin/whisper-cli \
  ./release/leantype-whisper-small.en-q5_1.bin ./samples/en en \
  ./converted/ggml-model.bin ./ggml-silero-v5.1.2.bin 4
```

## Releasing or adding a model

Follow the [release checklist](RELEASE_CHECKLIST.md): record provenance and license, dry-run the workflow with `publish_release: false`, commit the device benchmark to `bench/`, then dispatch with `publish_release: true` and merge the generated catalog entry. Models whose license is not on the allowlist (MIT, Apache-2.0, BSD-3-Clause, CC0-1.0) need an explicit attribution override in `models.json`.

## License

Tooling and scripts are licensed under the [MIT License](LICENSE). Model weights keep the license of their upstream checkpoint, as listed in `catalog.json` and in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
