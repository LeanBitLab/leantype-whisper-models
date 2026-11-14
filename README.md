# LeanType Whisper Models

Automated model conversion, quantization, validation, and release pipeline for offline Whisper speech-to-text models used by the [LeanType Android IME](https://github.com/LeanBitLab/HeliboardL) and [LeanType Voice Plugin](https://github.com/LeanBitLab/LeanType-Voice-Plugin).

## Overview

Upstream `ggml-org` binaries distributed on Hugging Face are opaque binaries without auditable build pipelines. Building models directly via this repository provides:
- **Verifiable Provenance:** Pinned commits for `whisper.cpp`, `openai/whisper`, and Hugging Face revisions.
- **Fail-Closed Licensing:** Automated scanning of model card frontmatter to enforce redistributable open source licenses (rejecting non-commercial `-nc` licenses).
- **Quality Gates:** Automated 4-clip inference tests (evaluating Word Error Rate vs. f16 and verifying silence handling) on every build.
- **Device Gating:** Mandatory on-device benchmark metrics before releases can be published.
- **Security & Integrity:** Cryptographic SHA-256 checksums and exact byte sizes published in `catalog.json` for in-app integrity verification.

## Repository Layout

```
leantype-whisper-models/
├── .editorconfig
├── .gitattributes
├── .github/workflows/build-whisper-model.yml  # GitHub Actions build and release workflow
├── bench/<model-id>.json                     # On-device benchmark results (gates release)
├── catalog.json                              # Single source of truth consumed by LeanType
├── models.json                               # Per-model license overrides and attributions
├── docs/MODEL_CARD_TEMPLATE.md
├── samples/
│   ├── README.md                             # Per-clip provenance and licensing
│   └── en/
│       ├── sample-en-clean.wav
│       ├── sample-en-commands.wav
│       ├── sample-en-noisy.wav
│       ├── sample-short-silence.wav
│       └── expected-transcripts.txt          # Tab-separated reference transcripts
├── scripts/
│   ├── convert_and_quantize.sh               # Converts HF checkpoint to ggml f16 and quantizes
│   ├── validate_model.sh                     # Runs validation against sample audio and checks WER
│   └── benchmark_wer.py                      # Multi-clip WER evaluation script
├── requirements.txt                          # CPU PyTorch and Python build dependencies
├── LICENSE                                   # MIT (scripts and tooling)
├── THIRD_PARTY_NOTICES.md
├── RELEASE_CHECKLIST.md
└── guide.md                                  # Technical integration specification
```

## Quick Start (Local Build)

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

## Catalog Contract

`catalog.json` serves as the contract between this release repository and the LeanType IME / Voice Plugin. See [guide.md](guide.md) for full architecture and release details.

## License

Tooling and scripts are licensed under the [MIT License](LICENSE). Individual model weights retain their respective upstream licenses as specified in their model cards and `catalog.json`.
