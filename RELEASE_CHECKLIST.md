# LeanType Whisper Model Release Checklist

Use this checklist before triggering a production release dispatch (`publish_release: true`).

## 1. Provenance & Licensing
- [ ] Upstream Hugging Face model repository and 40-hex commit SHA recorded.
- [ ] `whisper.cpp` commit SHA and `openai/whisper` commit SHA verified and pinned.
- [ ] Upstream model card YAML frontmatter license verified.
- [ ] If license is not on the auto-approve allowlist (MIT, Apache-2.0, BSD-3-Clause, CC0-1.0), verify `models.json` has an explicit attribution override approved by maintainers.
- [ ] Confirm no non-commercial (`-nc`) terms exist.
- [ ] Confirm `THIRD_PARTY_NOTICES.md` is bundled in the release.

## 2. Build & Quality Verification
- [ ] Workflow dispatched with `publish_release: false` for initial dry run.
- [ ] Built with `CMAKE_BUILD_TYPE=Release` and `GGML_NATIVE=OFF`.
- [ ] Quantization type rationale documented.
- [ ] 4-clip validation gate passed with silence producing zero tokens.
- [ ] Word Error Rate (WER) regression verified within 5 points of f16 baseline.
- [ ] `SIZES.txt` generated and logged.
- [ ] `pip freeze` dependencies locked into `requirements.lock.txt`.

## 3. On-Device Validation & Benchmarks
- [ ] Converted `.bin` tested on physical Android ARM64 device via LeanType Voice Plugin or test harness.
- [ ] `Private_Dirty` memory footprint measured via `/proc/<pid>/smaps_rollup`.
- [ ] Real-time factor (RTF) and cold load time recorded.
- [ ] Benchmark results committed to `bench/<output_prefix>-<quant_type>.json`.

## 4. Release & Catalog Synchronization
- [ ] Action commits pinned to SHAs in `.github/workflows/build-whisper-model.yml`.
- [ ] Workflow dispatched with `publish_release: true`.
- [ ] Generated `catalog-entry.json` merged into `catalog.json`.
- [ ] LeanType `VoiceModelRegistry.kt` or in-app catalog updated with new version, URL, and SHA-256.
