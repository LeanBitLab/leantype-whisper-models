# Device Benchmarks

This directory contains on-device benchmark records that gate public model releases.

Before a model can be published via the `build-whisper-model.yml` workflow, a corresponding benchmark file `bench/<output_prefix>-<quant_type>.json` must be committed to this directory.

## Benchmark Schema

```json
{
  "peak_private_dirty_kb": 224500,
  "cold_load_ms": 680,
  "rtf": 0.28,
  "device": "Pixel 7a",
  "soc": "Google Tensor G2",
  "threads": 4
}
```
