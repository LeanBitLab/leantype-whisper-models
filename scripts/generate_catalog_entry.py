#!/usr/bin/env python3
"""Generates catalog-entry.json, vad-entry.json, and MODEL_CARD.md for release."""

import hashlib
import json
import os
import pathlib
import textwrap

prefix = os.environ["OUTPUT_PREFIX"]
quant = os.environ["QUANT_TYPE"]
ver = os.environ["MODEL_VERSION"]
repo = os.environ["GITHUB_REPOSITORY"]
mid = f"{prefix}-{quant}"
name = f"{prefix}-{quant}.bin"

data = pathlib.Path(name).read_bytes()
sha = hashlib.sha256(data).hexdigest()
lic = pathlib.Path("../meta/license.txt").read_text().strip()
attr = pathlib.Path("../meta/attribution.txt").read_text().strip()
wcv = os.environ["WCPP_VER"]
wcc = os.environ["WCPP_COMMIT"]

entry = {
    "id": mid,
    "version": int(ver),
    "file": name,
    "url": f"https://github.com/{repo}/releases/download/{mid}-v{ver}/{name}",
    "sha256": sha,
    "size_bytes": len(data),
    "base_model": os.environ["MODEL_NAME"],
    "model_revision": os.environ["MODEL_REVISION"],
    "quant_type": quant,
    "languages": os.environ["MODEL_LANGS"].split(","),
    "license": lic,
    "attribution": attr,
    "whisper_cpp_commit": wcc,
    "min_runtime_version": wcv,
}
pathlib.Path("catalog-entry.json").write_text(json.dumps(entry, indent=2) + "\n")

vad_bytes = pathlib.Path("../ggml-silero-v5.1.2.bin").read_bytes()
pathlib.Path("vad-entry.json").write_text(
    json.dumps(
        {
            "file": "ggml-silero-v5.1.2.bin",
            "revision": os.environ["VAD_REVISION"],
            "sha256": hashlib.sha256(vad_bytes).hexdigest(),
            "size_bytes": len(vad_bytes),
            "bundled": True,
        },
        indent=2,
    )
    + "\n"
)

bpath = pathlib.Path(f"../bench/{mid}.json")
b = json.loads(bpath.read_text()) if bpath.exists() else {}
get_bench = lambda k: b.get(k, "PENDING (device bench)")

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
- **Intended Use:** Offline voice typing in LeanType (`:voice` isolated process / voice plugin).

## Attribution
{attr or "Upstream OpenAI Whisper weights (MIT); see THIRD_PARTY_NOTICES.md."}

## Device Benchmarks
- **Peak Private_Dirty RAM:** {get_bench('peak_private_dirty_kb')} kB
- **Cold Load Time:** {get_bench('cold_load_ms')} ms
- **RTF:** {get_bench('rtf')}
- **Device / SoC / threads:** {get_bench('device')} / {get_bench('soc')} / {get_bench('threads')}
""")
pathlib.Path("MODEL_CARD.md").write_text(card)
print(f"Generated catalog-entry.json, vad-entry.json, and MODEL_CARD.md for {mid}")
