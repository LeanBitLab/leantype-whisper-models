#!/usr/bin/env python3
"""Downloads Hugging Face checkpoint files and enforces fail-closed licensing policy."""

import json
import os
import pathlib
import sys
import yaml
from huggingface_hub import snapshot_download

repo = os.environ["MODEL_NAME"]
rev = os.environ["MODEL_REVISION"]

print(f"Downloading checkpoint {repo}@{rev}...")
snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir="./hf_model",
    allow_patterns=[
        "README.md",
        "config.json",
        "generation_config.json",
        "vocab.json",
        "added_tokens.json",
        "tokenizer*.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ],
)

p = pathlib.Path("./hf_model/README.md")
text = p.read_text(encoding="utf-8") if p.exists() else ""
parts = text.split("---", 2)
meta = yaml.safe_load(parts[1]) if text.startswith("---") and len(parts) >= 3 else {}
meta = meta if isinstance(meta, dict) else {}
raw = meta.get("license") or "unknown"
lics = [raw.lower()] if isinstance(raw, str) else [str(x).lower() for x in raw]

ALLOW = {"mit", "apache-2.0", "bsd-3-clause", "cc0-1.0"}
attribution = ""

if len(lics) == 1 and lics[0] in ALLOW:
    lic = lics[0]
else:
    ov = (
        json.loads(pathlib.Path("models.json").read_text()).get("overrides", {})
        if pathlib.Path("models.json").exists()
        else {}
    )
    entry = ov.get(f"{repo}@{rev}")
    if not entry:
        print(
            f"FATAL: license {lics!r} requires an explicit models.json override for {repo}@{rev}",
            file=sys.stderr,
        )
        sys.exit(1)
    lic = str(entry["license"]).lower()
    attribution = entry.get("attribution", "").strip()
    if "-nc" in lic:
        print(
            f"FATAL: non-commercial license {lic!r} is never redistributable",
            file=sys.stderr,
        )
        sys.exit(1)
    if lic.startswith("cc-by") and not attribution:
        print("FATAL: CC-BY override requires attribution text", file=sys.stderr)
        sys.exit(1)
    print(
        f"license override applied by {entry.get('approved_by','?')}: {entry.get('rationale','')}"
    )

pathlib.Path("./meta/license.txt").write_text(lic + "\n")
pathlib.Path("./meta/attribution.txt").write_text(attribution + "\n")
print(f"Verified license: {lic}")
