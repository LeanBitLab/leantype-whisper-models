#!/usr/bin/env python3
"""Benchmark Word Error Rate (WER) across audio samples.

Usage:
    python3 benchmark_wer.py --samples-dir ./samples/en --model ./release/model.bin --cli ./whisper-cli
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys


def norm(s: str) -> list[str]:
    return re.sub(r"[^\w\s']", "", s.lower()).split()


def compute_wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    return d[-1][-1], max(len(ref), 1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark WER for Whisper model")
    parser.add_argument("--samples-dir", required=True, help="Directory containing samples and expected-transcripts.txt")
    parser.add_argument("--model", required=True, help="Path to quantized model binary")
    parser.add_argument("--cli", required=True, help="Path to whisper-cli binary")
    parser.add_argument("--vad", default="", help="Path to Silero VAD binary")
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--threads", type=int, default=4, help="Inference threads")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.samples_dir) / "expected-transcripts.txt"
    if not exp_path.exists():
        print(f"FATAL: Missing {exp_path}", file=sys.stderr)
        sys.exit(1)

    total_errors = 0
    total_words = 0
    failures = []

    for line in exp_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        wav, _, expected = line.partition("\t")
        expected = expected.strip()
        wav_path = pathlib.Path(args.samples_dir) / wav

        cmd = [
            args.cli,
            "-m", args.model,
            "-f", str(wav_path),
            "-t", str(args.threads),
            "-l", args.lang,
            "--no-timestamps",
            "--no-prints",
            "--best-of", "1",
            "--temperature", "0.0",
            "-otxt",
            "-of", f"/tmp/{wav_path.stem}",
        ]
        if args.vad:
            cmd.extend(["--vad", "-vm", args.vad])

        res = subprocess.run(cmd, capture_output=True, text=True)
        out_txt = pathlib.Path(f"/tmp/{wav_path.stem}.txt")
        actual = out_txt.read_text().strip() if out_txt.exists() else ""

        if expected == "<silence>":
            if actual:
                failures.append(f"{wav}: Expected silence, got {actual!r}")
            print(f"[silence] {wav}: {'OK' if not actual else 'FAIL'}")
        else:
            e, n = compute_wer(norm(expected), norm(actual))
            total_errors += e
            total_words += n
            print(f"[eval] {wav}: ref={expected!r} hyp={actual!r} ({e}/{n})")

    wer = total_errors / (total_words or 1)
    print(f"Overall WER: {wer:.4f} ({total_errors}/{total_words})")
    if failures:
        print("\n".join(f"FAIL: {f}" for f in failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
