# Audio Samples for Quality Gates

This directory contains test audio clips used by `scripts/validate_model.sh` and `scripts/benchmark_wer.py` to evaluate speech-to-text accuracy and silence handling.

## Structure

```
samples/
├── README.md
└── en/
    ├── expected-transcripts.txt   # Tab-separated mapping: filename <TAB> transcript
    ├── sample-en-clean.wav        # Clean conversational utterance ("hello world")
    ├── sample-en-commands.wav     # Voice typing commands ("new line send message")
    ├── sample-en-noisy.wav        # Speech with background ambient noise ("on my way")
    └── sample-short-silence.wav   # Ambient silence (must decode as empty string)
```

## Attribution & Provenance

When adding sample audio from external corpora:
- LibriSpeech clips require CC-BY 4.0 attribution.
- Common Voice clips require CC0 attribution.
- Recorded clips must be documented with microphone model and environment notes.
