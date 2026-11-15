# Audio Samples for Quality Gates

This directory contains the test clips used by `scripts/validate_model.sh` and `scripts/benchmark_wer.py` to evaluate speech-to-text accuracy and silence handling.

## Structure

```
samples/
├── README.md
└── en/
    ├── expected-transcripts.txt   # Tab-separated mapping: filename <TAB> transcript
    ├── sample-en-clean.wav        # 11 s of clear speech ("and so my fellow americans ask not ...")
    ├── sample-en-short.wav        # 3.2 s, the first words of the clip above, a short utterance as in voice typing
    ├── sample-en-noisy.wav        # the clean clip with white noise at about 10 dB signal to noise ratio
    └── sample-short-silence.wav   # 1.5 s of digital silence (must decode as an empty string with VAD enabled)
```

All clips are 16 kHz, mono, 16 bit.

## What the gate checks

- Every speech clip must contain audio (RMS of at least 100 on the 16 bit scale), otherwise the gate fails: a silent file cannot test a model.
- At least one speech clip must be scored, the word error rate over all speech clips must be at most `MAX_WER` (default 0.30), and, when an f16 reference model is given, at most 5 points above the f16 model.
- The silence clip must produce no text. This needs the VAD model, as Whisper alone tends to invent a word such as "you" on silence.

## Attribution & Provenance

The speech clips are made from the 11 second excerpt of John F. Kennedy's inaugural address (20 January 1961) that ships with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) as `samples/jfk.wav`. It is a recording of a US government work, which is not covered by copyright (public domain). They are created reproducibly (fixed noise seed) with:

```bash
python3 scripts/make_samples.py ./whisper.cpp/samples/jfk.wav ./samples/en
```

When adding sample audio from external corpora:
- LibriSpeech clips require CC-BY 4.0 attribution.
- Common Voice clips require CC0 attribution.
- Recorded clips must be documented with microphone model and environment notes.
