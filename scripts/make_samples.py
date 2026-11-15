#!/usr/bin/env python3
"""Create the speech clips in samples/en from the whisper.cpp JFK sample.

Usage:
    python3 scripts/make_samples.py ./whisper.cpp/samples/jfk.wav ./samples/en

The source is the 11 second excerpt of John F. Kennedy's inaugural address (20 January 1961) that ships with
whisper.cpp, a recording of a US government work, which is not covered by copyright (public domain). The output is deterministic:
  sample-en-clean.wav  the whole excerpt
  sample-en-short.wav  its first words ("and so my fellow americans"), a short utterance as in voice typing
  sample-en-noisy.wav  the whole excerpt with white noise at about 10 dB signal to noise ratio (fixed seed)
sample-short-silence.wav is not created here, it is digital silence.
"""
import array
import math
import random
import sys
import wave

SNR_DB = 10.0
SHORT_START_S, SHORT_END_S = 0.25, 3.45  # "and so my fellow americans" are spoken between 0.32 s and 3.29 s


def read(path):
    with wave.open(path) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), "need 16 kHz mono 16 bit"
        samples = array.array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    return samples


def write(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(array.array("h", samples).tobytes())


def rms(samples):
    return math.sqrt(sum(s * s for s in samples) / max(len(samples), 1))


def main():
    src, out = sys.argv[1], sys.argv[2]
    speech = read(src)
    write(f"{out}/sample-en-clean.wav", speech)
    write(f"{out}/sample-en-short.wav", speech[int(SHORT_START_S * 16000):int(SHORT_END_S * 16000)])
    sigma = rms(speech) / (10 ** (SNR_DB / 20))
    rng = random.Random(1961)
    noisy = [max(-32768, min(32767, int(round(s + rng.gauss(0, sigma))))) for s in speech]
    write(f"{out}/sample-en-noisy.wav", noisy)


if __name__ == "__main__":
    main()
