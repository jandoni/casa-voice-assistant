#!/usr/bin/env python3
"""Offline wake-word evaluation (silent): stream WAVs through pymicro-wakeword like the Echo does and report, per model,
the hit rate on pos_<word>__* clips and false wakes on neg_*.wav at several thresholds.
  venv/bin/python eval_wake.py <wav dir> <model.json> [<model.json> …]"""
import collections, glob, json, os, sys, wave
import numpy as np
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def scores(cfg, path):
    """-> list of (seconds, sliding-window mean probability) for the whole file"""
    mww = MicroWakeWord.from_config(cfg)
    feats = MicroWakeWordFeatures()
    with wave.open(path) as w:
        audio = w.readframes(w.getnframes())
    out, t = [], 0.0
    win = collections.deque(maxlen=mww.sliding_window_size)
    for i in range(0, len(audio), 320):                     # 10 ms chunks
        for f in feats.process_streaming(audio[i:i + 320]):
            mww.process_streaming(f)
            probs = list(mww._probabilities)
            if probs:
                out.append((t, float(np.mean(probs))))
        t += 0.01
    mww.close()
    return out


def main():
    d, cfgs = sys.argv[1], sys.argv[2:]
    for cfg in cfgs:
        word = json.load(open(cfg))["model"].replace(".tflite", "")
        pos = sorted(glob.glob(os.path.join(d, f"pos_{word}__*.wav")))
        peaks = {p: max((s for _, s in scores(cfg, p)), default=0.0) for p in pos}
        print(f"\n== {word}  ({len(pos)} clips)  model cutoff {json.load(open(cfg))['micro']['probability_cutoff']}")
        for th in THRESHOLDS:
            hit = sum(1 for v in peaks.values() if v >= th)
            by = collections.defaultdict(lambda: [0, 0])
            for p, v in peaks.items():
                noise = os.path.basename(p).split("__")[3]
                by[noise][0] += v >= th; by[noise][1] += 1
            print(f"  th {th:.1f}: hits {hit}/{len(pos)} ({100 * hit / max(1, len(pos)):.0f}%)  " +
                  "  ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by.items())))
        # false wakes on everyday talk (2 s cool-down between detections, like VACA)
        negs = sorted(glob.glob(os.path.join(d, "neg_*.wav")))
        total_s, fa = 0.0, collections.Counter()
        for n in negs:
            sc = scores(cfg, n); total_s += sc[-1][0] if sc else 0
            for th in THRESHOLDS:
                last = -9
                for t, s in sc:
                    if s >= th and t - last > 2:
                        fa[th] += 1; last = t
        print(f"  false wakes in {total_s / 60:.1f} min of everyday talk: " + "  ".join(f"th {th}: {fa[th]}" for th in THRESHOLDS))


if __name__ == "__main__":
    main()
