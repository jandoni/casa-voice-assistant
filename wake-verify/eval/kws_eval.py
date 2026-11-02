#!/usr/bin/env python3
"""kws_eval.py - would an ASR keyword spotter (sherpa-onnx KWS zipformer, gigaspeech 3.3M) catch "Alexa" false wakes?
Runs the spotter over positive clips (someone says "Alexa") and negative audio (room audio, TV false wakes, sound-alikes,
other wake phrases, real commands recorded by the Echos) for a grid of (boost score, threshold). Prints recall and
false alarms per set + ms per 2 s clip."""
import glob, json, os, sys, time, wave
from multiprocessing import Pool
import numpy as np
import sherpa_onnx

M = "/srv/wakeword/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KW = os.environ.get("KW", "▁A LE X A").split("|")
CFG = None
SPOT = None


def load(path):
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        b = w.readframes(n)
    x = np.frombuffer(b, np.int16 if sw == 2 else np.int32).astype(np.float32)
    x /= 32768.0 if sw == 2 else 2147483648.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        t = np.arange(0, len(x) / sr, 1 / 16000)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return prep(x)


def prep(x):
    """raw Echo mic clips: DC + rumble below ~90 Hz, quiet -> high-pass, then peak-normalise to -6 dBFS (max +30 dB)"""
    if os.environ.get("PREP", "1") == "0" or len(x) < 1600:
        return x
    X = np.fft.rfft(x); X[: int(90 * len(x) / 16000)] = 0
    y = np.fft.irfft(X, len(x)).astype(np.float32)
    pk = float(np.abs(y).max()) + 1e-9
    return (y * min(0.5 / pk, 31.6)).astype(np.float32)


def init(cfg):
    global CFG, SPOT
    CFG = cfg
    kwf = f"/tmp/kw_{os.getpid()}.txt"
    open(kwf, "w").write("".join(f"{k} :{cfg[0]} #{cfg[1]} @ALEXA\n" for k in KW))
    SPOT = sherpa_onnx.KeywordSpotter(
        tokens=f"{M}/tokens.txt", encoder=f"{M}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        decoder=f"{M}/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        joiner=f"{M}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        num_threads=1, max_active_paths=4, keywords_file=kwf, keywords_score=cfg[0], keywords_threshold=cfg[1],
        num_trailing_blanks=1, provider="cpu")


def spot(x):
    """number of detections + their end times (s)"""
    s = SPOT.create_stream()
    s.accept_waveform(16000, x)
    s.accept_waveform(16000, np.zeros(8000, np.float32))   # 0.5 s tail: the word may end the clip
    s.input_finished()
    hits = []
    while SPOT.is_ready(s):
        SPOT.decode_stream(s)
        r = SPOT.get_result(s)
        if r:
            hits.append(round(s.frame_offset * 0.01 if hasattr(s, "frame_offset") else -1, 2))
            SPOT.reset_stream(s)
    return hits


def job(item):
    name, path = item
    try:
        x = load(path)
    except Exception as e:  # noqa: BLE001
        return name, path, None, 0, 0.0
    t = time.perf_counter()
    h = spot(x)
    return name, path, h, len(x) / 16000, (time.perf_counter() - t) * 1000


def sets():
    S = {}
    S["pos_tts_alexa"] = sorted(glob.glob("/opt/voicelab/wake/pos_alexa__*.wav"))
    real = [l.split()[0] for l in open(sys.argv[1])] if len(sys.argv) > 1 else []
    S["pos_real_alexa"] = real
    allw = sorted(glob.glob("/opt/casa-stt/wakeclips/*.wav"))
    # the night of TV false wakes that motivated the word check (verifier clips are named <utc>_<room>_<score>_...)
    S["neg_tv_false_wakes_tonight"] = [p for p in allw if os.environ.get("TV_NIGHT", "20261102_0") in p
                                       and "_" + os.environ.get("TV_ROOM", "living") + "_" in p]
    S["neg_older_wake_clips"] = [p for p in allw if os.path.basename(p) < "20261101"]
    S["neg_roomcap"] = sorted(glob.glob("/srv/wakeword/wwlab/roomcap/*.wav"))
    S["neg_soundalikes"] = sorted(glob.glob("/opt/voicelab/wake/negalexa_*.wav"))
    S["neg_sentences"] = sorted(glob.glob("/opt/voicelab/wake/neg_*.wav")) + sorted(glob.glob("/opt/voicelab/negsent/*.wav"))
    S["neg_other_wakewords"] = sorted(glob.glob("/opt/voicelab/wake/pos_hey_coco__*.wav") + glob.glob("/opt/voicelab/wake/pos_okay_nabu__*.wav"))
    S["neg_echo_commands"] = sorted(glob.glob("/opt/casa-stt/recordings/*.wav"))
    return S


if __name__ == "__main__":
    grid = [(float(a), float(b)) for a, b in (g.split("/") for g in os.environ.get("GRID", "1.0/0.25").split(","))]
    S = sets()
    items = [(k, p) for k, v in S.items() for p in v]
    print({k: len(v) for k, v in S.items()}, flush=True)
    for cfg in grid:
        with Pool(10, initializer=init, initargs=(cfg,)) as pool:
            res = pool.map(job, items, chunksize=8)
        out = {}
        ms2 = []
        for name, path, h, dur, ms in res:
            if h is None:
                continue
            o = out.setdefault(name, {"n": 0, "hit": 0, "sec": 0.0, "hits": 0, "miss": [], "fa": []})
            o["n"] += 1; o["sec"] += dur; o["hits"] += len(h)
            if h:
                o["hit"] += 1
                if name.startswith("neg"):
                    o["fa"].append(os.path.basename(path))
            elif name.startswith("pos"):
                o["miss"].append(os.path.basename(path))
            if 1.5 < dur < 2.5:
                ms2.append(ms)
        print(f"\n== boost {cfg[0]} threshold {cfg[1]}  (~{np.median(ms2) if ms2 else 0:.0f} ms per 2 s clip)")
        for k, o in out.items():
            if k.startswith("pos"):
                print(f"  {k:28s} recall {o['hit']}/{o['n']}  missed: {o['miss'][:6]}")
                if k == "pos_tts_alexa":
                    from collections import Counter
                    print("     misses by voice:", dict(Counter(m.split('__')[1] for m in o['miss'])), " by noise:", dict(Counter(m.split('__')[3] for m in o['miss'])), " by phrase:", dict(Counter(m.split('__')[4][0] for m in o['miss'])))
            else:
                print(f"  {k:28s} clips with a hit {o['hit']}/{o['n']}  detections {o['hits']} in {o['sec']/3600:.2f} h  e.g. {o['fa'][:4]}")
        sys.stdout.flush()
