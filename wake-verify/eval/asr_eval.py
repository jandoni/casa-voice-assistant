#!/usr/bin/env python3
"""asr_eval.py <engine kroko|parakeet> <real_pos.txt> [sets,...] - second-stage idea: transcribe the 2 s wake clip and
accept only if an "Alexa"-like word is in it. Prints recall on positives, hits on negatives, transcripts of the key
clips and ms per 2 s clip."""
import os, re, sys, time
from difflib import SequenceMatcher
from multiprocessing import Pool
import numpy as np
import sherpa_onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws_eval import load, sets  # noqa: E402

KROKO = "/opt/casa-stt/models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
PARA = "/opt/casa-stt/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
ENG = sys.argv[1]
HOT_SCORE = float(os.environ.get("HOT", "2.0"))
REC = None
ALEXA_RE = re.compile(r"\b(alexa|alexia|alexis|alexus|a lexa|elexa|lexa|alex a|alex)\b")


def alexaish(t):
    t = t.lower()
    if ALEXA_RE.search(t):
        return True
    words = re.findall(r"[a-z]+", t)
    cands = words + [a + b for a, b in zip(words, words[1:])]
    return any(SequenceMatcher(None, w, "alexa").ratio() >= 0.75 for w in cands)


def init():
    global REC
    if ENG == "kroko":
        hot = f"/tmp/hot_{os.getpid()}.txt"
        open(hot, "w").write("ALEXA\nalexa\nAlexa\n")
        REC = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{KROKO}/tokens.txt", encoder=f"{KROKO}/encoder.onnx", decoder=f"{KROKO}/decoder.onnx",
            joiner=f"{KROKO}/joiner.onnx", num_threads=1, decoding_method="modified_beam_search", max_active_paths=4,
            **({"hotwords_file": hot, "hotwords_score": HOT_SCORE, "modeling_unit": "bpe",
                "bpe_vocab": "/opt/casa-stt/kroko.bpe.vocab"} if HOT_SCORE > 0 else {}))
    else:
        REC = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{PARA}/encoder.int8.onnx", decoder=f"{PARA}/decoder.int8.onnx", joiner=f"{PARA}/joiner.int8.onnx",
            tokens=f"{PARA}/tokens.txt", num_threads=int(os.environ.get("TH", "2")), model_type="nemo_transducer",
            decoding_method="greedy_search")


def text(x):
    if ENG == "kroko":
        s = REC.create_stream()
        s.accept_waveform(16000, np.zeros(int(float(os.environ.get("PRE", "0.5")) * 16000), np.float32))
        s.accept_waveform(16000, x)
        s.accept_waveform(16000, np.zeros(int(float(os.environ.get("POST", "1.2")) * 16000), np.float32))
        s.input_finished()
        while REC.is_ready(s):
            REC.decode_stream(s)
        return REC.get_result(s)
    s = REC.create_stream()
    s.accept_waveform(16000, np.concatenate([x, np.zeros(4800, np.float32)]))
    REC.decode_stream(s)
    return s.result.text


def job(item):
    name, path = item
    try:
        x = load(path)
    except Exception:  # noqa: BLE001
        return name, path, None, 0, 0
    if len(x) > 16000 * 6:      # long files: 2 s windows, hop 1 s (what the verifier would see)
        out, ms = [], 0.0
        for i in range(0, len(x) - 32000, 16000):
            t = time.perf_counter(); out.append(text(x[i:i + 32000])); ms += (time.perf_counter() - t) * 1000
        return name, path, " | ".join(out), len(x) / 16000, ms
    t = time.perf_counter()
    r = text(x)
    return name, path, r, len(x) / 16000, (time.perf_counter() - t) * 1000


if __name__ == "__main__":
    _a = sys.argv[:]; sys.argv = [_a[0], _a[2]]; S = sets(); sys.argv = _a
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else list(S)
    items = [(k, p) for k, v in S.items() if k in want for p in v]
    with Pool(int(os.environ.get("NP", "6")), initializer=init) as pool:
        res = pool.map(job, items, chunksize=4)
    out, ms2 = {}, []
    for name, path, t, dur, ms in res:
        if t is None:
            continue
        o = out.setdefault(name, {"n": 0, "hit": 0, "ex": []})
        o["n"] += 1
        hit = any(alexaish(seg) for seg in t.split(" | "))
        o["hit"] += hit
        if 1.5 < dur < 2.5:
            ms2.append(ms)
        if name in ("pos_real_alexa", "neg_tv_false_wakes_tonight") or (hit != name.startswith("pos") and len(o["ex"]) < 6):
            o["ex"].append(f"{'HIT ' if hit else 'miss'} {os.path.basename(path)[:44]}: {t[:90]!r}")
    print(f"engine {ENG} hot {HOT_SCORE}  ~{np.median(ms2) if ms2 else 0:.0f} ms per 2 s clip")
    for k, o in out.items():
        print(f"  {k:28s} alexa-like {o['hit']}/{o['n']}")
        for e in o["ex"]:
            print("      ", e)
