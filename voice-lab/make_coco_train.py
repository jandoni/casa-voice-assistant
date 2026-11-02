#!/usr/bin/env python3
"""Training data for a custom "Hey Coco" microWakeWord model (voice host, casa-tts venv). 16 kHz mono WAVs.
negative/ = confusable phrases (hard negatives) -> trainer negative_samples/;  positive/ = accent-matched "Hey Coco"
(Spanish/Italian/French/Portuguese/Hindi/UK/US Kokoro voices) -> trainer personal_samples/.
Voices here are DIFFERENT from the evaluation set in make_wake_clips.py, so the offline test stays honest.
"hey cocoa" is deliberately NOT a negative: it sounds identical to "hey coco"."""
import os, sys, wave
import numpy as np
from kokoro_onnx import Kokoro

OUT = sys.argv[1] if len(sys.argv) > 1 else "/opt/voicelab/cocotrain"
SR = 24000
TRAIN_VOICES = [("af_alloy", "en-us"), ("af_aoede", "en-us"), ("af_bella", "en-us"), ("af_jessica", "en-us"), ("af_kore", "en-us"),
                ("af_nicole", "en-us"), ("af_nova", "en-us"), ("af_river", "en-us"), ("af_sky", "en-us"), ("am_echo", "en-us"),
                ("am_eric", "en-us"), ("am_fenrir", "en-us"), ("am_liam", "en-us"), ("am_onyx", "en-us"), ("am_puck", "en-us"),
                ("bf_alice", "en-gb"), ("bf_isabella", "en-gb"), ("bf_lily", "en-gb"), ("bm_daniel", "en-gb"), ("bm_fable", "en-gb"),
                ("bm_lewis", "en-gb"), ("em_santa", "en-us"), ("ff_siwis", "en-us"), ("hf_alpha", "en-us"), ("hf_beta", "en-us"),
                ("hm_psi", "en-us"), ("im_nicola", "en-us"), ("pf_dora", "en-us"), ("pm_alex", "en-us")]
SPANISHISH = [("em_santa", "es"), ("em_alex", "es"), ("ef_dora", "es"), ("pm_alex", "pt-br"), ("pf_dora", "pt-br"), ("im_nicola", "it"), ("if_sara", "it")]
CONFUSABLES = ["coco", "cocoa", "hot cocoa", "coconut", "coco chanel", "the movie coco", "kokomo", "go go", "go go go", "echo", "echo echo",
               "hey cool", "hey mom", "hey joe", "hey josh", "hey rocco", "hey you", "hey there", "hey look", "hey go home", "hey coach",
               "hey google", "hey siri", "alexa", "okay nabu", "okay", "okay so", "coffee", "coffee shop", "toucan", "cuckoo", "taco",
               "poco", "loco", "cocoa puffs", "morocco", "yoko", "koko the gorilla", "no no no", "so so", "hello",
               "hey hey", "cocktail", "cool cool cool", "hey nico", "hey mocha"]
FORMS = ["{p}", "so {p}, you know", "I think {p} is right"]
POS_TEXTS = ["Hey Coco", "Hey, Coco", "Hey Coco!", "Hey Coco?", "Hey... Coco"]


def wav_write(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def to16k(x):
    return np.interp(np.arange(0, len(x), SR / 16000.0), np.arange(len(x)), x).astype(np.float32)


def norm(x, dbfs):
    return x * (10 ** (dbfs / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9))


def main():
    # (2026-09-27) one phrase per process: kokoro-onnx grew to 8.5 GB over thousands of clips and made the mini PC swap
    # (HA + voice slowed down). Usage: make_coco_train.py OUT <first> <last> [pos]  - the wrapper loops over phrases.
    import onnxruntime as ort  # noqa
    k = Kokoro("/opt/casa-tts/models/kokoro-v1.0.onnx", "/opt/casa-tts/models/voices-v1.0.bin")
    rng = np.random.default_rng(11 + int(sys.argv[2]) if len(sys.argv) > 2 else 11)
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    last = int(sys.argv[3]) if len(sys.argv) > 3 else len(CONFUSABLES) - 1
    do_pos = len(sys.argv) > 4 and sys.argv[4] == "pos"
    os.makedirs(f"{OUT}/negative", exist_ok=True); os.makedirs(f"{OUT}/positive", exist_ok=True)
    n = 0
    for pi, p in enumerate(CONFUSABLES):
        if not (first <= pi <= last):
            continue
        for voice, lang in TRAIN_VOICES:
            for fi, form in enumerate(FORMS):
                speed = float(rng.choice([0.85, 1.0, 1.15]))
                try:
                    a, _ = k.create(form.format(p=p), voice=voice, speed=speed, lang=lang)
                except Exception:
                    continue
                x = norm(np.concatenate([np.zeros(4000, np.float32), a.astype(np.float32), np.zeros(4000, np.float32)]), rng.uniform(-30, -16))
                wav_write(f"{OUT}/negative/c{pi:02d}_{fi}_{voice}.wav", to16k(x)); n += 1
        print(f"confusable {pi + 1}/{len(CONFUSABLES)} {p!r} -> {n} clips", flush=True)
    m = 0
    if not do_pos:
        print("negatives", n); return
    for voice, lang in TRAIN_VOICES + SPANISHISH:
        for t in POS_TEXTS:
            for speed in (0.85, 1.0, 1.15):
                try:
                    a, _ = k.create(t, voice=voice, speed=speed, lang=lang)
                except Exception:
                    continue
                x = norm(np.concatenate([np.zeros(3200, np.float32), a.astype(np.float32), np.zeros(3200, np.float32)]), rng.uniform(-28, -16))
                wav_write(f"{OUT}/positive/p_{voice}_{lang}_{abs(hash(t)) % 1000}_{speed}.wav", to16k(x)); m += 1
    print("negatives", n, "positives", m)


if __name__ == "__main__":
    main()
