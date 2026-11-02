#!/usr/bin/env python3
"""Offline wake-word test audio (voice host, casa-tts venv): 16 kHz mono WAVs, nothing is played.
pos_<word>__<voice>__<speed>__<noise>__<n>.wav = the wake phrase (+ a short command), neg_*.wav = everyday talk with
near-miss words and no wake word (~10 min). Used by eval_wake.py (pymicro-wakeword) on the Mac."""
import os, sys
import numpy as np
from kokoro_onnx import Kokoro
import wave


def wav_write(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())

OUT = sys.argv[1] if len(sys.argv) > 1 else "/opt/voicelab/wake"
SR = 24000
VOICES = [("us_m", "am_michael", "en-us"), ("us_f", "af_sarah", "en-us"), ("us_f2", "af_heart", "en-us"), ("us_m2", "am_adam", "en-us"),
          ("uk_m", "bm_george", "en-gb"), ("uk_f", "bf_emma", "en-gb"), ("es_m", "em_alex", "en-us"), ("es_f", "ef_dora", "en-us"),
          ("it_f", "if_sara", "en-us"), ("in_m", "hm_omega", "en-us")]
PHRASES = {"hey_coco": ["Hey Coco", "Hey Coco, turn on the lights", "Hey Coco, what's the weather"],
           "okay_nabu": ["Okay Nabu", "Okay Nabu, turn on the lights", "Okay Nabu, what's the weather"],
           "alexa": ["Alexa", "Alexa, turn on the lights", "Alexa, what's the weather"]}
# (2026-10-23) "Alexa" sound-alikes for the Alexa-vs-Okay-Nabu comparison -> negalexa_<voice>.wav
NEG_ALEXA = ["Alex, can you pass me the salt please?", "Alexander the Great conquered half of the known world.",
             "My cousin Alexis is coming over for dinner on Friday.", "I think my next car will be a Lexus.",
             "The election results come out tonight after nine.", "The electric bill went up again this month.",
             "She wrote a paper about dyslexia in young kids.", "That book is a lexicon of old sailing words.",
             "We're flying to Alexandria in the spring.", "Alessandra said the restaurant was fully booked.",
             "Alejandra called this morning about the apartment.", "A lexical choice like that changes the meaning.",
             "Hey Alex, are you watching the game later?", "He's at the Lexington branch of the bank.",
             "Relax, you're going to be fine, I promise.", "Select the second option and press next.",
             "The electricity went out during the storm.", "A lexus or an Acura, I can't decide."]
NEG = ["I made some hot cocoa for the kids this morning.", "Can you pass me the coconut milk from the fridge?",
       "Hey Rocco, are you coming to the game tonight?", "Go go go, we're going to be late!", "Hey, look at that, it's snowing again.",
       "Coco Chanel was a famous designer from France.", "She said okay, but nobody knew what she meant.", "Hey, could you open the window?",
       "The coffee shop on the corner has a new cold brew.", "We watched Coco again, it's such a good movie.",
       "Honestly I think we should call the plumber tomorrow.", "Hey you, how was your day at work?", "Nobody knew the answer, not even the teacher.",
       "Okay, now let's talk about the budget for next month.", "The cocoa beans come from Ghana and Ivory Coast.", "Hey, hey, calm down, it's fine.",
       "My friend Nico is visiting from Barcelona next week.", "The toucan at the zoo was eating papaya.", "Turn left at the light and then keep going.",
       "I'm going to the gym, see you later.", "Kokomo is that old Beach Boys song, right?", "He speaks Spanish, Basque and a bit of French.",
       "Hey mom, can you call me back when you get this?", "The weather channel says it will rain on Saturday.", "Oh no, I forgot the keys again.",
       "Do you want tacos or pasta for dinner tonight?", "Hey, cool, that looks amazing!", "Echo, echo, echo, the cave repeats everything.",
       "We need more coffee, cocoa, and cookies.", "Okay so the meeting moved to three o'clock."]


def tts(k, text, voice, lang, speed):
    a, sr = k.create(text, voice=voice, speed=speed, lang=lang)
    return a.astype(np.float32)


def norm(x, dbfs=-20.0):
    return x * (10 ** (dbfs / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9))


def to16k(x):
    idx = np.arange(0, len(x), SR / 16000.0)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def main():
    # optional 2nd argument: comma-separated wake words to (re)generate, e.g. "alexa" (default: all)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PHRASES)
    os.makedirs(OUT, exist_ok=True)
    k = Kokoro("/opt/casa-tts/models/kokoro-v1.0.onnx", "/opt/casa-tts/models/voices-v1.0.bin")
    rng = np.random.default_rng(3)
    babble = norm(np.concatenate([tts(k, t, v, "en-us", 1.0) for t, v in zip(NEG[:8], ["af_nicole", "am_eric", "bf_lily", "am_puck"] * 2)]))

    def noise(kind, n):
        if kind == "fan":
            b = np.cumsum(rng.normal(0, 1, n)); b -= np.convolve(b, np.ones(2400) / 2400, mode="same"); return norm(b.astype(np.float32), -32)
        if kind == "keyboard":
            x = np.zeros(n, np.float32); t = 0
            while t < n - 800:
                t += int(rng.uniform(0.08, 0.3) * SR)
                if t < n - 800:
                    x[t:t + 400] += rng.normal(0, 1, 400) * np.exp(-np.arange(400) / 40)
            return norm(x, -32)
        if kind == "tv":
            s = rng.integers(0, len(babble)); return norm(np.tile(babble, n // len(babble) + 2)[s:s + n], -30)
        return np.zeros(n, np.float32)

    count = 0
    for word, phrases in PHRASES.items():
        if word not in only:
            continue
        for vid, voice, lang in VOICES:
            for speed in (0.9, 1.0, 1.15):
                for i, p in enumerate(phrases):
                    sp = norm(np.concatenate([np.zeros(int(SR * .5), np.float32), tts(k, p, voice, lang, speed), np.zeros(int(SR * .6), np.float32)]))
                    for nk in ("none", "fan", "tv", "keyboard"):
                        x = sp + noise(nk, len(sp))
                        wav_write(os.path.join(OUT, f"pos_{word}__{vid}__{speed}__{nk}__{i}.wav"), to16k(x))
                        count += 1
        print(word, "done", count, flush=True)
    if "alexa" in only:
        for vid, voice, lang in VOICES:
            seg = []
            for t in NEG_ALEXA:
                seg.append(tts(k, t, voice, lang, 1.0)); seg.append(np.zeros(int(SR * rng.uniform(.3, .9)), np.float32))
            wav_write(os.path.join(OUT, f"negalexa_{vid}.wav"), to16k(norm(np.concatenate(seg))))
    if len(sys.argv) > 2:
        print("positives", count); return
    for j, (vid, voice, lang) in enumerate(VOICES):
        seg = []
        for t in NEG:
            seg.append(tts(k, t, voice, lang, 1.0)); seg.append(np.zeros(int(SR * rng.uniform(.3, .9)), np.float32))
        x = norm(np.concatenate(seg))
        wav_write(os.path.join(OUT, f"neg_{vid}.wav"), to16k(x))
    print("positives", count, "negatives", len(VOICES))


if __name__ == "__main__":
    main()
