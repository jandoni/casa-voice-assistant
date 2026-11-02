#!/usr/bin/env python3
"""Voice lab clip generator (runs on the voice host with the casa-tts venv: /opt/casa-tts/venv/bin/python make_clips.py OUT).

Every test clip = "Okay Nabu" + 0.6 s + a phrase, spoken by one Kokoro voice (US / UK / Spanish / Italian / Indian
accents), mixed with a background (none / keyboard / fan / TV chatter) at ~10 dB SNR. A '|' in a phrase is a 1.6 s
pause (people pause mid-question). Also writes false_reel.mp3: minutes of TV-style talk full of near-miss words and NO
wake word, for counting false wakes. clips.json lists every clip with its reference text.
"""
import json, os, subprocess, sys
import numpy as np
from kokoro_onnx import Kokoro

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/voicelab"
SR = 24000
PHRASES = [
    ("monitor_off", "turn off the monitor light"),
    ("timer10", "set a timer for ten minutes"),
    ("kitchen_on", "turn on the kitchen lights"),
    ("jazz", "play some jazz in the living room"),
    ("dim30", "dim the bedroom lights to thirty percent"),
    ("weather", "what's the weather tomorrow"),
    ("capital", "what is the capital of Australia"),
    ("steak", "how long should I rest a steak after cooking it"),
    ("game", "who won the last football game"),
    ("bread", "so I was cooking bread | and I noticed it came out really hard, why could that be"),
    ("back", "my back hurts after a ten kilometer run | what could it be"),
    ("meat", "I overcooked the meat | would pouring water over it help"),
]
VOICES = [("us_m", "am_michael", "en-us"), ("us_f", "af_sarah", "en-us"), ("uk_m", "bm_george", "en-gb"),
          ("es_m", "em_alex", "en-us"), ("it_f", "if_sara", "en-us"), ("in_m", "hm_omega", "en-us")]
TV_TEXT = [
    "Okay, now let's take a look at the weather for the weekend, it's going to be a nice one.",
    "And I said, okay but what about the kitchen, can you believe they never finished it?",
    "Coming up after the break, the home team tries to bounce back after a tough loss on the road.",
    "Hey, can you pass me the remote? No, the other one, the one next to the lamp.",
    "Nobody knew what to do, so I called my brother and he said, okay, just turn it off and on again.",
    "It's a new day in the city and traffic is building on the freeway, so give yourself extra time.",
    "Okay everyone, settle down, the timer starts now and you have ten minutes to finish.",
    "The recipe calls for two cups of flour, a pinch of salt, and a lot of patience.",
    "Honestly I think the lights in the living room are too bright, we should get dimmer bulbs.",
    "Okay, nah, but seriously, what time does the game start tonight?",
]


def tts(k, text, voice, lang):
    parts = [p.strip() for p in text.split("|")]
    out = []
    for i, p in enumerate(parts):
        a, sr = k.create(p, voice=voice, speed=1.0, lang=lang)
        assert sr == SR
        out.append(a.astype(np.float32))
        if i < len(parts) - 1:
            out.append(np.zeros(int(SR * 1.6), np.float32))
    return np.concatenate(out)


def norm(x, dbfs=-18.0):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    return x * (10 ** (dbfs / 20) / rms)


def noise(kind, n, rng, babble=None):
    if kind == "keyboard":
        x = np.zeros(n, np.float32)
        t = 0
        while t < n - 800:
            t += int(rng.uniform(0.08, 0.3) * SR)
            if t >= n - 800:
                break
            click = rng.normal(0, 1, 400) * np.exp(-np.arange(400) / 40)
            x[t:t + 400] += click * rng.uniform(0.5, 1.0)
        return x
    if kind == "fan":
        w = rng.normal(0, 1, n)
        b = np.cumsum(w); b -= np.convolve(b, np.ones(2400) / 2400, mode="same")   # brown noise, drift removed
        return b.astype(np.float32)
    if kind == "tv":
        reps = int(np.ceil(n / len(babble)))
        start = rng.integers(0, len(babble))
        return np.tile(babble, reps + 1)[start:start + n]
    return np.zeros(n, np.float32)


def write_mp3(x, path):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-ar", "44100", "-b:a", "128k", path], input=pcm, check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    k = Kokoro("/opt/casa-tts/models/kokoro-v1.0.onnx", "/opt/casa-tts/models/voices-v1.0.bin")
    rng = np.random.default_rng(7)
    babble = norm(np.concatenate([tts(k, t, v, "en-us") for t, v in zip(TV_TEXT, ["af_nicole", "am_eric", "bf_lily", "am_puck", "af_river"] * 2)]))
    clips = []
    for vid, voice, lang in VOICES:
        wake = tts(k, "Okay Nabu", voice, lang)
        for pid, text in PHRASES:
            speech = norm(np.concatenate([np.zeros(int(SR * 0.4), np.float32), wake, np.zeros(int(SR * 0.6), np.float32),
                                          tts(k, text, voice, lang), np.zeros(int(SR * 0.8), np.float32)]))
            for nk in ("none", "keyboard", "fan", "tv"):
                x = speech.copy()
                if nk != "none":
                    nz = noise(nk, len(x), rng, babble)
                    x = x + norm(nz, -28.0)          # ~10 dB below the speech
                name = f"{pid}__{vid}__{nk}.mp3"
                write_mp3(x * 0.9, os.path.join(OUT, name))
                clips.append({"file": name, "phrase": pid, "text": text.replace(" | ", " "), "voice": vid, "noise": nk,
                              "secs": round(len(x) / SR, 2)})
        print("voice done:", vid, flush=True)
    # false-trigger reel: TV talk in many voices, no wake word, ~5 min
    reel = []
    voices = ["af_nicole", "am_eric", "bf_lily", "am_puck", "af_river", "bm_daniel", "em_alex", "af_heart"]
    for rep in range(4):
        for i, t in enumerate(TV_TEXT):
            reel.append(tts(k, t, voices[(i + rep) % len(voices)], "en-us"))
            reel.append(np.zeros(int(SR * rng.uniform(0.3, 1.2)), np.float32))
    reel = norm(np.concatenate(reel))
    write_mp3(reel * 0.9, os.path.join(OUT, "false_reel.mp3"))
    json.dump({"clips": clips, "false_reel": {"file": "false_reel.mp3", "secs": round(len(reel) / SR, 1)}},
              open(os.path.join(OUT, "clips.json"), "w"), indent=1)
    print("clips:", len(clips), "false reel:", round(len(reel) / SR / 60, 1), "min")


if __name__ == "__main__":
    main()
