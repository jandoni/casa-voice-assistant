#!/usr/bin/env python3
"""
stt_bench.py - benchmark Wyoming speech-to-text servers (the HA Whisper add-on or any
wyoming STT on the LAN): latency after end-of-audio + word accuracy, on synthetic clips.

  python3 stt_bench.py gen [--voices Samantha,Daniel,Karen]      # make clips (macOS `say` + ffmpeg)
  python3 stt_bench.py run HOST:PORT [--set clean|noisy|all] [--runs 2] [--label name] [--realtime]
      --realtime sends the audio at 1x speed (like a real satellite) - the fair way to time STREAMING engines
  python3 stt_bench.py config '<json add-on options>'            # set HA Whisper add-on options + restart + wait
  python3 stt_bench.py compare HOST:PORT                          # run all presets below and print a table

Clips live in ./stt_clips/{clean,noisy}/ next to this file (p*.wav + p*.txt). "noisy" = -8 dB,
room echo, pink noise, 0.8 s trailing silence (≈ Echo Show far-field mic). Latency = time from the
last audio byte sent until the transcript event arrives (what the user feels after they stop talking).
Needs: `pip install wyoming` (venv), ffmpeg, and for `config`/`compare` ssh access to the HA OS host
(HA_SSH, e.g. root@<ha host>, with the Advanced SSH add-on).
"""
import asyncio, glob, json, os, re, subprocess, sys, time, wave

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "stt_clips")
HA_SSH = ["ssh", "-o", "BatchMode=yes", os.environ.get("HA_SSH", "root@homeassistant")]

SENTENCES = [
    "Turn on the bedroom pendant.",
    "It's too bright in the office.",
    "What's the capital of Spain?",
    "Set the thermostat to 70 degrees.",
    "Dim the monitor light to 20 percent.",
    "What time is it?",
    "How tall is Barack Obama?",
    "Who is the CEO of Nvidia?",
    "Turn off everything in the living room.",
    "Set a timer for twelve minutes.",
    "How far is Denver from Boston?",
    "Is the bedroom humidifier on?",
    "It's a bit dark at Jon's desk.",
    "Play Chillout 2026 in the kitchen.",
    "What's the weather tomorrow?",
    "Remind me to call the dentist at five.",
]

# Presets for the HA Whisper add-on (core_whisper >= 3.5). All keys must be present in a POST.
BASE = {"language": "en", "beam_size": 1, "initial_prompt": "", "whisper_task": "transcribe",
        "sherpa_streaming": False, "vad_clip": False, "local_files_only": False,
        "debug_logging": False, "custom_model_type": "faster-whisper"}
PRESETS = {
    "parakeet-tdt-0.6b-v3": {**BASE, "model": "auto", "stt_library": "sherpa"},
    "zipformer-streaming":  {**BASE, "model": "auto", "stt_library": "sherpa", "sherpa_streaming": True},
    "whisper-small-int8":   {**BASE, "model": "small-int8", "stt_library": "faster-whisper"},
    "whisper-base-int8":    {**BASE, "model": "base-int8", "stt_library": "faster-whisper"},
    "whisper-distil-small": {**BASE, "model": "distil-small.en", "stt_library": "faster-whisper"},
}


UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_digits(tokens):
    """Collapse spoken numbers ('two thousand and twenty six', 'seventy') into digits, in place."""
    out, i = [], 0
    while i < len(tokens):
        j, total, cur, seen = i, 0, 0, False
        while j < len(tokens):
            t = tokens[j]
            if t in UNITS: cur += UNITS[t]; seen = True
            elif t in TENS: cur += TENS[t]; seen = True
            elif t == "hundred" and seen: cur *= 100
            elif t == "thousand" and seen: total += cur * 1000; cur = 0
            elif t == "and" and seen and j + 1 < len(tokens) and (tokens[j + 1] in UNITS or tokens[j + 1] in TENS): pass
            else: break
            j += 1
        if seen:
            out.append(str(total + cur)); i = j
        else:
            out.append(tokens[i]); i += 1
    return out


def norm(s):
    s = s.lower().replace("%", " percent").replace("-", " ")
    toks = re.sub(r"[^a-z0-9' ]+", " ", s).split()
    return " ".join(words_to_digits(toks))


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def gen(voices):
    for sub in ("clean", "noisy"):
        os.makedirs(os.path.join(CLIPS, sub), exist_ok=True)
    n = 0
    for v in voices:
        for s in SENTENCES:
            n += 1
            aiff = os.path.join(CLIPS, f"tmp{n}.aiff")
            subprocess.run(["say", "-v", v, "-o", aiff, s], check=True)
            clean = os.path.join(CLIPS, "clean", f"p{n:02d}.wav")
            noisy = os.path.join(CLIPS, "noisy", f"p{n:02d}.wav")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-af", "apad=pad_dur=0.8",
                            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", clean], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-filter_complex",
                            "[0:a]apad=pad_dur=0.8,aecho=0.8:0.7:40|90:0.25|0.15,volume=-8dB,aresample=16000[v];"
                            "anoisesrc=color=pink:amplitude=0.02:sample_rate=16000:duration=8[n];"
                            "[v][n]amix=inputs=2:duration=first:normalize=0,aformat=sample_fmts=s16:channel_layouts=mono[out]",
                            "-map", "[out]", noisy], check=True)
            for p in (clean, noisy):
                open(p[:-4] + ".txt", "w").write(s + "\n")
            os.remove(aiff)
    print(f"generated {n} clips x2 in {CLIPS}")


REALTIME = False


async def transcribe(host, port, path):
    from wyoming.client import AsyncTcpClient
    from wyoming.audio import AudioStart, AudioChunk, AudioStop
    from wyoming.asr import Transcribe, Transcript
    with wave.open(path, "rb") as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        pcm = w.readframes(w.getnframes())
    async with AsyncTcpClient(host, port) as c:
        await c.write_event(Transcribe(language="en").event())
        await c.write_event(AudioStart(rate=rate, width=width, channels=ch).event())
        step = rate * width * ch // 10
        for i in range(0, len(pcm), step):
            await c.write_event(AudioChunk(rate=rate, width=width, channels=ch, audio=pcm[i:i + step]).event())
            if REALTIME:
                await asyncio.sleep(0.1)
        await c.write_event(AudioStop().event())
        t0 = time.perf_counter()
        while True:
            ev = await c.read_event()
            if ev is None:
                raise RuntimeError("connection closed")
            if Transcript.is_type(ev.type):
                return time.perf_counter() - t0, Transcript.from_event(ev).text


async def run(hostport, subset, runs, label, verbose=True):
    host, port = hostport.split(":")
    sets = ["clean", "noisy"] if subset == "all" else [subset]
    out = {}
    for sub in sets:
        files = sorted(glob.glob(os.path.join(CLIPS, sub, "p*.wav")))
        lat, errs, worst = [], [], []
        for r in range(runs):
            for f in files:
                ref = open(f[:-4] + ".txt").read().strip()
                dt, text = await transcribe(host, int(port), f)
                lat.append(dt); e = wer(ref, text); errs.append(e)
                if e > 0:
                    worst.append((e, ref, text.strip()))
                if verbose:
                    print(f"{sub} {os.path.basename(f)} {dt:.2f}s wer={e:.2f} {text.strip()!r}")
        lat.sort()
        res = {"avg": sum(lat) / len(lat), "p90": lat[int(0.9 * (len(lat) - 1))], "max": lat[-1],
               "wer": sum(errs) / len(errs), "worst": sorted(worst, reverse=True)[:3]}
        out[sub] = res
        print(f"== {label or hostport} [{sub}] latency avg {res['avg']:.2f}s  p90 {res['p90']:.2f}s  "
              f"max {res['max']:.2f}s  WER {res['wer']*100:.1f}%")
        for e, ref, hyp in res["worst"]:
            print(f"     miss {e:.2f}: {ref!r} -> {hyp!r}")
    return out


def config(opts):
    body = json.dumps({"options": opts})
    cmd = ("curl -sS -X POST -H \"Authorization: Bearer $SUPERVISOR_TOKEN\" -H 'Content-Type: application/json' "
           "-d @- http://supervisor/addons/core_whisper/options; echo; ha apps restart core_whisper >/dev/null 2>&1; "
           "for i in $(seq 1 60); do sleep 5; ha apps logs core_whisper 2>/dev/null | tail -40 | grep -q 'INFO:__main__:Ready' "
           "&& { echo READY after $((i*5))s; break; }; done")
    r = subprocess.run(HA_SSH + [cmd], input=body, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    time.sleep(3)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "gen":
        voices = "Samantha,Daniel"
        if "--voices" in sys.argv: voices = sys.argv[sys.argv.index("--voices") + 1]
        gen(voices.split(","))
    elif cmd == "run":
        global REALTIME
        REALTIME = "--realtime" in sys.argv
        subset = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "all"
        runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
        label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else ""
        asyncio.run(run(sys.argv[2], subset, runs, label))
    elif cmd == "config":
        config(json.loads(sys.argv[2]))
    elif cmd == "compare":
        table = {}
        for name, opts in PRESETS.items():
            print(f"\n##### {name}")
            config(opts)
            table[name] = asyncio.run(run(sys.argv[2], "all", 1, name, verbose=False))
        print("\n%-24s %-28s %-28s" % ("preset", "clean lat avg/p90 | WER", "noisy lat avg/p90 | WER"))
        for name, r in table.items():
            c, n = r["clean"], r["noisy"]
            print("%-24s %5.2fs/%5.2fs | %5.1f%%        %5.2fs/%5.2fs | %5.1f%%" %
                  (name, c["avg"], c["p90"], c["wer"] * 100, n["avg"], n["p90"], n["wer"] * 100))
        print("\nNOTE: the add-on is left on the LAST preset - re-run `config` with the one you want.")


if __name__ == "__main__":
    main()
