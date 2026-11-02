#!/usr/bin/env python3
"""
tts_bench.py - benchmark a Wyoming text-to-speech server (HA Piper add-on or any wyoming TTS).

  python3 tts_bench.py run HOST:PORT [--voice en_US-lessac-medium] [--runs 2] [--label name]
  python3 tts_bench.py config '<json add-on options>'     # set HA Piper add-on options + restart + wait
  python3 tts_bench.py compare HOST:PORT                  # try the VOICES list below, print a table

Reports, per phrase: time to FIRST audio chunk (what the user hears as "starts talking") and total
synthesis time; plus audio seconds produced per second of compute (RTF⁻¹). Needs `pip install wyoming`.
"""
import asyncio, json, subprocess, sys, time

HA_SSH = ["ssh", "-o", "BatchMode=yes", __import__("os").environ.get("HA_SSH", "root@homeassistant")]
PHRASES = [
    "Done.",
    "The bedroom ceiling light is now on.",
    "It's 72 degrees in the living room and the thermostat is set to 70.",
    "Madrid is the capital of Spain. It has about 3.4 million people and sits in the centre of the country.",
]
VOICES = ["en_US-lessac-medium", "en_US-lessac-low", "en_US-amy-medium", "en_US-amy-low",
          "en_US-hfc_female-medium", "en_US-ryan-medium", "en_GB-alba-medium"]
BASE = {"debug_logging": False, "length_scale": 1, "noise_scale": 0.667, "noise_w": 0.333,
        "speaker": 0, "update_voices": False}


async def synth(host, port, text, voice):
    from wyoming.client import AsyncTcpClient
    from wyoming.tts import Synthesize, SynthesizeVoice
    from wyoming.audio import AudioChunk, AudioStop
    first, total, nbytes, rate = None, None, 0, 22050
    t0 = time.perf_counter()
    async with AsyncTcpClient(host, port) as c:
        await c.write_event(Synthesize(text=text, voice=SynthesizeVoice(name=voice) if voice else None).event())
        while True:
            ev = await c.read_event()
            if ev is None:
                break
            if AudioChunk.is_type(ev.type):
                ch = AudioChunk.from_event(ev)
                if first is None:
                    first = time.perf_counter() - t0
                nbytes += len(ch.audio); rate = ch.rate; width, chans = ch.width, ch.channels
            elif AudioStop.is_type(ev.type):
                total = time.perf_counter() - t0
                break
    secs = nbytes / (rate * width * chans) if nbytes else 0
    return first, total, secs


async def run(hostport, voice, runs, label, verbose=True):
    host, port = hostport.split(":")
    firsts, totals, speed = [], [], []
    for r in range(runs):
        for p in PHRASES:
            f, t, secs = await synth(host, int(port), p, voice)
            firsts.append(f); totals.append(t); speed.append(secs / t if t else 0)
            if verbose:
                print(f"first {f:.2f}s total {t:.2f}s audio {secs:.1f}s  {p[:50]!r}")
    res = {"first_avg": sum(firsts) / len(firsts), "first_max": max(firsts),
           "total_avg": sum(totals) / len(totals), "total_max": max(totals), "x_realtime": sum(speed) / len(speed)}
    print(f"== {label or voice or hostport}: first-audio avg {res['first_avg']:.2f}s max {res['first_max']:.2f}s | "
          f"total avg {res['total_avg']:.2f}s max {res['total_max']:.2f}s | {res['x_realtime']:.1f}x realtime")
    return res


def config(opts):
    body = json.dumps({"options": opts})
    cmd = ("curl -sS -X POST -H \"Authorization: Bearer $SUPERVISOR_TOKEN\" -H 'Content-Type: application/json' "
           "-d @- http://supervisor/addons/core_piper/options; echo; ha apps restart core_piper >/dev/null 2>&1; "
           "for i in $(seq 1 40); do sleep 3; ha apps logs core_piper 2>/dev/null | tail -30 | grep -q 'INFO:__main__:Ready' "
           "&& { echo READY after $((i*3))s; break; }; done")
    r = subprocess.run(HA_SSH + [cmd], input=body, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    time.sleep(2)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "run":
        voice = sys.argv[sys.argv.index("--voice") + 1] if "--voice" in sys.argv else None
        runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
        label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else ""
        asyncio.run(run(sys.argv[2], voice, runs, label))
    elif cmd == "config":
        config(json.loads(sys.argv[2]))
    elif cmd == "compare":
        table = {}
        for v in VOICES:
            print(f"\n##### {v}")
            config({**BASE, "voice": v})
            asyncio.run(run(sys.argv[2], v, 1, v, verbose=False))  # warm-up (voice download/load)
            table[v] = asyncio.run(run(sys.argv[2], v, 2, v, verbose=False))
        print("\n%-26s %-22s %-22s %s" % ("voice", "first audio avg/max", "total avg/max", "x realtime"))
        for v, r in table.items():
            print("%-26s %5.2fs/%5.2fs          %5.2fs/%5.2fs          %4.1fx" %
                  (v, r["first_avg"], r["first_max"], r["total_avg"], r["total_max"], r["x_realtime"]))
        print("\nNOTE: the add-on is left on the LAST voice - re-run `config` with the one you want.")


if __name__ == "__main__":
    main()
