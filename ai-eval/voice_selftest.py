#!/usr/bin/env python3
"""
voice_selftest.py - end-to-end voice test WITHOUT a human: plays "Hey Jarvis, <command>" through a speaker
next to the Echo Show, then checks (a) what the pipeline heard, (b) what it answered, (c) whether the
device really changed. Clips are made with macOS `say` (Samantha = US English, Monica/Jorge = Spanish
voices reading English = an exaggerated Spanish accent, only useful for wake-word sensitivity; Daniel = UK, Rishi = Indian English).

  set -a; source config.env; set +a
  python3 ai-eval/voice_selftest.py [--volume 0.15] [--voices Samantha,Monica] [--only desk]
                                     [--speaker media_player.test_room_speaker] [--dry]

TESTS below targets three example lights in one room: replace the entity ids with yours and only run it when nobody
is in that room. Restores light states and the speaker volume at the end. Results are printed and appended to
ai-eval/voice_selftest.log.
"""
import argparse, datetime, json, os, subprocess, sys, time, urllib.request

HA = os.environ.get("HA_URL", "http://127.0.0.1:8123")
TOKEN = os.environ["HA_TOKEN"]
PIPE = os.environ["CASA_PIPELINE_ID"]   # the "Casa AI" pipeline id
HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "selftest_clips")
LOG = os.path.join(HERE, "voice_selftest.log")
HA_SSH = os.environ.get("HA_SSH", "root@homeassistant")   # ssh to the HA OS host (clips go to /config/www/selftest)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HA_SSH]
SCP = ["scp", "-q", "-o", "ConnectTimeout=10"]
WAKE = "Okay Nabu"

# (key, spoken command, setup {entity: service}, check(states_before, states_after, response) -> (ok, note))
def _on(e): return lambda b, a, r: (a[e]["state"] == "on", f"{e}={a[e]['state']}")
def _off(e): return lambda b, a, r: (a[e]["state"] == "off", f"{e}={a[e]['state']}")
def _dimmer(e): return lambda b, a, r: ((a[e]["attributes"].get("brightness") or 0) < (b[e]["attributes"].get("brightness") or 0), f"brightness {b[e]['attributes'].get('brightness')}→{a[e]['attributes'].get('brightness')}")
def _bri(e, pct): return lambda b, a, r: (abs((a[e]["attributes"].get("brightness") or 0) / 2.55 - pct) <= 3, f"brightness {round((a[e]['attributes'].get('brightness') or 0) / 2.55)}%")
def _says(*words): return lambda b, a, r: (any(w.lower() in (r or "").lower() for w in words), f"said {r!r}")

def _state(e, want): return lambda b, a, r: (a[e]["state"] == want, f"{e}={a[e]['state']}")
def _brighter(e): return lambda b, a, r: ((a[e]["attributes"].get("brightness") or 0) > (b[e]["attributes"].get("brightness") or 0), f"brightness {b[e]['attributes'].get('brightness')}→{a[e]['attributes'].get('brightness')}")
def _vol_lower(e): return lambda b, a, r: ((a[e]["attributes"].get("volume_level") or 0) < (b[e]["attributes"].get("volume_level") or 0), f"volume {b[e]['attributes'].get('volume_level')}→{a[e]['attributes'].get('volume_level')}")
def _vol(e, pct): return lambda b, a, r: (abs((a[e]["attributes"].get("volume_level") or 0) * 100 - pct) <= 3, f"volume {a[e]['attributes'].get('volume_level')}")
def _all_off(*es): return lambda b, a, r: (all(a[e]["state"] == "off" for e in es), ", ".join(f"{e.split('.')[1]}={a[e]['state']}" for e in es))
def _all_on(*es): return lambda b, a, r: (all(a[e]["state"] == "on" for e in es), ", ".join(f"{e.split('.')[1]}={a[e]['state']}" for e in es))
def _any(*fns): return lambda b, a, r: (any(f(b, a, r)[0] for f in fns), " / ".join(f(b, a, r)[1] for f in fns))

# Example test room: three lights, a speaker. Replace the entity ids with your own.
MONITOR, DESK, READING = "light.monitor_bar", "light.office_desk", "light.office_reading"
SPEAKER = "media_player.test_room_speaker"
TESTS = [
    # --- lights: on/off/dim (device state is checked)
    ("monitor_on",    "turn on the monitor light",             {MONITOR: "turn_off"},                    _on(MONITOR)),
    ("monitor_off",   "turn off the monitor light",            {MONITOR: "turn_on"},                     _off(MONITOR)),
    ("monitor_dim20", "dim the monitor light to 20 percent",   {MONITOR: "turn_on"},                     _bri(MONITOR, 20)),
    ("monitor_set60", "set the monitor light to 60 percent",   {MONITOR: "turn_on"},                     _bri(MONITOR, 60)),
    ("desk_on",       "turn on the desk lamp",                 {DESK: "turn_off"},                       _on(DESK)),
    ("desk_off",      "turn off the desk lamp",                {DESK: "turn_on"},                        _off(DESK)),
    ("desk_30",       "set the desk lamp to 30 percent",       {DESK: "turn_on"},                        _bri(DESK, 30)),
    ("reading_on",    "turn on the reading lamp",              {READING: "turn_off"},                    _on(READING)),
    ("reading_50",    "dim the reading lamp to 50 percent",    {READING: "turn_on"},                     _bri(READING, 50)),
    ("office_off",    "turn off the office lights",            {DESK: "turn_on", READING: "turn_on"},    _all_off(DESK, READING)),
    ("office_on",     "turn on the office lights",             {DESK: "turn_off", READING: "turn_off"},  _all_on(DESK, READING)),
    ("office_all_off","turn off everything in the office",     {DESK: "turn_on", MONITOR: "turn_on"},    _all_off(DESK, MONITOR)),
    # --- comfort phrases
    ("too_bright",    "it's too bright in the office",         {DESK: "turn_on", READING: "turn_on"},    _dimmer(DESK)),
    ("too_dark",      "it's too dark in the office",           {DESK: "turn_on"},                        _any(_brighter(DESK), _brighter(READING))),
    ("cozy_office",   "make it cozy in the office",            {DESK: "turn_on"},                        _any(_bri(DESK, 30), _dimmer(DESK))),
    # --- status questions (spoken answer is checked)
    ("status_monitor","is the monitor light on",               {MONITOR: "turn_off"},                    _says("off", "no")),
    ("status_desk",   "is the desk lamp on",                   {DESK: "turn_on"},                        _says("yes", "on")),
    ("status_office", "which lights are on in the office",     {DESK: "turn_on", READING: "turn_off"},   _says("desk", "on")),
    # --- speaker
    ("too_loud",      "it's too loud in the office",           {},                                       _vol_lower(SPEAKER)),
    ("speaker_vol",   "set the office speaker volume to 20 percent", {},                                 _vol(SPEAKER, 20)),
    # --- general, nothing changes
    ("time",          "what time is it",                       {},                                       _says(":", "o'clock", "am", "pm")),
    ("weather",       "what's the weather",                    {},                                       _says("°", "degrees", "cloud", "rain", "sun", "clear")),
    ("temp_inside",   "what's the temperature inside",         {},                                       _says("°", "degrees")),
    ("timer_set",     "set a timer for 2 minutes",             {},                                       _says("timer", "2 minutes", "two minutes", "started")),
    ("timer_cancel",  "cancel the timer",                      {},                                       _says("cancel", "timer", "stopped", "no timer")),
    ("capital",       "what's the capital of Spain",           {},                                       _says("madrid")),
]
LIGHTS = [MONITOR, DESK, READING, SPEAKER]


def api(path, data=None, method=None):
    req = urllib.request.Request(HA + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
                                 method=method or ("POST" if data is not None else "GET"))
    return json.load(urllib.request.urlopen(req, timeout=30))


def ws(cmd):
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "ha_ws.py"), json.dumps(cmd)], capture_output=True, text=True, timeout=25).stdout
        return json.loads(out)
    except Exception:  # noqa: BLE001
        return {}


def state(e):
    return api(f"/api/states/{e}")


def make_clip(key, text, voice, gap=1.0):
    os.makedirs(CLIPS, exist_ok=True)
    out = os.path.join(CLIPS, f"{key}_{voice}.mp3")
    if os.path.exists(out):
        return out
    w, c = os.path.join(CLIPS, "_w.aiff"), os.path.join(CLIPS, "_c.aiff")
    subprocess.run(["say", "-v", voice, "-r", "165", "-o", w, WAKE], check=True)
    subprocess.run(["say", "-v", voice, "-r", "165", "-o", c, text], check=True)
    # 1.5 s lead-in (Sonos clips the start of a stream), wake word, gap (tone + reaction), command, 1 s tail
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", w, "-i", c, "-filter_complex",
                    f"[0:a]adelay=1500|1500,apad=pad_dur={gap}[a];[1:a]apad=pad_dur=1.0[b];[a][b]concat=n=2:v=0:a=1,aformat=sample_rates=44100:channel_layouts=mono[out]",
                    "-map", "[out]", "-codec:a", "libmp3lame", "-b:a", "128k", out], check=True)
    for f in (w, c):
        os.remove(f)
    return out


def latest_run_after(ts_iso):
    runs = ws({"type": "assist_pipeline/pipeline_debug/list", "pipeline_id": PIPE}).get("pipeline_runs") or []
    new = [r for r in runs if r["timestamp"] > ts_iso]
    return new[-1]["pipeline_run_id"] if new else None


def run_events(rid):
    return ws({"type": "assist_pipeline/pipeline_debug/get", "pipeline_id": PIPE, "pipeline_run_id": rid}).get("events") or []


def summarize(ev):
    stt, resp, err, done = None, None, None, False
    t0 = datetime.datetime.fromisoformat(ev[0]["timestamp"]) if ev else None
    stt_t = None
    for e in ev:
        if e["type"] == "stt-end":
            stt = e["data"].get("stt_output", {}).get("text"); stt_t = (datetime.datetime.fromisoformat(e["timestamp"]) - t0).total_seconds()
        elif e["type"] == "intent-end":
            resp = ((e["data"].get("intent_output") or {}).get("response") or {}).get("speech", {}).get("plain", {}).get("speech")
        elif e["type"] == "error":
            err = e["data"].get("code")
        elif e["type"] == "run-end":
            done = True
    return stt, resp, err, done, stt_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=float, default=0.15)
    ap.add_argument("--voices", default="Samantha,Monica")
    ap.add_argument("--only", default="")
    ap.add_argument("--speaker", default=SPEAKER)
    ap.add_argument("--dry", action="store_true", help="make the clips only")
    a = ap.parse_args()
    voices = a.voices.split(",")
    keys = [k for k in a.only.split(",") if k]
    tests = [t for t in TESTS if not keys or any(k in t[0] for k in keys)]
    if a.dry:
        for key, text, _, _ in tests:
            for v in voices:
                print(make_clip(key, text, v))
        return
    snap = {e: state(e) for e in LIGHTS}
    vol_before = state(a.speaker)["attributes"].get("volume_level")
    api("/api/services/media_player/volume_set", {"entity_id": a.speaker, "volume_level": a.volume})
    results = []
    # make + upload every clip once (Sonos fetches them from HA's /local/selftest/)
    clips = [make_clip(key, text, v) for key, text, _, _ in tests for v in voices]
    subprocess.run(SSH + ["mkdir -p /config/www/selftest"], capture_output=True, timeout=30)
    subprocess.run(SCP + clips + [HA_SSH + ":/config/www/selftest/"], timeout=120)
    try:
        for key, text, setup, check in tests:
            for v in voices:
                clip = make_clip(key, text, v)
                for e, svc in setup.items():
                    dom = e.split(".")[0]
                    api(f"/api/services/{dom}/{svc}", {"entity_id": e} | ({"brightness_pct": 100} if svc == "turn_on" and dom == "light" else {}))
                time.sleep(2.5)
                before = {e: state(e) for e in LIGHTS}
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                api("/api/services/media_player/play_media", {"entity_id": a.speaker, "media_content_id": f"{HA}/local/selftest/{os.path.basename(clip)}", "media_content_type": "music"})
                rid = None
                for _ in range(16):
                    time.sleep(1.0)
                    rid = latest_run_after(ts)
                    if rid:
                        break
                stt = resp = err = None; stt_t = None
                if rid:
                    for _ in range(45):
                        ev = run_events(rid)
                        stt, resp, err, done, stt_t = summarize(ev)
                        if done or err:
                            break
                        time.sleep(1.0)
                time.sleep(2.0)
                after = {e: state(e) for e in LIGHTS}
                try:
                    ok, note = check(before, after, resp)
                except Exception as ex:  # noqa: BLE001
                    ok, note = False, f"check error {ex}"
                if not rid:
                    ok, note = False, "no wake-up (pipeline never ran)"
                line = (f"{datetime.datetime.now().strftime('%H:%M:%S')} {'PASS' if ok else 'FAIL'} {key:12s} {v:9s} "
                        f"heard={stt!r} stt@{stt_t if stt_t is None else round(stt_t, 1)}s response={resp!r} err={err} | {note}")
                print(line, flush=True)
                results.append(ok)
                with open(LOG, "a") as f:
                    f.write(line + "\n")
                time.sleep(4.0)
    finally:
        # restore lights and volume
        for e, s in snap.items():
            dom = e.split(".")[0]
            if dom == "media_player":
                continue  # its volume is restored below
            if s["state"] == "on":
                api(f"/api/services/{dom}/turn_on", {"entity_id": e} | ({"brightness": s["attributes"]["brightness"]} if dom == "light" and s["attributes"].get("brightness") else {}))
            elif s["state"] in ("off",):
                api(f"/api/services/{dom}/turn_off", {"entity_id": e})
        if vol_before is not None:
            api("/api/services/media_player/volume_set", {"entity_id": a.speaker, "volume_level": vol_before})
    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
