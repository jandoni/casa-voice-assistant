#!/usr/bin/env python3
"""Voice lab runner (Mac; config.env exported). Plays lab clips on a speaker next to an Echo and scores that Echo.

  python3 voice-lab/run_lab.py --tag base --volume 0.25 --match 'us_m__none' [--gain 5] [--ns off] [--out results.jsonl]
  python3 voice-lab/run_lab.py --tag falsereel --false-reel --volume 0.3

Per clip: did the Echo wake (satellite -> listening), the transcript it reported (casa_lab_result, test mode on = no
actions), word error rate vs the reference, time from wake to transcript. --gain / --ns set the VACA mic gain / noise
suppression first. Test mode (input_boolean.casa_voice_lab) is switched on for the run and off at the end; the Sonos
volume is restored.
"""
import argparse, asyncio, json, os, re, sys, time, urllib.request

import websockets

HA, TOKEN = os.environ["HA_URL"], os.environ["HA_TOKEN"]
WAKE_ONLY = False
# VOICELAB_ROOMS: JSON {"<room>": [satellite entity, speaker that plays the clips, VACA device id, clip list file]}.
# The clips (make_clips.py) are served by HA from /config/www/voicelab/ -> <HA_URL>/local/voicelab/.
ROOMS = json.loads(os.environ.get("VOICELAB_ROOMS", json.dumps({
    "office": ["assist_satellite.vaca_<id>", "media_player.office_speaker", "<id>", "clips.json"]})))
SAT, SPK, DEV = next(iter(ROOMS.values()))[:3]
BASE = HA.rstrip("/") + "/local/voicelab/"
NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
       "9": "nine", "10": "ten", "30": "thirty", "20": "twenty", "%": " percent"}


def rest(path, data=None):
    req = urllib.request.Request(HA + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def svc(domain, service, data):
    return rest(f"/api/services/{domain}/{service}", data)


def words(t):
    t = t.lower().replace("%", " percent").replace("what's", "what is").replace("'s", " is")
    t = re.sub(r"\b(\d+)\b", lambda m: NUM.get(m.group(1), m.group(1)), t)
    t = re.sub(r"\bkilometres?\b", "kilometer", t)
    return re.sub(r"[^a-z' ]+", " ", t).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = [[i + j if i * j == 0 else 0 for j in range(len(h) + 1)] for i in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


class Events:
    def __init__(self):
        self.q = []
        self.sat = None

    async def run(self, ws_url):
        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            await ws.recv(); await ws.send(json.dumps({"type": "auth", "access_token": TOKEN})); await ws.recv()
            for i, ev in enumerate(("state_changed", "casa_lab_result"), 1):
                await ws.send(json.dumps({"id": i, "type": "subscribe_events", "event_type": ev}))
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") != "event":
                    continue
                e = m["event"]
                if e["event_type"] == "state_changed":
                    if e["data"]["entity_id"] != SAT or not e["data"]["new_state"]:
                        continue
                    self.sat = e["data"]["new_state"]["state"]
                    self.q.append((time.time(), "sat", self.sat))
                elif e["data"].get("dev") == DEV:
                    self.q.append((time.time(), "lab", e["data"].get("text", "")))


async def wait_for(ev, pred, since, timeout):
    end = time.time() + timeout
    while time.time() < end:
        for t, k, v in ev.q:
            if t >= since and pred(k, v):
                return t, v
        await asyncio.sleep(0.05)
    return None, None


async def trial(ev, clip):
    await wait_idle(ev)
    t0 = time.time()
    svc("media_player", "play_media", {"entity_id": SPK, "media_content_id": BASE + clip["file"], "media_content_type": "music"})
    tw, _ = await wait_for(ev, lambda k, v: k == "sat" and v == "listening", t0, clip["secs"] + 5)
    res = {"file": clip["file"], "phrase": clip.get("phrase"), "voice": clip.get("voice"), "noise": clip.get("noise"),
           "woke": tw is not None, "wake_s": round(tw - t0, 2) if tw else None}
    if tw and WAKE_ONLY:
        # wake-word-only clip: end the listening session at once (script.casa_voice_cancel) instead of letting the Echo
        # listen to silence until it times out (~12 s per clip, and whisper invents "Thank you for watching")
        await asyncio.sleep(0.8)
        svc("script", "turn_on", {"entity_id": "script.casa_voice_cancel", "variables": {"dev": DEV}})
        res.update(text="", wer=None, result_s=None)
        await asyncio.sleep(5)   # the cancel mutes the mic 0.7 s and reconnects the satellite: let it settle
    elif tw:
        tr, text = await wait_for(ev, lambda k, v: k == "lab", tw, 25)
        res.update(text=text, wer=round(wer(clip["text"], text or ""), 3) if tr else 1.0,
                   result_s=round(tr - tw, 2) if tr else None)
    await asyncio.sleep(max(0, t0 + clip["secs"] + 2.5 - time.time()))   # let the clip finish
    return res


async def wait_idle(ev, quiet=1.5, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        last = ev.q[-1][0] if ev.q else 0
        if ev.sat in (None, "idle") and time.time() - last > quiet:
            return
        await asyncio.sleep(0.2)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--volume", type=float, default=0.25)
    ap.add_argument("--match", default=""); ap.add_argument("--gain", type=float); ap.add_argument("--ns")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--false-reel", nargs="?", const="false_reel", default=None)
    ap.add_argument("--room", default=next(iter(ROOMS)), choices=list(ROOMS))
    ap.add_argument("--gap", type=float, default=0, help="extra seconds between clips (2026-10-18: misses came in windows)")
    ap.add_argument("--wake-only", action="store_true"); ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "results.jsonl"))
    a = ap.parse_args()
    global WAKE_ONLY, SAT, SPK, DEV
    WAKE_ONLY = a.wake_only
    SAT, SPK, DEV, meta_file = ROOMS[a.room]
    meta = json.load(urllib.request.urlopen(BASE + meta_file, timeout=20))
    spk = rest(f"/api/states/{SPK}")
    if spk["state"] == "playing":
        sys.exit(f"{SPK} is playing - not running the lab")
    old_vol = spk["attributes"].get("volume_level", 0.17)
    ev = Events()
    task = asyncio.ensure_future(ev.run(HA.replace("http", "ws", 1) + "/api/websocket"))
    await asyncio.sleep(1.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    try:
        svc("input_boolean", "turn_on", {"entity_id": "input_boolean.casa_voice_lab"})
        # (the music ducking skips itself while input_boolean.casa_voice_lab is on - no more turning it off here)
        if a.gain is not None:
            svc("number", "set_value", {"entity_id": f"number.vaca_{DEV}_mic_gain", "value": a.gain})
        if a.ns:
            svc("select", "select_option", {"entity_id": f"select.vaca_{DEV}_noise_suppression_level", "option": a.ns})
        svc("media_player", "volume_set", {"entity_id": SPK, "volume_level": a.volume})
        await asyncio.sleep(2)
        if a.false_reel:
            fr = meta[a.false_reel]; t0 = time.time()
            svc("media_player", "play_media", {"entity_id": SPK, "media_content_id": BASE + fr["file"], "media_content_type": "music"})
            await asyncio.sleep(fr["secs"] + 3)
            wakes = [t for t, k, v in ev.q if k == "sat" and v == "listening" and t >= t0]
            heard = [v for t, k, v in ev.q if k == "lab" and t >= t0]
            out = {"tag": a.tag, "false_reel": a.false_reel, "volume": a.volume, "minutes": round(fr["secs"] / 60, 1), "false_wakes": len(wakes), "heard": heard}
            print(json.dumps(out)); open(a.out, "a").write(json.dumps(out) + "\n")
            return
        clips = [c for c in meta["clips"] if re.search(a.match, c["file"])]
        if a.limit:
            clips = clips[:a.limit]
        for i, c in enumerate(clips, 1):
            r = await trial(ev, c)
            r.update(tag=a.tag, volume=a.volume, gain=a.gain, ns=a.ns, ts=round(time.time()))
            open(a.out, "a").write(json.dumps(r) + "\n")
            print(f"[{i}/{len(clips)}] {c['file']:34} woke={r['woke']!s:5} wer={r.get('wer', '-')!s:5} t={r.get('result_s')}  {r.get('text', '')!r}", flush=True)
            if a.gap:
                await asyncio.sleep(a.gap)
    finally:
        svc("media_player", "volume_set", {"entity_id": SPK, "volume_level": old_vol})
        svc("input_boolean", "turn_off", {"entity_id": "input_boolean.casa_voice_lab"})
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
