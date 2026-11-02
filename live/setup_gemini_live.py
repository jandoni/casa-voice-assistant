#!/usr/bin/env python3
"""Set up Gemini 3.8 Live as a separate HA voice pipeline "Casa Live" (2026-10-16). Silent: touches no Echo.

  python3 live/setup_gemini_live.py             # create/refresh the Gemini Live entry + the "Casa Live" pipeline
  python3 live/setup_gemini_live.py --assign bedroom|office|living   # put ONE Echo on Casa Live
  python3 live/setup_gemini_live.py --revert bedroom|office|living   # back to "Casa AI" (the old pipeline)

Needs GEMINI_API_KEY in config.env (keep the key itself in your password manager) - OR create the integration entry by hand in
HA (Settings -> Devices & services -> Add integration -> Gemini Live, paste the key) and this script finds and
reconfigures it, so the key never has to touch config.env.
"""
import json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # casa-voice repo root
# config.env (copy config.env.example; never commit it); HA_CONFIG_ENV points somewhere else.
CONFIG_ENV = next((c for c in [os.environ.get("HA_CONFIG_ENV", ""), os.path.join(ROOT, "config.env")]
                   if c and os.path.isfile(c)), os.path.join(ROOT, "config.env"))
env = {}
for line in open(CONFIG_ENV):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
HA = env.get("HA_URL", "http://127.0.0.1:8123").rstrip("/")
TOKEN = env["HA_TOKEN"]
# CASA_ECHOS="<room>=<VACA device id>,..." (the id in entity names like select.vaca_<id>_assistant)
ECHOS = dict(p.split("=", 1) for p in env.get("CASA_ECHOS", "").split(",") if "=" in p)
OLD_PIPELINE = "Casa AI"
PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "casa_live_prompt.txt")).read().strip()
SETTINGS = {
    "model": "gemini-3.8-live",
    "voice": "Aoede",                 # relaxed, natural female (closest to Kokoro 'Bella'); try Sulafat / Achernar / Despina
    "system_instruction": PROMPT,
    "detailed_logging": False,
    "transcribe_gemini": True,        # text of what you said / the reply -> the Echo voice screen + logbook
    "response_timeout": 30,           # the Live session closes 30 s after the last turn (cost: history is re-billed)
    "enable_sound_responses": False,
    "llm_hass_api": ["casa"],         # OUR tools only (custom_components/casa_live) - not HA's big 'assist' entity dump
    "search_grounding": False,        # OFF: audio turns get cut off with grounding on (ha-gemini-live issue #13, 2026-10-14)
}


def rest(method, path, body=None):
    req = urllib.request.Request(HA + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        t = r.read()
        return json.loads(t) if t else {}


def ws(msg):
    import asyncio
    import websockets  # pip install websockets (already used by ai-eval/ha_ws.py)

    async def go():
        async with websockets.connect(HA.replace("http", "ws", 1) + "/api/websocket", max_size=2**24) as s:
            await s.recv(); await s.send(json.dumps({"type": "auth", "access_token": TOKEN})); await s.recv()
            await s.send(json.dumps({"id": 1, **msg}))
            while True:
                r = json.loads(await s.recv())
                if r.get("id") == 1:
                    return r
    return asyncio.run(go())


def entry():
    es = [e for e in rest("GET", "/api/config/config_entries/entry") if e["domain"] == "gemini_live"]
    return es[0] if es else None


def setup():
    e = entry()
    if not e:
        key = env.get("GEMINI_API_KEY")
        if not key:
            sys.exit("No Gemini Live entry and no GEMINI_API_KEY in config.env. Add the integration in HA (paste the key) or "
                     "put the key in config.env, then run me again.")
        f = rest("POST", "/api/config/config_entries/flow", {"handler": "gemini_live"})
        f = rest("POST", f"/api/config/config_entries/flow/{f['flow_id']}", {"provider": "gemini"})
        f = rest("POST", f"/api/config/config_entries/flow/{f['flow_id']}", {"api_key": key, **SETTINGS})
        print("created entry:", f.get("type"), f.get("title"))
    else:
        # options flow = the settings above (the key stays what it is)
        f = rest("POST", "/api/config/config_entries/options/flow", {"handler": e["entry_id"]})
        f = rest("POST", f"/api/config/config_entries/options/flow/{f['flow_id']}", SETTINGS)
        print("updated entry options:", f.get("type"))
    e = entry()
    reg = ws({"type": "config/entity_registry/list"})["result"]
    mine = [x["entity_id"] for x in reg if x.get("config_entry_id") == e["entry_id"]]
    pick = lambda d: next(x for x in mine if x.startswith(d + "."))
    stt, conv, tts = pick("stt"), pick("conversation"), pick("tts")
    pipes = ws({"type": "assist_pipeline/pipeline/list"})["result"]["pipelines"]
    cur = next((p for p in pipes if p["name"] == "Casa Live"), None)
    body = {"name": "Casa Live", "language": "en", "conversation_engine": conv, "conversation_language": "en",
            "stt_engine": stt, "stt_language": "en", "tts_engine": tts, "tts_language": "en", "tts_voice": None,
            "wake_word_entity": None, "wake_word_id": None, "prefer_local_intents": False}
    if cur:
        r = ws({"type": "assist_pipeline/pipeline/update", "pipeline_id": cur["id"], **body})
    else:
        r = ws({"type": "assist_pipeline/pipeline/create", **body})
    print("pipeline Casa Live:", r.get("success"), stt, conv, tts)


def assign(room, name):
    dev = ECHOS[room]
    # the helper first: every automation that re-selects the pipeline (reconnect, mic watchdog, X) follows it
    rest("POST", "/api/services/input_select/select_option", {"entity_id": f"input_select.casa_pipeline_{dev}", "option": name})
    rest("POST", "/api/services/select/select_option", {"entity_id": f"select.vaca_{dev}_assistant", "option": name})
    print(f"{room} Echo -> {name}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--assign":
        assign(sys.argv[2], "Casa Live")
    elif len(sys.argv) == 3 and sys.argv[1] == "--revert":
        assign(sys.argv[2], OLD_PIPELINE)
    else:
        setup()
