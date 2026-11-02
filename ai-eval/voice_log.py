#!/usr/bin/env python3
"""voice_log.py [N] - show the last N voice interactions of the Casa AI pipeline (what the Echo heard, who
answered, what it said, and stage timings). Needs HA_URL/HA_TOKEN (config.env) and the ha_ws.py helper path
in HA_WS (defaults to ./ha_ws.py)."""
import datetime, json, os, subprocess, sys
WS = os.environ.get("HA_WS", os.path.join(os.path.dirname(__file__), "ha_ws.py"))
PIPE = os.environ["CASA_PIPELINE_ID"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def ws(msg):
    return json.loads(subprocess.run(["python3", WS, json.dumps(msg)], capture_output=True, text=True).stdout)

runs = ws({"type": "assist_pipeline/pipeline_debug/list", "pipeline_id": PIPE})["pipeline_runs"]
for r in runs[-N:]:
    rid = r["pipeline_run_id"] if isinstance(r, dict) else r
    ev = ws({"type": "assist_pipeline/pipeline_debug/get", "pipeline_id": PIPE, "pipeline_run_id": rid})["events"]
    t = {e["type"]: datetime.datetime.fromisoformat(e["timestamp"]) for e in ev}
    d = {e["type"]: (e.get("data") or {}) for e in ev}
    start = min(t.values())
    heard = ((d.get("stt-end") or {}).get("stt_output") or {}).get("text") or (d.get("intent-start") or {}).get("intent_input")
    io = (d.get("intent-end") or {}).get("intent_output") or {}
    resp = io.get("response") or {}
    said = ((resp.get("speech") or {}).get("plain") or {}).get("speech", "")
    local = (d.get("intent-end") or {}).get("processed_locally")
    tools = [c.get("tool_name") for e in ev if e["type"] == "intent-progress" for c in ((e.get("data") or {}).get("chat_log_delta") or {}).get("tool_calls") or []]
    def dur(a, b):
        return f"{(t[b]-t[a]).total_seconds():.1f}s" if a in t and b in t else "-"
    who = "LOCAL" if local else ("LLM " + ("+".join(tools) if tools else "")).strip()
    print(f"{start.astimezone().strftime('%H:%M:%S')}  heard: {heard!r}")
    print(f"           {who:<22} said: {said[:100]!r}")
    print(f"           stt {dur('stt-start','stt-end')} | intent {dur('intent-start','intent-end')} | tts {dur('tts-start','tts-end')} | total {dur('run-start','run-end')}")
