#!/usr/bin/env python3
"""assist-learn - the "learning loop" for the local fast path.

Every few minutes (systemd timer on the voice host) it pulls Home Assistant's Assist pipeline debug runs and
records every phrase that was NOT answered by the local matcher/Canonicalizer, i.e. that reached the
LLM (Casa), together with what the LLM did (tool calls) and answered. Reviewing that list is how new
aliases / sentence templates get added, so the LLM sees fewer and fewer of the same requests.

  assist_learn.py poll     # append new fall-through runs to STATE_DIR/fallthrough.jsonl
  assist_learn.py report   # print the phrases seen so far, most frequent first

Needs HA_URL and HA_TOKEN in the environment (/etc/assist-learn.env on the voice host).
A run counts as "fell through" when the intent stage took longer than LOCAL_MAX_S (local paths finish
in < 0.5 s; the model needs > 1.5 s) or when HA reports it was not processed locally.
"""
import asyncio, json, os, sys, time
from collections import Counter
from datetime import datetime, timezone
import websockets

URL = os.environ["HA_URL"].replace("http", "ws", 1) + "/api/websocket"
TOKEN = os.environ["HA_TOKEN"]
STATE_DIR = os.environ.get("STATE_DIR", "/var/lib/assist-learn")
LOCAL_MAX_S = float(os.environ.get("LOCAL_MAX_S", "1.5"))
SEEN = os.path.join(STATE_DIR, "seen_runs.json")
LOG = os.path.join(STATE_DIR, "fallthrough.jsonl")


async def ws_call(ws, mid, msg):
    msg["id"] = mid
    await ws.send(json.dumps(msg))
    while True:
        r = json.loads(await ws.recv())
        if r.get("id") == mid:
            return r.get("result") if r.get("success") else None


def parse_run(events):
    text = None; started = None; ended = None; local = None; speech = ""; rtype = ""; calls = []
    for ev in events:
        t = ev.get("type"); d = ev.get("data") or {}; ts = ev.get("timestamp")
        if t == "intent-start":
            text = d.get("intent_input"); started = ts
        elif t == "intent-progress":
            for c in ((d.get("chat_log_delta") or {}).get("tool_calls") or []):
                calls.append({"tool": c.get("tool_name"), "args": c.get("tool_args")})
        elif t == "intent-end":
            ended = ts; local = d.get("processed_locally")
            resp = (d.get("intent_output") or {}).get("response") or {}
            speech = ((resp.get("speech") or {}).get("plain") or {}).get("speech", "")
            rtype = resp.get("response_type", "")
    if not text or not started or not ended:
        return None
    dur = (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds()
    return {"text": text, "seconds": round(dur, 1), "local": bool(local), "response_type": rtype,
            "speech": speech[:200], "tool_calls": calls, "at": started}


async def poll():
    os.makedirs(STATE_DIR, exist_ok=True)
    seen = set(json.load(open(SEEN))) if os.path.exists(SEEN) else set()
    new = 0
    async with websockets.connect(URL, max_size=50_000_000) as ws:
        await ws.recv(); await ws.send(json.dumps({"type": "auth", "access_token": TOKEN})); await ws.recv()
        mid = 1
        pipelines = (await ws_call(ws, mid, {"type": "assist_pipeline/pipeline/list"}))["pipelines"]
        for p in pipelines:
            mid += 1
            runs = await ws_call(ws, mid, {"type": "assist_pipeline/pipeline_debug/list", "pipeline_id": p["id"]})
            for run in (runs or {}).get("pipeline_runs", []):
                run_id = run["pipeline_run_id"] if isinstance(run, dict) else run
                key = f'{p["id"]}:{run_id}'
                if key in seen:
                    continue
                mid += 1
                run = await ws_call(ws, mid, {"type": "assist_pipeline/pipeline_debug/get", "pipeline_id": p["id"], "pipeline_run_id": run_id})
                seen.add(key)
                info = parse_run((run or {}).get("events", []))
                if not info:
                    continue
                if info["seconds"] > LOCAL_MAX_S or (info["local"] is False and info["seconds"] > 1.0):
                    info["pipeline"] = p["name"]
                    with open(LOG, "a") as f:
                        f.write(json.dumps(info, ensure_ascii=False) + "\n")
                    new += 1
    json.dump(sorted(seen)[-2000:], open(SEEN, "w"))
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} poll: {new} new fall-through phrase(s)")


def report():
    if not os.path.exists(LOG):
        print("nothing recorded yet"); return
    rows = [json.loads(l) for l in open(LOG) if l.strip()]
    by = {}
    for r in rows:
        k = r["text"].strip().lower()
        by.setdefault(k, []).append(r)
    print(f"{len(rows)} fall-through runs, {len(by)} distinct phrases\n")
    print(f"{'count':>5}  {'avg s':>6}  phrase  ->  what the model did")
    for k, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        avg = sum(r["seconds"] for r in rs) / len(rs)
        tools = Counter(c["tool"] for r in rs for c in r["tool_calls"])
        did = ", ".join(f"{t}×{n}" for t, n in tools.most_common(3)) or rs[-1]["response_type"]
        print(f"{len(rs):>5}  {avg:>6.1f}  {k!r}  ->  {did} | {rs[-1]['speech'][:60]!r}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    asyncio.run(poll()) if cmd == "poll" else report()
