#!/usr/bin/env python3
"""ha_ws.py '<json command without id>' [...]  - run HA WebSocket commands, print results."""
import asyncio, json, os, sys, websockets
URL = os.environ["HA_URL"].replace("http", "ws", 1) + "/api/websocket"
TOKEN = os.environ["HA_TOKEN"]
async def main(cmds):
    async with websockets.connect(URL, max_size=20_000_000) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": TOKEN}))
        r = json.loads(await ws.recv()); assert r["type"] == "auth_ok", r
        for i, c in enumerate(cmds, 1):
            msg = json.loads(c); msg["id"] = i
            # HA's websocket call_service takes "service_data"; a "data" key was silently ignored (2026-09-26)
            if msg.get("type") == "call_service" and "data" in msg:
                msg["service_data"] = msg.pop("data")
            await ws.send(json.dumps(msg))
            while True:
                r = json.loads(await ws.recv())
                if r.get("id") == i: break
            print(json.dumps(r.get("result") if r.get("success") else {"ERROR": r.get("error")}, indent=1))
asyncio.run(main(sys.argv[1:]))
