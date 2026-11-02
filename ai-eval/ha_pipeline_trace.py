import asyncio, json, os, sys, time, websockets
URL=os.environ["HA_URL"].replace("http","ws",1)+"/api/websocket"; TOKEN=os.environ["HA_TOKEN"]
PIPE=os.environ["CASA_PIPELINE_ID"]  # the "Casa AI" pipeline id (assist_pipeline/pipeline/list)
async def main(text):
    async with websockets.connect(URL, max_size=20_000_000) as ws:
        await ws.recv(); await ws.send(json.dumps({"type":"auth","access_token":TOKEN})); await ws.recv()
        t=time.time()
        await ws.send(json.dumps({"id":1,"type":"assist_pipeline/run","start_stage":"intent","end_stage":"intent","input":{"text":text},"pipeline":PIPE}))
        deltas=[]
        while True:
            m=json.loads(await ws.recv())
            if m.get("type")!="event": 
                if m.get("type")=="result" and not m.get("success"): print("ERR", m); return
                continue
            ev=m["event"]; et=ev["type"]; d=ev.get("data") or {}
            if et=="intent-progress":
                delta=d.get("chat_log_delta") or {}
                deltas.append(delta)
            elif et=="intent-end":
                io=d["intent_output"]; r=io.get("conversation_response") or io.get("response") or {}; r=r if "speech" in r else {"response_type":str(io)[:80],"speech":{"plain":{"speech":""}}}
                print(f"[{time.time()-t:.1f}s] {text!r} -> {r['response_type']} | {r['speech']['plain']['speech'][:100]!r}")
                break
            elif et=="run-end": break
        # summarize what the model emitted
        content="".join(x.get("content") or "" for x in deltas); thinking="".join(x.get("thinking_content") or "" for x in deltas)
        calls=[x.get("tool_calls") for x in deltas if x.get("tool_calls")]; results=[x for x in deltas if x.get("role")=="tool_result"]
        print("   model content :", repr(content[:300])); print("   thinking      :", len(thinking), "chars", repr(thinking[:200]))
        print("   tool calls    :", json.dumps(calls)[:400]); print("   tool results  :", [json.dumps(r)[:200] for r in results])
asyncio.run(main(sys.argv[1]))
