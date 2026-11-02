import json, urllib.request, time, os, sys
HA=os.environ["HA_URL"]; TOK=os.environ["HA_TOKEN"]; H={"Authorization":f"Bearer {TOK}","Content-Type":"application/json"}
agent=sys.argv[1]
def state(e): 
    d=json.loads(urllib.request.urlopen(urllib.request.Request(f"{HA}/api/states/{e}",headers=H),timeout=10).read()); return d["state"], d["attributes"].get("brightness")
def conv(text, cid=None):
    body={"text":text,"agent_id":agent,"language":"en"}
    if cid: body["conversation_id"]=cid
    t=time.time(); d=json.loads(urllib.request.urlopen(urllib.request.Request(f"{HA}/api/conversation/process",data=json.dumps(body).encode(),headers=H),timeout=600).read())
    return time.time()-t, d["response"]["speech"]["plain"]["speech"], d["response"]["response_type"], d.get("conversation_id")
def call(svc, data):
    urllib.request.urlopen(urllib.request.Request(f"{HA}/api/services/{svc}",data=json.dumps(data).encode(),headers=H),timeout=20).read()
# Two lights in a room nobody is using (never test on a room where someone is working)
CEIL = os.environ.get("EVAL_LIGHT_CEILING", "light.bedroom_ceiling")
PEND = os.environ.get("EVAL_LIGHT_PENDANT", "light.bedroom_pendant")
snap={e:state(e) for e in (CEIL,PEND)}
print("agent:", agent, "| bedroom before:", snap)
call("light/turn_off",{"entity_id":[CEIL,PEND]}); time.sleep(1)
tests=[("turn on the bedroom pendant", lambda: state(PEND)[0]=="on"),
       ("set the bedroom pendant to 30 percent", lambda: state(PEND)[0]=="on" and abs((state(PEND)[1] or 0)-77)<=8),
       ("what's the thermostat set to?", None),
       ("which lights are on in the bedroom?", None),
       ("turn off the bedroom pendant", lambda: state(PEND)[0]=="off"),
       ("it's a bit dark in the bedroom", lambda: any(state(e)[0]=="on" for e in (CEIL,PEND))),
       ("turn on the bedroom ceiling light", lambda: state(CEIL)[0]=="on"),
       ("turn off everything in the bedroom", lambda: all(state(e)[0]=="off" for e in (CEIL,PEND))),
       ("who painted the Mona Lisa?", None),
       ("how many lights are in the kitchen?", None)]
tot=0; okc=0
for text, check in tests:
    dt, speech, rtype, _ = conv(text); time.sleep(1.5)
    ok = check() if check else (rtype!="error" and bool(speech))
    tot+=1; okc+=ok
    print(f"  {'OK ' if ok else 'BAD'} {dt:5.1f}s | {text!r} -> [{rtype}] {speech.strip()[:110]!r}")
print(f"score {okc}/{tot}")
call("light/turn_off",{"entity_id":[CEIL,PEND]})
for e,(st,br) in snap.items():
    if st=="on": call("light/turn_on",{"entity_id":e,"brightness":br} if br else {"entity_id":e})
print("bedroom restored")
