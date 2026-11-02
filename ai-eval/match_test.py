import json, subprocess, sys
sents=["I am going to bed","we are leaving","dinner is ready","I am working","let us watch a movie","movie time","I am cooking","the air is very dry","it is a bit cold in here","make it warmer","it is too loud","turn the music down a bit","turn it up in the bedroom","make it cozy","cozy lights in the bedroom","I cannot hear the tv","it is too hot","rise and shine","it's too bright in the office","I'm off to bed","time to eat"]
out=subprocess.run(["python3", sys.argv[1], json.dumps({"type":"conversation/agent/homeassistant/debug","sentences":sents})],capture_output=True,text=True).stdout
r=json.loads(out)
for s,res in zip(sents, r["results"]):
    if not res: print(f"  X {s!r}: no match"); continue
    slots=", ".join(f"{k}={v.get('value') if isinstance(v,dict) else v}" for k,v in (res.get('slots') or {}).items())
    print(f"  ok {s!r:34s} -> {res.get('intent',{}).get('name')} [{slots}]")
