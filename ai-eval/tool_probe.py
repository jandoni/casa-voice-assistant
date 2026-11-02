import json, os, urllib.request, time, sys
AI=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
TOOLS=[{"type":"function","function":{"name":"HassTurnOn","description":"Turns on a device or entity","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Name of a device or entity"},"area":{"type":"string"},"domain":{"type":"array","items":{"type":"string"}}},"required":[]}}},
       {"type":"function","function":{"name":"HassTurnOff","description":"Turns off a device or entity","parameters":{"type":"object","properties":{"name":{"type":"string"},"area":{"type":"string"},"domain":{"type":"array","items":{"type":"string"}}},"required":[]}}},
       {"type":"function","function":{"name":"HassLightSet","description":"Sets the brightness percentage or color of a light","parameters":{"type":"object","properties":{"name":{"type":"string"},"area":{"type":"string"},"brightness":{"type":"integer","minimum":0,"maximum":100},"color":{"type":"string"}},"required":[]}}},
       {"type":"function","function":{"name":"GetLiveContext","description":"Use to get current state of devices when the user asks about them","parameters":{"type":"object","properties":{}}}}]
SYS = """You are the voice assistant of this home. Be brief. After performing an action, confirm it in one short sentence. Plain text only.
An overview of the areas and the devices in this smart home:
- names: Sofa (light.living_room_sofa) area Living Room
- names: Overhead (light.living_room_overhead) area Living Room
- names: TV Light, backlight (light.living_room_backlight) area Living Room
- names: Ceiling, Cooking (light.kitchen_ceiling) area Kitchen
- names: Countertop, island (light.kitchen_island) area Kitchen
- names: Corridor (light.kitchen_corridor) area Kitchen
- names: Dining Table, chandelier (light.dining_room_pendant) area Dining Room
- names: Jon, Jon's light, Jon's desk (light.office_desk) area Office
- names: Reading Lamp (light.office_reading) area Office
- names: Monitor Light (light.monitor_bar) area Office
- names: Pendant (light.bedroom_pendant) area Bedroom
- names: Ceiling (light.bedroom_ceiling) area Bedroom
- names: Thermostat (climate.thermostat) area Living Room
- names: Soundbar (media_player.soundbar) area Living Room
- names: Blind (cover.bedroom_blind) area Bedroom
- names: Movie (scene.movie) ; Dinner (scene.dinner) ; Bedtime (scene.bedtime)
"""
CASES=[("turn on the sofa light", "HassTurnOn", {"name":"Sofa"}),
       ("dim the TV light to 20 percent", "HassLightSet", {"brightness":20}),
       ("it's a bit dark at Jon's desk", "HassLightSet|HassTurnOn", {}),
       ("switch off everything in the kitchen", "HassTurnOff", {"area":"Kitchen"}),
       ("what's the thermostat set to?", "GetLiveContext", {}),
       ("who painted the Mona Lisa?", None, {})]
def run(model):
    print(f"\n### {model}")
    for text, want, args in CASES:
        body={"model":model,"stream":False,"think":False,"tools":TOOLS,"options":{"num_predict":200,"num_ctx":8192},
              "messages":[{"role":"system","content":SYS},{"role":"user","content":text}]}
        t=time.time()
        try:
            d=json.loads(urllib.request.urlopen(urllib.request.Request(AI+"/api/chat",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600).read())
        except Exception as e: print(f"  {text!r}: ERROR {e}"); continue
        m=d["message"]; calls=[(c["function"]["name"],c["function"]["arguments"]) for c in m.get("tool_calls",[]) or []]
        content=m.get("content","").strip().replace("\n"," ")
        thinking = len(m.get("thinking") or "")
        if want is None: ok = not calls and bool(content)
        else: ok = any(n in want.split("|") and all(str(v).lower() in str(a).lower() for v in args.values()) for n,a in calls) if args else any(n in want.split("|") for n,a in calls)
        print(f"  {'OK ' if ok else 'BAD'} {time.time()-t:5.1f}s | {text!r} -> calls={calls} content={content[:70]!r} thinking={thinking}")
for m in sys.argv[1:]: run(m)
