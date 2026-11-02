#!/usr/bin/env python3
"""llm_bench.py MODEL [prompt_tokens_target] - long-prompt + generation benchmark against local Ollama."""
import sys, json, time, urllib.request
m = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
filler = ("The living room has a sofa light, an overhead light and a TV backlight. "
          "The kitchen has a ceiling light, an island pendant and a corridor light. "
          "The bedroom has a ceiling light and a pendant over the bed. ") * n
q = filler + "\nQuestion: which rooms have a pendant? Answer in one short sentence."
for run in (1, 2):
    body = json.dumps({"model": m, "prompt": q, "stream": False, "think": False,
                       "options": {"num_ctx": 8192, "num_predict": 40}}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://localhost:11434/api/generate", data=body,
        headers={"Content-Type": "application/json"}), timeout=1800))
    pe, pd = d["prompt_eval_count"], d["prompt_eval_duration"] / 1e9
    ec, ed = d["eval_count"], d["eval_duration"] / 1e9
    print(f"  {m} run{run}: prompt {pe} tok in {pd:.1f}s = {pe/pd:.0f} tok/s | gen {ec} tok {ec/ed:.1f} tok/s | wall {time.time()-t0:.1f}s | {d['response'].strip()[:70]!r}", flush=True)
