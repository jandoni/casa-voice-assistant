#!/usr/bin/env python3
"""Push claude/voice_prompt.txt into HA's Claude conversation agent (Anthropic integration subentry) and keep
the rest of its settings: Haiku 4.5, Assist control, automatic prompt caching, 400 max tokens, no thinking, Anthropic
web search off (our free script.casa_web_search is used instead). Run with config.env exported."""
import json, os, pathlib, urllib.request

H, T = os.environ["HA_URL"], os.environ["HA_TOKEN"]
# the Anthropic integration's config entry id and the conversation subentry id (Settings -> Devices & services)
ENTRY, SUB = os.environ["CLAUDE_ENTRY_ID"], os.environ["CLAUDE_SUBENTRY_ID"]
PROMPT = (pathlib.Path(__file__).parent / "voice_prompt.txt").read_text().strip()


def post(path, data):
    req = urllib.request.Request(H + path, data=json.dumps(data).encode(),
                                 headers={"Authorization": "Bearer " + T, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


r = post("/api/config/config_entries/subentries/flow", {"handler": [ENTRY, "conversation"], "source": "reconfigure", "subentry_id": SUB})
fid = r["flow_id"]
post(f"/api/config/config_entries/subentries/flow/{fid}", {"prompt": PROMPT, "llm_hass_api": ["assist"], "recommended": False})
post(f"/api/config/config_entries/subentries/flow/{fid}", {"chat_model": "claude-haiku-4-5", "prompt_caching": "automatic"})
r = post(f"/api/config/config_entries/subentries/flow/{fid}", {"max_tokens": 400, "thinking_budget": 0, "code_execution": False,
                                                              "web_search": False, "web_search_max_uses": 1, "user_location": True,
                                                              "web_fetch": False, "web_fetch_max_uses": 1})
print(r.get("type"), r.get("reason"))
