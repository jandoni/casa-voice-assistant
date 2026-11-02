"""Casa Live tools (2026-10-16): a small LLM API ("Casa house tools") for realtime voice models (Gemini Live via the
gemini_live custom integration, or any HA conversation agent that takes LLM APIs).

Why our own API instead of HA's built-in "Assist" one: Assist sends every exposed entity + ~20 intent tools with EVERY
request (thousands of tokens, re-billed each turn of a Live session) and bypasses our own logic. These few tools are
short, and `house` runs the request through HA's local agent WITH the device the user is talking to, so everything we
built keeps working: timers ring on that Echo, room-relative commands, recipes, music (casa_play), weather cards,
bedroom routines, multi-commands, lists.

Tools: house · add_chore · add_grocery · whats_due · complete_chore · check_off_grocery.
The chore and grocery tools call services of my `casa_tasks` integration (Donetick chores + Mealie groceries), which is
not in this repo: keep `house` and drop or replace the others.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

_LOGGER = logging.getLogger(__name__)
DOMAIN = "casa_live"
API_ID = "casa"
LOCAL_AGENT = "conversation.home_assistant"   # HA's local agent: all our custom sentences/intents, never an LLM
# light groups / non-lamp "lights" to leave out of the live snapshot (they double-count real lamps)
SNAPSHOT_SKIP_LIGHTS = ("light.all_lights",)
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

API_PROMPT = (
    "You control this home with these tools. "
    "Use `house` for EVERY device or home request: lights, thermostat, blinds, humidifier, scents, timers and alarms, "
    "music, routines (movie, dinner, cooking, bedtime, sleep, good night, good morning, work mode), recipes, weather, "
    "device states. Pass ONE short plain English command per call, phrased like a person would say it to a smart speaker "
    "('turn off the office lights', 'set a timer for 10 minutes', 'play some jazz in the bedroom', 'what's the weather "
    "tomorrow'); several actions = several calls. Use add_chore for chores/reminders/to-dos (with repeats, days, time, "
    "person) and add_grocery for groceries (with quantity). Only say something was done if the tool result says so; if "
    "`house` did not understand, try once more with simpler wording, then tell the user briefly."
)


# (2026-10-19) several commands in one sentence ("turn off the kitchen light and turn on the sofa light"): same split as the
# canonicalizer patch (patches/assist_canonicalizer-1.14.0-casa.patch), every part runs locally at the same time.
_MULTI_SEP_RE = re.compile(r"\s*,?\s+(?:and then|and also|and|then|also|plus)\s+|\s*[,;]\s+(?=(?:please\s+)?(?:turn|switch|set|dim|brighten|open|close|start|stop|lock|unlock|play|pause)\b)", re.I)
_MULTI_VERB_RE = re.compile(r"^\W*(?:(?:please|can you|could you|and)\s+)*((?:turn|switch)\s+(?:on|off)|turn|switch|set|dim|brighten|open|close|start|stop|lock|unlock|put|play|pause|resume|skip)\b", re.I)
_MULTI_DEVICE_RE = re.compile(r"\b(lights?|lamps?|monitor|desk|sofa|overhead|countertop|counter|kitchen|dining|hallway|entry|corridor|ceiling|pendant|"
                              r"mirror|vanity|closets?|wardrobe|walk-in|hatch|thermostat|heat(?:ing)?|temperature|ac|fan|scent|diffuser|blinds?|tv|music|"
                              r"humidifier|bedroom|office|living|degrees)\b", re.I)


def split_multi(text: str) -> list[str]:
    if not text or not _MULTI_VERB_RE.match(text):
        return [text]
    raw = [p.strip(" ,.;") for p in _MULTI_SEP_RE.split(text) if p and p.strip(" ,.;")]
    if len(raw) < 2:
        return [text]
    out, verb = [], None
    for p in raw:
        m = _MULTI_VERB_RE.match(p)
        if m:
            verb = m.group(1)
            out.append(p)
        elif verb and _MULTI_DEVICE_RE.search(p):
            out.append(f"{verb} {p}")   # "turn off the kitchen and the sofa" -> "turn off the sofa"
        else:
            return [text]
    return out


def _speech(result: conversation.ConversationResult) -> tuple[str, bool]:
    r = result.response
    speech = ((r.speech or {}).get("plain") or {}).get("speech", "") if r else ""
    ok = bool(r)
    try:
        from homeassistant.helpers.intent import IntentResponseType
        ok = r.response_type != IntentResponseType.ERROR
    except Exception:  # noqa: BLE001
        pass
    return speech, ok


class HouseTool(llm.Tool):
    name = "house"
    description = ("Run one home command or question through the house's own assistant (lights, climate, blinds, timers, "
                   "alarms, music, routines, recipes, weather, states). Returns what the house replied.")
    parameters = vol.Schema({vol.Required("command", description="One short spoken-style English command"): str})

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> JsonObjectType:
        text = str(tool_input.tool_args.get("command", "")).strip()
        if not text:
            return {"ok": False, "reply": "empty command"}
        dev = getattr(llm_context, "device_id", None)

        async def run(t: str) -> tuple[str, bool]:
            # a NEW conversation id: reusing the live session's id would deadlock on HA's per-conversation chat log
            return _speech(await conversation.async_converse(
                hass, t, None, llm_context.context, language="en", agent_id=LOCAL_AGENT, device_id=dev))

        parts = split_multi(text)
        if len(parts) > 1:
            res = await asyncio.gather(*(run(p) for p in parts), return_exceptions=True)
            said, ok = [], True
            for p, r in zip(parts, res):
                if isinstance(r, BaseException) or not r[1]:
                    ok = False
                    said.append(f"I couldn't do '{p}'")
                elif r[0].strip():
                    said.append(r[0].strip().rstrip("."))
            speech = (". ".join(said) + ".") if said else "Done."
        else:
            speech, ok = await run(text)
        _LOGGER.debug("casa_live house %r -> ok=%s %r", text, ok, speech)
        hass.bus.async_fire("casa_live_tool", {"tool": "house", "input": text, "ok": ok, "reply": speech[:200]})
        return {"ok": ok, "reply": speech or ("done" if ok else "not understood")}


class AddChoreTool(llm.Tool):
    name = "add_chore"
    description = "Add a household chore / reminder / to-do to the shared list (Donetick), optionally repeating, at a time, for a person."
    parameters = vol.Schema({
        vol.Required("name", description="What to do, short, e.g. 'Water the plants'"): str,
        vol.Optional("repeat", description="none, daily, weekly, monthly, yearly or interval"): vol.In(["none", "daily", "weekly", "monthly", "yearly", "interval"]),
        vol.Optional("days", description="For weekly: weekdays in lowercase English, e.g. ['sunday']"): [vol.In(DAYS)],
        vol.Optional("every", description="For interval: every N units, e.g. 2"): int,
        vol.Optional("unit", description="For interval: days, weeks or months"): vol.In(["days", "weeks", "months"]),
        vol.Optional("due", description="First due date/time in house local time, 'YYYY-MM-DDTHH:MM'"): str,
        vol.Optional("who", description="a household member's first name, lowercase; leave out for anyone"): str,
        vol.Optional("remind", description="Send a phone reminder when due (default true)"): bool,
    })

    async def async_call(self, hass, tool_input, llm_context):
        d = dict(tool_input.tool_args)
        if d.get("due") and len(d["due"]) == 16:
            d["due"] += ":00"
        d["device_id"] = getattr(llm_context, "device_id", None)
        r = await hass.services.async_call("casa_tasks", "create", d, blocking=True, return_response=True)
        hass.bus.async_fire("casa_live_tool", {"tool": "add_chore", "input": str(d)[:200], "ok": True})
        return {"ok": True, "reply": (r or {}).get("speech", "added")}


class AddGroceryTool(llm.Tool):
    name = "add_grocery"
    description = ("Add an item to the shared grocery list (Mealie). If the user gave no amount, add it without quantity "
                   "and ask 'How many?'; when they answer, call this again with the same name and the quantity (it updates "
                   "the item, no duplicate). The list is shown on the Echo's screen.")
    parameters = vol.Schema({
        vol.Required("name", description="Just the item, lowercase, no verbs or amounts: 'buying eggs' -> 'eggs'"): str,
        vol.Optional("quantity", description="How many, e.g. 12 for 'a dozen'"): vol.Coerce(float),
        vol.Optional("note", description="Brand / size / detail"): str,
    })

    async def async_call(self, hass, tool_input, llm_context):
        d = dict(tool_input.tool_args, device_id=getattr(llm_context, "device_id", None), merge=True)
        r = await hass.services.async_call("casa_tasks", "grocery_add", d, blocking=True, return_response=True)
        hass.bus.async_fire("casa_live_tool", {"tool": "add_grocery", "input": str(tool_input.tool_args)[:200], "ok": True})
        return {"ok": True, "reply": (r or {}).get("speech", "added")}


class WhatsDueTool(llm.Tool):
    name = "whats_due"
    description = "Read the chores that are due (today, this week or all) and the grocery list."
    parameters = vol.Schema({vol.Optional("scope", description="today, week or all"): vol.In(["today", "week", "all"])})

    async def async_call(self, hass, tool_input, llm_context):
        r = await hass.services.async_call("casa_tasks", "due", {"scope": tool_input.tool_args.get("scope", "today")},
                                           blocking=True, return_response=True)
        st = hass.states.get("sensor.casa_lists")
        groc = [(f"{g['qty']} " if g.get("qty") else "") + g["name"] for g in (st.attributes.get("groceries", []) if st else [])]
        return {"chores": (r or {}).get("speech", ""), "groceries": groc}


class CompleteChoreTool(llm.Tool):
    name = "complete_chore"
    description = "Mark a chore on the shared list as done."
    parameters = vol.Schema({vol.Required("name", description="The chore, roughly as it is called"): str})

    async def async_call(self, hass, tool_input, llm_context):
        r = await hass.services.async_call("casa_tasks", "complete", {"text": tool_input.tool_args["name"]}, blocking=True, return_response=True)
        return {"ok": True, "reply": (r or {}).get("speech", "")}


class CheckOffGroceryTool(llm.Tool):
    name = "check_off_grocery"
    description = "Remove / tick off an item from the grocery list (bought or not needed)."
    parameters = vol.Schema({vol.Required("name", description="The item"): str})

    async def async_call(self, hass, tool_input, llm_context):
        st = hass.states.get("sensor.casa_lists")
        items = st.attributes.get("groceries", []) if st else []
        want = re.sub(r"^(the|some)\s", "", tool_input.tool_args["name"].lower().strip())
        names = [g["name"].lower() for g in items]
        best = difflib.get_close_matches(want, names, n=1, cutoff=0.5)
        if not best:
            return {"ok": False, "reply": f"{want} is not on the grocery list"}
        g = items[names.index(best[0])]
        await hass.services.async_call("casa_tasks", "grocery_update", {"id": g["uid"], "done": True}, blocking=True)
        return {"ok": True, "reply": f"ticked off {g['name']}"}


class CasaAPI(llm.API):
    def __init__(self, hass: HomeAssistant) -> None:
        super().__init__(hass=hass, id=API_ID, name="Casa house tools")

    def _snapshot(self) -> str:
        """A few live facts (~150 tokens) so 'how warm is it inside / is anything on' needs no tool call."""
        h = self.hass
        bits = []
        c = h.states.get("climate.thermostat")
        if c:
            a = c.attributes
            bits.append(f"inside {a.get('current_temperature')}°C, thermostat {c.state}"
                        + (f" set to {a.get('temperature')}°C" if a.get("temperature") is not None else ""))
        on = [s.name for s in h.states.async_all("light") if s.state == "on" and s.entity_id not in SNAPSHOT_SKIP_LIGHTS]
        bits.append("lights on: " + (", ".join(on[:12]) if on else "none"))
        playing = [f"{s.name}: {s.attributes.get('media_title')}" for s in h.states.async_all("media_player")
                   if s.state == "playing" and s.attributes.get("media_title")]
        if playing:
            bits.append("playing: " + "; ".join(dict.fromkeys(playing[:4])))
        t = h.states.get("sensor.casa_timers")
        if t and t.state not in ("0", "unknown", "unavailable"):
            bits.append(f"{t.state} timer(s) running")
        return " Live house state: " + "; ".join(bits) + "."

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        tools = [HouseTool(), AddChoreTool(), AddGroceryTool(), WhatsDueTool(), CompleteChoreTool(), CheckOffGroceryTool()]
        from homeassistant.helpers import area_registry as ar, device_registry as dr
        from homeassistant.util import dt as dt_util
        now = dt_util.now()
        where = ""
        dev_id = getattr(llm_context, "device_id", None)
        if dev_id:
            dev = dr.async_get(self.hass).async_get(dev_id)
            area = dev and dev.area_id and ar.async_get(self.hass).async_get_area(dev.area_id)
            if area:
                where = f" The user is talking to you from the {area.name}; 'the lights' etc. mean that room unless they say otherwise."
        prompt = (API_PROMPT + f" Now (house time): {now.strftime('%A %B %-d %Y, %-I:%M %p')}." + where + self._snapshot())
        return llm.APIInstance(api=self, api_prompt=prompt, llm_context=llm_context, tools=tools)


async def async_setup(hass: HomeAssistant, config) -> bool:
    # Start speaking the reply on its FIRST words: HA 2026.9 only streams TTS once a reply has > 60 characters (else it
    # waits for the whole answer), which made every short Gemini Live answer wait until fully generated (~1.5-3 s).
    # Also fine for the Casa AI pipeline (Kokoro streams sentence by sentence).
    try:
        from homeassistant.components.assist_pipeline import pipeline as ap
        ap.STREAM_RESPONSE_CHARS = 0
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning("casa_live: could not lower the TTS streaming threshold: %s", err)
    api = CasaAPI(hass)
    llm.async_register_api(hass, api)

    # ---- Gemini Live cost tracking (2026-10-19): token counts from gemini_live (CASA PATCH USAGE_HOOK) ----
    # $ per 1M tokens, Gemini 3.8 Live list prices (check ai.google.dev/gemini-api/docs/pricing if they change)
    PRICE = {("in", "AUDIO"): 3.0, ("in", "TEXT"): 0.5, ("out", "AUDIO"): 12.0, ("out", "TEXT"): 2.0}
    from homeassistant.helpers.storage import Store
    from homeassistant.util import dt as dt_util
    store = Store(hass, 1, "casa_live_cost")
    st = (await store.async_load()) or {}
    cost = {"month": st.get("month", ""), "day": st.get("day", ""), "usd_month": st.get("usd_month", 0.0),
            "usd_day": st.get("usd_day", 0.0), "turns_month": st.get("turns_month", 0), "tokens": st.get("tokens", {})}

    def publish() -> None:
        hass.states.async_set("sensor.gemini_live_cost", round(cost["usd_month"], 4), {
            "friendly_name": "Gemini Live cost this month", "unit_of_measurement": "USD", "icon": "mdi:cash",
            "today_usd": round(cost["usd_day"], 4), "turns_this_month": cost["turns_month"], "tokens_this_month": cost["tokens"],
            "prices_per_million": {f"{d} {m}": v for (d, m), v in PRICE.items()}})

    @callback
    def on_usage(um: dict) -> None:
        now = dt_util.now()
        if cost["month"] != now.strftime("%Y-%m"):
            cost.update(month=now.strftime("%Y-%m"), usd_month=0.0, turns_month=0, tokens={})
        if cost["day"] != now.strftime("%Y-%m-%d"):
            cost.update(day=now.strftime("%Y-%m-%d"), usd_day=0.0)
        usd = 0.0
        for d, key, total_key in (("in", "prompt_tokens_details", "prompt_token_count"), ("out", "response_tokens_details", "response_token_count")):
            details = um.get(key) or []
            seen = 0
            for m in details:
                mod = str(m.get("modality", "TEXT")).split(".")[-1].upper()
                n = int(m.get("token_count") or 0); seen += n
                usd += n * PRICE.get((d, mod), PRICE[(d, "TEXT")]) / 1e6
                cost["tokens"][f"{d}_{mod.lower()}"] = cost["tokens"].get(f"{d}_{mod.lower()}", 0) + n
            rest = int(um.get(total_key) or 0) - seen   # tokens without a modality breakdown: count as text
            if rest > 0:
                usd += rest * PRICE[(d, "TEXT")] / 1e6
                cost["tokens"][f"{d}_other"] = cost["tokens"].get(f"{d}_other", 0) + rest
        thoughts = int(um.get("thoughts_token_count") or 0)
        if thoughts:
            usd += thoughts * PRICE[("out", "TEXT")] / 1e6
        cost["usd_month"] += usd; cost["usd_day"] += usd; cost["turns_month"] += 1
        _LOGGER.info("gemini usage: %s -> $%.5f", um, usd)
        hass.bus.async_fire("casa_gemini_usage", {"usd": round(usd, 6), **{k: v for k, v in um.items() if isinstance(v, (int, float))}})
        publish()
        store.async_delay_save(lambda: dict(cost), 10)

    try:
        from custom_components.gemini_live import gemini as _gem
        _gem.USAGE_HOOK = lambda um: hass.loop.call_soon_threadsafe(on_usage, um)
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning("casa_live: gemini_live usage hook not set: %s", err)
    publish()

    # casa_live.debug {tool?, args?, device_id?}: the prompt Gemini would get, or one tool run for real (tests)
    from homeassistant.core import ServiceCall, SupportsResponse

    async def debug(call: ServiceCall):
        ctx = llm.LLMContext(platform=DOMAIN, context=call.context, language="en", assistant="conversation",
                             device_id=call.data.get("device_id"))
        inst = await api.async_get_api_instance(ctx)
        if not call.data.get("tool"):
            return {"prompt": inst.api_prompt, "tools": [t.name for t in inst.tools]}
        res = await inst.async_call_tool(llm.ToolInput(tool_name=call.data["tool"], tool_args=call.data.get("args") or {}))
        return {"result": res}

    hass.services.async_register(DOMAIN, "debug", debug, supports_response=SupportsResponse.ONLY)
    return True
