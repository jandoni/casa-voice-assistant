"""The Casa Brain conversation entity."""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import DOMAIN

_LOGGER = logging.getLogger(__name__)

HOUSE_RE = re.compile(
    r"\b(light|lights|lamp|lamps|scent|diffuser|monitor|blind|blinds|humidifier|thermostat|temperature inside|"
    r"speaker|soundbar|apple tv|tv|office|bedroom|kitchen|living room|dining|hallway|entry|foyer|closet|wardrobe|"
    r"mirror|vanity|pendant|ceiling|sofa|overhead|counter|countertop|corridor|cooking|desk|"
    r"everything|all the lights|scene|movie mode|dinner mode|bedtime|wake up|work mode|good night|timer|alarm|"
    r"humidity|dim|brighten|too (bright|dark|cold|hot|loud|quiet|warm)|cozy|turn (on|off)|switch (on|off)|"
    r"open|close|volume|pause|resume|next song|skip|play (something|music|the)|sleep sound|white noise|rain sound|ping)\b",
    re.IGNORECASE,
)


def speechify(text: str) -> str:
    """Markdown / links -> plain speech, at most two sentences."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = " ".join(parts[:2]).strip()
    return out if len(out) <= 320 else out[:317].rstrip() + "…"


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (t or "").lower()).split())


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    conf = hass.data.get(DOMAIN, {}).get("yaml") or {}
    if not conf.get("chat_url"):
        _LOGGER.error("casa_brain: add the casa_brain: block to configuration.yaml")
        return
    ent = CasaBrainEntity(entry, conf)
    async_add_entities([ent])

    # Head start (2026-09-24): casa-stt announces a question as soon as its streaming pass has it
    # (`casa_stream_question`), ~1.1 s before the accurate transcript reaches us. Start answering it now; the real
    # request uses this answer when its text is (nearly) the same, otherwise it is cancelled.
    @callback
    def _on_question(event) -> None:
        text = (event.data.get("text") or "").strip()
        if not text or HOUSE_RE.search(text):
            return
        for key, (task, _t, _txt) in list(ent.spec.items()):   # a new question: older guesses are useless
            task.cancel(); ent.spec.pop(key, None)
        task = hass.async_create_task(ent.ask_chat(text), "casa_brain head start")
        ent.spec[_norm(text)] = (task, time.monotonic(), text)
        hass.loop.call_later(25, lambda k=_norm(text): ent.drop_spec(k))

    entry.async_on_unload(hass.bus.async_listen("casa_stream_question", _on_question))


class CasaBrainEntity(conversation.ConversationEntity):
    _attr_has_entity_name = False
    _attr_name = "Casa Brain"
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry, conf: dict) -> None:
        self.conf = conf
        self.spec: dict = {}          # normalised text -> (task, started, original text) - see _on_question
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = None

    @property
    def supported_languages(self) -> list[str] | str:
        return MATCH_ALL

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        text = (user_input.text or "").strip()
        if not text:
            return self._say(user_input, "Sorry, I didn't catch that. Could you say it again?", True)
        if HOUSE_RE.search(text):
            # NOTE: a derived conversation id - reusing the caller's id deadlocks on HA's per-conversation chat log
            # (the canonicalizer that called us still holds it) and ends in "Unable to get response" after 60 s.
            cid = f"{user_input.conversation_id}-house" if user_input.conversation_id else None
            result = await conversation.async_converse(
                self.hass, text, cid, user_input.context,
                language=user_input.language, agent_id=self.conf["house_agent"], device_id=user_input.device_id,
            )
            return conversation.ConversationResult(response=result.response, conversation_id=user_input.conversation_id,
                                                   continue_conversation=result.continue_conversation)
        answer = ""
        spec = self.take_spec(text)
        if spec is not None:
            try:
                answer = await spec
                _LOGGER.debug("casa_brain: used the head-start answer for %r", text)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                answer = ""
        if not answer:
            answer = await self.ask_chat(text)
        if not answer:
            return self._say(user_input, "Sorry, I couldn't find that right now.", False)
        return self._say(user_input, speechify(answer), False)

    def drop_spec(self, key: str) -> None:
        item = self.spec.pop(key, None)
        if item and not item[0].done():
            item[0].cancel()

    def take_spec(self, text: str):
        """The head-start task for this text if its question was (nearly) the same; cancels any other."""
        key = _norm(text)
        best, best_r = None, 0.0
        for k in list(self.spec):
            r = 1.0 if k == key else difflib.SequenceMatcher(None, k, key).ratio()
            if r > best_r:
                best, best_r = k, r
        chosen = None
        if best is not None and best_r >= 0.93 and time.monotonic() - self.spec[best][1] < 25:
            chosen = self.spec.pop(best)[0]
        for k in list(self.spec):
            self.drop_spec(k)
        return chosen

    async def ask_chat(self, text: str) -> str:
        answer = ""
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    self.conf["chat_url"],
                    json={"model": self.conf["chat_model"], "stream": False,
                          "messages": [{"role": "user", "content": "[voice] " + text}]},
                    headers={"Authorization": f"Bearer {self.conf['chat_api_key']}"},
                    timeout=aiohttp.ClientTimeout(total=self.conf.get("chat_timeout", 25)),
                ) as r:
                    j = await r.json(content_type=None)
            answer = (((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning("chat brain failed: %s", e)
        return answer

    def _say(self, user_input, text, cont):
        resp = intent.IntentResponse(language=user_input.language)
        resp.async_set_speech(text)
        return conversation.ConversationResult(response=resp, conversation_id=user_input.conversation_id,
                                               continue_conversation=cont)
