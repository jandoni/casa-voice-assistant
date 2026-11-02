"""Casa smart end-of-speech (2026-09-26).

HA ends a voice command after a FIXED silence (VoiceCommandSegmenter.silence_seconds: 0.7 s 'default', 1.25 s
'relaxed'). People pause mid-sentence ("So I was cooking bread … and it came out really hard"), so long questions were
cut at the first pause, while short commands want the short wait. Alexa / OpenAI's semantic VAD decide from the WORDS
so far. Here: casa-stt already streams its partial transcript as `casa_live_text` events (~4/s); while a command is in
progress the segmenter's silence budget is set from that text:
  - sounds unfinished (ends on "and/the/to/because/um…", or a story opener with no question/command yet) -> 2.2 s
  - otherwise (a command, a finished question, anything complete) -> 0.7 s
The segmenter is patched in place (class method wrapper), so every pipeline / satellite gets it; HA's own
'finished_speaking_detection' select stays at 'default'.
"""
from __future__ import annotations

import logging
import re
import time

from homeassistant.core import Event, HomeAssistant, callback

_LOGGER = logging.getLogger(__name__)
DOMAIN = "casa_endpoint"

SHORT_S = 0.7
LONG_S = 2.2
MID_S = 1.2    # (2026-09-29) a plain statement that is not a story opener: often a misheard command, don't wait 2.2 s
LIVE_MAX_AGE_S = 2.5
LAB_S = 4.0
# (2026-10-24, Jon) Gemini Live turns: Gemini's own end-of-speech decides (its automatic activity detection). When it
# starts answering, gemini_live/stt.py fires `casa_gemini_replied` and the turn ends right there (2026-11-01: before
# that the Echo's mic stream stayed open until this safety net, ~10 s "listening" after every question).
GEMINI_S = 10.0
# (2026-11-01) ...but when Gemini has transcribed NO words in this turn, 3 s of silence ends it: there is nothing to wait
# for (a whisper / noise after the wake word kept the bedroom Echo "listening" ~19 s, then an error sound)
GEMINI_NOWORDS_S = 3.0
MAX_S = 60.0   # one turn (a conversation has as many turns as you like); casa-stt splits >28 s for whisper

# ends on a word that can't end a sentence -> they're mid-thought
STRONG_TAIL_RE = re.compile(
    r"\b(and|but|or|so|because|cause|the|a|an|to|of|with|for|in|on|at|from|about|into|my|your|our|his|her|their|"
    r"if|than|like|um|uh|erm|hmm|kind of|sort of)\W*$", re.I)
# ends on a word that usually continues - unless a question is already there ("what could it be")
WEAK_TAIL_RE = re.compile(
    r"\b(is|was|were|are|am|be|been|that|which|who|when|while|then|i|we|you|it|this|some|really|very|just|also|maybe|"
    r"what|how|why|where|can|could|would|should|do|does|did|have|had|has)\W*$", re.I)
OPENER_RE = re.compile(r"^\W*(so|well|okay so|ok so|hey|um|uh|i was|i am|i'm|i have|i've|i had|i did|i made|i went|i noticed|"
                       r"we were|we have|we had|my|yesterday|today i|this morning|last night|you know|listen|so i|so we|so my)\b", re.I)
QUESTION_RE = re.compile(r"\b(what|why|how|when|where|who|which|can you|could you|would you|should i|do you|is it|are there|"
                         r"does it|any idea|what's|whats|tell me|help me|explain|recommend)\b", re.I)
# (2026-09-28) "Hey Coco … stop / shut up / never mind": one short word is often too short for HA's VAD to count as
# speech, so the segmenter never saw a command and waited for its timeout (11 s measured). When the live transcript of
# THIS turn is just a stop phrase and has been stable for STOP_STABLE_S, the turn ends right away.
STOP_RE = re.compile(r"^\W*(?:(?:okay|ok|oh|no|hey)[\s,]+)?(stop|stop it|stop that|stop listening|stop talking|shut up|shush|hush|quiet|"
                     r"be quiet|silence|enough|cancel|cancel that|never ?mind|nothing|forget it|forget about it|go away|"
                     r"that'?s all|that'?s it|i'?m done|no thanks|no thank you|thanks|thank you)(?:\s+(?:please|thanks|thank you))?\W*$", re.I)
STOP_STABLE_S = 0.3
# (2026-09-28) nobody speaks after the wake (a false wake, or nothing to say): stop listening after NO_SPEECH_S instead
# of the per-turn limit (the 60 s MAX_S applies once speech has started)
NO_SPEECH_S = 6.0
# (2026-09-28) far-field (living room, from the sofa): the Echo's own wake chime is recorded at the start of every turn
# (~-15 dBFS, far louder than a voice across the room). HA's VAD took the chime for the command, then the quiet voice for
# silence, and closed the turn ~2 s after the wake - before or while people spoke. Now the first CHIME_S never counts as
# speech, and until casa-stt has heard real WORDS a turn is never closed after a short pause (only after NO_SPEECH_S).
CHIME_S = 0.9
NO_WORDS_MAX_S = 5.0
COMMAND_RE = re.compile(r"^\W*(please\s+)?(turn|switch|set|dim|brighten|open|close|play|pause|stop|resume|skip|next|start|"
                        r"cancel|add|remind|lights?|make it|put)\b", re.I)


def unfinished(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if STRONG_TAIL_RE.search(t):
        return True
    if WEAK_TAIL_RE.search(t) and not (QUESTION_RE.search(t) and len(t.split()) > 4):
        return True
    if QUESTION_RE.search(t) or COMMAND_RE.match(t) or t.endswith("?"):
        return False
    # a story / context opener, or any statement of 3+ words with no question or command in it yet ("I overcooked the
    # meat", "the bread came out hard"): usually the set-up before the actual question (voice lab 2026-09-26)
    if OPENER_RE.match(t):
        return True
    return "mid" if len(re.findall(r"[A-Za-z']+", t)) >= 3 else False


async def async_setup(hass: HomeAssistant, config) -> bool:
    from homeassistant.components.assist_pipeline import vad

    live = {"text": "", "at": 0.0, "changed": 0.0, "gemini_at": 0.0, "gemini_reply": 0.0}

    @callback
    def _on_live(event: Event) -> None:
        txt = str(event.data.get("text") or "")
        now = time.monotonic()
        if txt != live["text"]:
            live["changed"] = now
        live["text"] = txt
        live["at"] = now
        if event.data.get("engine") == "gemini":
            live["gemini_at"] = now

    hass.bus.async_listen("casa_live_text", _on_live)

    @callback
    def _on_reply(event: Event) -> None:
        live["gemini_reply"] = time.monotonic()

    hass.bus.async_listen("casa_gemini_replied", _on_reply)

    seg = vad.VoiceCommandSegmenter
    if getattr(seg, "_casa_patched", False):
        return True
    orig = seg.process

    def process(self, chunk_seconds, speech_probability):
        try:
            now = time.monotonic()
            if not hasattr(self, "_casa_t0"):
                self._casa_t0 = now
            # Gemini started answering THIS turn: its own end-of-speech fired -> close the mic
            if live["gemini_reply"] > self._casa_t0:
                _LOGGER.debug("casa_endpoint: Gemini answered -> end of turn")
                return False
            # a stop phrase heard in THIS turn, stable for a moment -> done now (no VAD speech / silence wait needed)
            if (live["changed"] > self._casa_t0 and now - live["changed"] >= STOP_STABLE_S
                    and STOP_RE.match(live["text"]) and not hass.states.is_state("input_boolean.casa_voice_lab", "on")):
                _LOGGER.debug("casa_endpoint: stop phrase %r -> end of turn", live["text"])
                return False
            if (not self.in_command and now - self._casa_t0 > NO_SPEECH_S and live["changed"] <= self._casa_t0
                    and not hass.states.is_state("input_boolean.casa_voice_lab", "on")):
                _LOGGER.debug("casa_endpoint: no speech %.0f s after the wake -> end of turn", NO_SPEECH_S)
                return False
            if now - self._casa_t0 < CHIME_S:
                speech_probability = 0.0 if speech_probability is not None else speech_probability
            words_heard = live["changed"] > self._casa_t0 and bool(re.search(r"[A-Za-z]{2}", live["text"]))
            # HA stops any command after 15 s; a real question with context can be longer
            if self.timeout_seconds < MAX_S:
                self._timeout_seconds_left += MAX_S - self.timeout_seconds
                self.timeout_seconds = MAX_S
            if self.in_command:
                fresh = time.monotonic() - live["at"] < LIVE_MAX_AGE_S
                u = unfinished(live["text"]) if fresh else False
                want = MID_S if u == "mid" else LONG_S if u else SHORT_S
                if not words_heard and now - self._casa_t0 < NO_WORDS_MAX_S:
                    want = LONG_S   # the VAD heard something but no words yet: a quiet / distant voice - keep listening
                if now - live["gemini_at"] < 30 and live["gemini_at"] >= self._casa_t0 - 2:
                    want = GEMINI_S if words_heard else GEMINI_NOWORDS_S   # Gemini decides; these are only safety nets
                if hass.states.is_state("input_boolean.casa_voice_lab", "on"):
                    want = LAB_S   # voice lab: keep recording through the test clip (the blip alone counts as "speech")
                if want != self.silence_seconds:
                    self._silence_seconds_left += want - self.silence_seconds
                    if want == LONG_S:
                        _LOGGER.debug("casa_endpoint: waiting longer, sounds unfinished: %r", live["text"][-60:])
                    self.silence_seconds = want
        except Exception:  # noqa: BLE001 - never break the pipeline
            pass
        return orig(self, chunk_seconds, speech_probability)

    seg.process = process
    seg._casa_patched = True
    _LOGGER.info("casa_endpoint: smart end-of-speech active (%.1f s / %.1f s when unfinished)", SHORT_S, LONG_S)
    return True
