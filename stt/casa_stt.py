#!/usr/bin/env python3
"""
casa-stt - two-stage Wyoming speech-to-text server for the house (runs on the voice host).

Stage 1 (while you talk): Kroko streaming Zipformer (sherpa-onnx OnlineRecognizer, modified beam search,
         hotword biasing towards device / room / playlist / people names). ~0.2 s after end of speech.
Stage 2 (only when needed): if the streaming text is NOT a known Home Assistant command (checked with the
         HA sentence matcher, ~50 ms), the buffered audio is re-transcribed by the accurate engine: whisper.cpp
         medium on the iGPU (default, ~1.0-1.5 s, best with accents), or Parakeet TDT 0.6B v3 on the CPU (sherpa-onnx
         OfflineRecognizer, ~0.6 s) with --offline-engine parakeet.

Env / args: see `Config` below. Hotwords file = one phrase per line (plain text, optional " :score").
Wyoming protocol on tcp://0.0.0.0:10310 by default. Logs one line per utterance with timings.
"""
import argparse, asyncio, json, logging, os, re, time
from typing import List, Optional

import numpy as np
import sherpa_onnx
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("casa-stt")
RATE = 16000


class Config:
    def __init__(self, a):
        self.models = a.models
        self.stream_dir = os.path.join(a.models, a.stream_model)
        self.offline_dir = os.path.join(a.models, a.offline_model)
        self.hotwords = a.hotwords
        self.hotwords_score = a.hotwords_score
        self.beam = a.beam
        self.stream_threads = a.stream_threads
        self.offline_threads = a.offline_threads
        self.tail_silence = a.tail_silence
        self.ha_url = a.ha_url.rstrip("/")
        self.ha_token = a.ha_token
        self.always_offline = a.always_offline
        self.uri = a.uri
        self.record_dir = a.record_dir
        self.record_keep = a.record_keep
        self.offline_engine = a.offline_engine
        self.whisper_fast_url = a.whisper_fast_url
        self.whisper_url = a.whisper_url
        self.whisper_timeout = a.whisper_timeout


# ---------------------------------------------------------------- hotwords
def build_bpe_vocab(tokens_path: str, out_path: str) -> int:
    """sherpa-onnx encodes hotwords with its own sentencepiece port, which only needs a `piece<TAB>score` vocab.
    The Kroko tarball ships no bpe.vocab, so we derive one from tokens.txt (longer pieces score higher, so the
    segmentation prefers the fewest pieces - the same path the model most often emits)."""
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for line in open(tokens_path, encoding="utf-8"):
            tok = line.split()[0]
            if tok.startswith("<"):
                continue
            body = tok.lstrip("\u2581")
            f.write(f"{tok}\t{-2.0 + 0.3 * min(len(body), 5):.2f}\n")
            n += 1
    return n


def build_hotwords_file(phrases_path: str, out_path: str) -> int:
    """Plain phrases -> the hotwords file (both the given case and lower case, optional ' :score')."""
    lines, seen = [], set()
    for raw in open(phrases_path, encoding="utf-8"):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=>" in raw:
            continue
        phrase, score = raw, ""
        if " :" in raw:
            phrase, score = raw.rsplit(" :", 1)
            score = " :" + score.strip()
        phrase = re.sub(r"[^\w' ]+", " ", phrase).strip()
        for v in (phrase, phrase.lower()):
            if v and v not in seen:
                seen.add(v)
                lines.append(v + score)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------- text clean-up
UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
UNITS["a"] = 1  # "a hundred" -> 100 (only counted when followed by hundred/thousand, see below)
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_digits(text: str) -> str:
    """'two thousand and twenty six' -> '2026', 'seventy degrees' -> '70 degrees' (single 'one'/'a' left alone)."""
    toks = text.replace("-", " ").split()
    out, i = [], 0
    while i < len(toks):
        j, total, cur, seen, n = i, 0, 0, False, 0
        while j < len(toks):
            t = toks[j].lower().strip(".,?!")
            if t == "a":
                if j + 1 < len(toks) and toks[j + 1].lower() in ("hundred", "thousand"):
                    cur += 1; seen = True; n += 1
                else:
                    break
            elif t in UNITS: cur += UNITS[t]; seen = True; n += 1
            elif t in TENS: cur += TENS[t]; seen = True; n += 1
            elif t == "hundred" and seen: cur *= 100; n += 1
            elif t == "thousand" and seen: total += cur * 1000; cur = 0; n += 1
            elif t == "and" and seen and j + 1 < len(toks) and toks[j + 1].lower() in UNITS | TENS.keys(): pass
            else: break
            j += 1
        if seen and not (n == 1 and toks[i].lower().strip(".,?!") == "one"):
            tail = toks[j - 1][len(toks[j - 1].rstrip(".,?!")):]
            out.append(str(total + cur) + tail); i = j
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)


def load_repairs(path: str):
    """Lines '~ wrong => right' in the hotwords file: substitutions tried ONLY when a sentence fails to match
    (validated by the matcher afterwards), e.g. '~ like => light', '~ tone of => turn off'."""
    reps = []
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("~") and "=>" in line:
                a, b = [x.strip() for x in line[1:].split("=>", 1)]
                if a and b:
                    reps.append((re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE), b))
    return reps


def load_name_fixes(path: str):
    """Lines 'wrong => Right' in the hotwords file: spelling fixes applied to every transcript."""
    fixes = []
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if "=>" in line and not line.startswith("#") and not line.startswith("~"):
                a, b = [x.strip() for x in line.split("=>", 1)]
                if a and b:
                    fixes.append((re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE), b))
    return fixes


def clean_text(text: str, fixes) -> str:
    text = words_to_digits(text)
    for rx, rep in fixes:
        text = rx.sub(rep, text)
    return text.strip()


COMMAND_WORDS = ("turn on off the a an dim set to percent light lights in at is are it its what time weather open close "
                 "play stop pause next volume up down bright dark cold hot warm loud quiet too bit little please make "
                 "everything all my our and of for with timer minutes minute hour hours cancel start how far tall old who where "
                 "when why which thermostat degrees temperature humidity scene mode night morning").split()


def build_vocab(hotwords_path: str):
    """Words the streaming engine may mis-hear: every word of every hotword phrase + common command words."""
    words = set(COMMAND_WORDS)
    if hotwords_path and os.path.exists(hotwords_path):
        for line in open(hotwords_path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=>" in line:
                continue
            phrase = line.rsplit(" :", 1)[0]
            for w in re.sub(r"[^\w' ]+", " ", phrase.lower()).split():
                if len(w) > 1:
                    words.add(w)
    return sorted(words)


def load_phrases(path: str):
    """The plain hotword phrases (device / room / playlist names) for slot repair."""
    out = []
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=>" in line:
                continue
            out.append(line.rsplit(" :", 1)[0])
    return out


def closest_phrase(value: str, phrases, cutoff: float = 0.6):
    import difflib
    v = value.lower().strip()
    best, score = None, 0.0
    for p in phrases:
        r = difflib.SequenceMatcher(None, v, p.lower()).ratio()
        if r > score:
            best, score = p, r
    return (best, score) if score >= cutoff else (None, score)


def repair_candidates(text: str, vocab, max_fixes: int = 3):
    """Replace words that are NOT in the vocabulary by their closest vocabulary word (difflib) - yields a few
    alternative sentences, most conservative first. Numbers, short words and in-vocab words are left alone."""
    import difflib
    toks = re.sub(r"[^\w' %]+", " ", text).split()
    fixes = []  # (index, replacement, similarity)
    for i, t in enumerate(toks):
        w = t.lower().strip("'")
        if not w or w in vocab or w.isdigit() or "%" in w or len(w) < 3:
            continue
        m = difflib.get_close_matches(w, vocab, n=1, cutoff=0.6)
        if m and m[0] != w:
            fixes.append((i, m[0], difflib.SequenceMatcher(None, w, m[0]).ratio()))
    fixes.sort(key=lambda f: -f[2])
    fixes = fixes[:max_fixes]
    out = []
    for k in range(1, len(fixes) + 1):
        t2 = list(toks)
        for i, rep, _ in fixes[:k]:
            t2[i] = rep
        cand = " ".join(t2)
        if cand.lower() != text.lower() and cand not in out:
            out.append(cand)
    return out


def vocab_score(text: str, vocab) -> int:
    return sum(1 for w in re.sub(r"[^\w' %]+", " ", text.lower()).split() if w in vocab or w.isdigit() or "%" in w)


NOT_UNDERSTOOD = "casa did not understand"   # custom sentence in HA -> instant "Sorry, I didn't catch that"
WAKE_ECHO_RE = re.compile(r"^\s*(?:(?:hey|hi|hay|a|the|tv|okay|ok|o\.?k\.?)[\s,.]+(?:coco|cocoa|koko|kokko|nabu)\b[\s,.!?]*|(?:coco|koko|nabu)\b[,.!?]+\s*)", re.I)  # bare word only with punctuation ("cocoa recipe" stays)
HALLUCINATION_RE = re.compile(r"thank(s| you)( so much)? for watching|please subscribe|like and subscribe|subscribe to (my|the|our)|subtitles by|captions by|see you (in the )?next (video|time)|home assistant voice commands and general questions|^\W*devices: |^\W*(rooms?|room service|room\.? ?(service)?|it'?s a picture of a room|rooms are available|miami|everyone)\W*$", re.I)  # last two = whisper echoing its own vocabulary prompt
CMD_START_RE = re.compile(r"^\W*(please\s+)?(turn|switch|dim|brighten|set|make|put|flip|kill|shut|lights?)\b", re.I)
# (2026-09-28) several house commands in one sentence: "turn off the monitor light and turn on the desk lamp",
# "turn off the countertop and set the thermostat to 20". Split on and / then / also; a part without its own verb borrows
# the previous one ("turn off the monitor and the desk lamp") but only when it names a device - so "set a timer for
# 1 hour and 30 minutes" or "play rock and roll" never split.
_MULTI_SEP_RE = re.compile(r"\s*,?\s+(?:and then|and also|and|then|also|plus)\s+|\s*[,;]\s+(?=(?:please\s+)?(?:turn|switch|set|dim|brighten|open|close|start|stop|lock|unlock|play|pause)\b)", re.I)
_MULTI_VERB_RE = re.compile(r"^\W*(?:(?:please|can you|could you|and)\s+)*((?:turn|switch)\s+(?:on|off)|turn|switch|set|dim|brighten|open|close|start|stop|lock|unlock|put|play|pause|resume|skip)\b", re.I)
_MULTI_DEVICE_RE = re.compile(r"\b(lights?|lamps?|monitor|desk|sofa|overhead|countertop|counter|kitchen|dining|hallway|entry|corridor|ceiling|pendant|"
                              r"mirror|vanity|closets?|wardrobe|walk-in|hatch|thermostat|heat(?:ing)?|temperature|ac|fan|scent|diffuser|blinds?|tv|music|"
                              r"humidifier|bedroom|office|living|degrees)\b", re.I)


def split_multi(text: str) -> list:
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
            out.append(f"{verb} {p}")
        else:
            return [text]
    return out
MUSIC_RE = re.compile(r"\s*(play|put on|pause|resume|skip|next|previous)\b", re.I)
_STOP = set("a an the is are was were be to of in on at for and or do does did i you he she it we they them my your "
             "what what's whats who who's how when where why which that this there their these those can could would will "
             "should please me us our its with from by about as up down so if then than just".split())
_NUMS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
         "nine": "9", "ten": "10", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60"}


def content_words(t: str) -> list:
    """The words that carry the meaning (names, nouns, numbers), normalised - 'a one-minute timer' == 'a 1 minute timer'."""
    words = re.sub(r"[^a-z0-9' ]+", " ", (t or "").lower().replace("-", " ")).split()
    words = [_NUMS.get(w, w) for w in words]
    return [w.removesuffix("'s") for w in words if w not in _STOP]


def agree(a: str, b: str) -> bool:
    """Two engines heard the same request: every content word matches (only small words may differ)."""
    ca, cb = content_words(a), content_words(b)
    return bool(ca) and ca == cb


def vote(cands: list) -> str:
    """Pick the transcript the others agree with most (ties -> the earlier = more trusted one)."""
    cands = [c for c in cands if c]
    if len(cands) < 3:
        return cands[0] if cands else ""
    score = lambda i: sum(similar(cands[i], cands[j]) for j in range(len(cands)) if j != i)
    return cands[max(range(len(cands)), key=lambda i: (score(i), -i))]


def similar(a: str, b: str) -> float:
    """0..1 similarity of two transcripts, ignoring case / punctuation / number formatting."""
    import difflib
    n = lambda t: " ".join(re.sub(r"[^a-z0-9' ]+", " ", (t or "").lower()).split())
    return difflib.SequenceMatcher(None, n(a), n(b)).ratio()


QUESTION_RE = re.compile(r"^\W*(what|what's|whats|who|who's|how|when|where|why|which|is|are|was|were|do|does|did|can|could|would|will|"
                         r"should|tell|remind|play|pause|resume|stop|skip|next|previous|add|remove|search|find|show|give|read|open|"
                         r"start|cancel|set a timer|timer|weather|temperature|time|date|list|translate|explain|define|calculate|"
                         r"say|repeat|thanks|thank|hello|hi|good|ok|okay|never mind|nevermind|hey)\b", re.IGNORECASE)
HOUSE_RE = re.compile(r" (turn|switch|dim|brighten|set|make|put|flip|kill|shut|light|lights|lamp|scent|diffuser|monitor|blind|blinds|"
                      r"humidifier|thermostat|speaker|soundbar|tv|office|bedroom|kitchen|living|hallway|entry|foyer|closet|wardrobe|"
                      r"mirror|vanity|pendant|ceiling|sofa|overhead|counter|countertop|corridor|dining|cooking|desk|"
                      r"everything|all|on|off)(\W|$)")


# ---------------------------------------------------------------- HA matcher
class HAMatcher:
    """Asks Home Assistant's built-in sentence matcher whether a text is a known command."""

    def __init__(self, url: str, token: str):
        self.url = url.replace("http", "ws", 1) + "/api/websocket"
        self.token = token
        self._ws = None
        self._session = None
        self._id = 0
        self._lock = asyncio.Lock()

    async def _connect(self):
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession()
        self._ws = await asyncio.wait_for(
            self._session.ws_connect(self.url, timeout=aiohttp.ClientWSTimeout(ws_close=3), heartbeat=20), 3)
        await asyncio.wait_for(self._ws.receive_json(), 3)  # auth_required
        await self._ws.send_json({"type": "auth", "access_token": self.token})
        r = await asyncio.wait_for(self._ws.receive_json(), 3)
        if r.get("type") != "auth_ok":
            raise RuntimeError(f"HA auth failed: {r}")
        self._id = 0

    async def fire(self, event_type: str, data: dict) -> None:
        """Fire an HA event (the Echo dashboard shows the live transcript from `casa_live_text`). Best effort."""
        import aiohttp
        if not self.token:
            return
        try:
            if getattr(self, "_http", None) is None or self._http.closed:
                self._http = aiohttp.ClientSession()
            base = self.url.replace("ws", "http", 1).replace("/api/websocket", "")
            async with self._http.post(f"{base}/api/events/{event_type}", json=data,
                                       headers={"Authorization": f"Bearer {self.token}"},
                                       timeout=aiohttp.ClientTimeout(total=2)) as r:
                await r.read()
        except Exception as e:  # noqa: BLE001
            _LOGGER.debug("fire %s failed: %s", event_type, e)

    async def is_command(self, text: str, timeout: float = 0.4) -> Optional[str]:
        name, _ = await self.match(text, timeout)
        return name

    async def post_text(self, entity_id: str, value: str, stage: str = "") -> None:
        """Show the transcript on the dashboard (input_text helper); best effort, never blocks a reply."""
        import aiohttp
        if not self.token:
            return
        shown = value if value != NOT_UNDERSTOOD else "(didn't catch that)"
        try:
            async with aiohttp.ClientSession() as sess:
                await sess.post(self.url.replace("ws", "http", 1).replace("/api/websocket", "/api/services/input_text/set_value"),
                                json={"entity_id": entity_id, "value": shown[:250]},
                                headers={"Authorization": f"Bearer {self.token}"}, timeout=aiohttp.ClientTimeout(total=3))
        except Exception as e:  # noqa: BLE001
            _LOGGER.debug("post_text failed: %s", e)

    async def match(self, text: str, timeout: float = 0.4):
        """(intent name or None, unmatched slots dict) - HA's own sentence matcher on `text`."""
        import aiohttp
        if not text.strip() or not self.token:
            return None, {}
        async with self._lock:
            for attempt in range(2):
                try:
                    if self._ws is None or self._ws.closed:
                        await self._connect()
                    self._id += 1
                    await self._ws.send_json({"id": self._id, "type": "conversation/agent/homeassistant/debug",
                                              "sentences": [text]})
                    while True:
                        msg = await asyncio.wait_for(self._ws.receive(), timeout=timeout)
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise RuntimeError(f"websocket {msg.type}")
                        r = json.loads(msg.data)
                        if r.get("id") == self._id:
                            break
                    if not r.get("success"):
                        return None, {}
                    res = (r["result"].get("results") or [None])[0]
                    if not res:
                        return None, {}
                    name = (res.get("intent") or {}).get("name") or ""
                    unmatched = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in (res.get("unmatched_slots") or {}).items()}
                    # `match` is only true when the sentence fully matched AND its targets exist
                    if not res.get("match") or not name:
                        return None, unmatched
                    return name, unmatched
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("HA matcher error (%s), reconnecting", e)
                    try:
                        if self._ws:
                            await self._ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._ws = None
            return None, {}


# ---------------------------------------------------------------- engines
class Engines:
    def __init__(self, cfg: Config):
        sd, od = cfg.stream_dir, cfg.offline_dir
        hot = None
        if cfg.hotwords and os.path.exists(cfg.hotwords):
            base = os.path.dirname(os.path.abspath(cfg.hotwords))
            vocab = os.path.join(base, "kroko.bpe.vocab")
            build_bpe_vocab(os.path.join(sd, "tokens.txt"), vocab)
            hot = os.path.join(base, "hotwords.encoded.txt")
            n = build_hotwords_file(cfg.hotwords, hot)
            _LOGGER.info("hotwords: %d lines from %s (score %.1f)", n, cfg.hotwords, cfg.hotwords_score)
        kw = dict(tokens=os.path.join(sd, "tokens.txt"), encoder=os.path.join(sd, "encoder.onnx"),
                  decoder=os.path.join(sd, "decoder.onnx"), joiner=os.path.join(sd, "joiner.onnx"),
                  num_threads=cfg.stream_threads, decoding_method="modified_beam_search", max_active_paths=cfg.beam)
        if hot:
            kw.update(hotwords_file=hot, hotwords_score=cfg.hotwords_score, modeling_unit="bpe", bpe_vocab=vocab)
        t0 = time.perf_counter()
        self.online = sherpa_onnx.OnlineRecognizer.from_transducer(**kw)
        _LOGGER.info("streaming model ready (%s) in %.1fs", os.path.basename(sd), time.perf_counter() - t0)
        self.offline = None
        self.cfg = cfg
        if cfg.offline_engine in ("parakeet", "hybrid"):
            t0 = time.perf_counter()
            self.offline = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=os.path.join(od, "encoder.int8.onnx"), decoder=os.path.join(od, "decoder.int8.onnx"),
                joiner=os.path.join(od, "joiner.int8.onnx"), tokens=os.path.join(od, "tokens.txt"),
                num_threads=cfg.offline_threads, model_type="nemo_transducer", decoding_method="greedy_search")
            _LOGGER.info("offline model ready (%s) in %.1fs", os.path.basename(od), time.perf_counter() - t0)
        else:
            _LOGGER.info("stage 2 = whisper.cpp server at %s", cfg.whisper_url)
        self.tail = np.zeros(int(RATE * cfg.tail_silence), dtype=np.float32)
        self.fixes = load_name_fixes(cfg.hotwords)
        self.repairs = load_repairs(cfg.hotwords)
        self.phrases = load_phrases(cfg.hotwords)
        self.vocab = build_vocab(cfg.hotwords)
        _LOGGER.info("name fixes: %d, repair rules: %d, phrases: %d, vocabulary: %d words",
                     len(self.fixes), len(self.repairs), len(self.phrases), len(self.vocab))
        # warm up both
        s = self.online.create_stream(); s.accept_waveform(RATE, np.zeros(RATE, np.float32)); s.input_finished()
        while self.online.is_ready(s):
            self.online.decode_stream(s)
        if self.offline is not None:
            o = self.offline.create_stream(); o.accept_waveform(RATE, np.zeros(RATE, np.float32)); self.offline.decode_stream(o)

    @staticmethod
    def normalize(pcm: np.ndarray) -> np.ndarray:
        """(2026-09-28, far-field) level the VOICE, not the loudest peak. Every turn starts with the Echo's own wake chime
        (~-15 dBFS, recorded by its mic); scaling by the peak meant scaling by the chime, so a voice from the sofa
        (-40…-50 dBFS) stayed quiet for whisper. Now: the chime is ducked when it is clearly louder than the rest, and
        the loud frames of what remains (the speech) are brought to ~-20 dBFS (max +30 dB), with a soft limiter."""
        if not len(pcm):
            return pcm
        pcm = pcm.astype(np.float32).copy()
        c = int(RATE * CHIME_S)
        if len(pcm) > c + RATE // 4:
            head, rest = float(np.max(np.abs(pcm[:c]))), float(np.max(np.abs(pcm[c:])))
            if head > 2.5 * max(rest, 1e-4):
                pcm[:c] *= max(rest, 1e-4) / head * 0.5
        fr = RATE // 20
        n = len(pcm) // fr
        if n < 2:
            return pcm
        rms = np.sqrt(np.mean(pcm[: n * fr].reshape(n, fr) ** 2, axis=1))
        loud = float(np.percentile(rms, 90))
        if loud <= 0:
            return pcm
        gain = min(10 ** (30 / 20), max(0.5, 0.1 / loud))
        return np.tanh(pcm * gain * 1.2) / 1.2

    def offline_text(self, pcm: np.ndarray) -> str:
        """Parakeet (CPU, sherpa-onnx)."""
        pcm = self.normalize(pcm)
        s = self.offline.create_stream()
        s.accept_waveform(RATE, pcm)
        self.offline.decode_stream(s)
        return s.result.text.strip()

    async def whisper_text(self, pcm: np.ndarray, url: str | None = None) -> str:
        """Whisper sees at most 30 s at a time: a longer turn (up to 60 s) is cut at its quietest 20 ms between 20 and
        28 s and transcribed piece by piece (2026-09-26)."""
        if len(pcm) <= RATE * 28:
            return await self._whisper_once(pcm, url)
        lo, hi, fr = RATE * 20, RATE * 28, RATE // 50
        seg = pcm[lo:hi]
        n = len(seg) // fr
        energy = np.sqrt(np.mean(seg[: n * fr].reshape(n, fr) ** 2, axis=1))
        cut = lo + int(np.argmin(energy)) * fr
        first = await self._whisper_once(pcm[:cut], url)
        rest = await self.whisper_text(pcm[cut:], url)
        return " ".join(t for t in (first, rest) if t)

    async def _whisper_once(self, pcm: np.ndarray, url: str | None = None) -> str:
        """whisper.cpp server (iGPU). Up to 30 s of audio, normalised; empty on timeout/error.
        (2026-09-26) was cut to 8 s here + the server's -ac 512 (~10 s): long spoken questions lost their end ("…should
        I take it out"). The audio context is now sized per request: short commands stay as fast as before."""
        import io, math, wave, aiohttp
        pcm = self.normalize(pcm[: RATE * 30])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
        form = aiohttp.FormData()
        form.add_field("file", buf.getvalue(), filename="audio.wav", content_type="audio/wav")
        form.add_field("response_format", "json")
        form.add_field("temperature", "0")
        # 50 encoder frames per second of audio (1500 = 30 s); never below the server's 512 default
        form.add_field("audio_ctx", str(min(1500, max(512, math.ceil(len(pcm) / RATE * 50) + 64))))
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url or self.cfg.whisper_url, data=form, timeout=aiohttp.ClientTimeout(total=self.cfg.whisper_timeout)) as r:
                    j = await r.json(content_type=None)
            text = (j.get("text") or "").strip()
            # a repetition loop shows up as the same sentence twice
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if len(lines) > 1 and len(set(l.lower() for l in lines)) == 1:
                text = lines[0]
            return " ".join(lines) if len(lines) > 1 else text
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning("whisper error: %s", e)
            return ""


CAPTURE_FLAG, CAPTURE_DIR = "/opt/casa-stt/capture.flag", "/opt/casa-stt/capture"
CHIME_S = 0.9   # the Echo's wake chime at the start of every turn (recorded by its own mic) - whisper's normalize()
#   only ducks it when it is clearly louder than the speech
STREAM_CHIME_S = 0.0   # (2026-10-19) the LIVE text no longer ducks the first 0.9 s (was CHIME_S): people talk right
#   after the wake word since VACA patch v2, and the duck turned "What's the..." into fragments ("tomorrow").
#   (A first attempt set CHIME_S itself to 0 -> normalize() crashed on an empty slice: every request failed 9:00-10:30.)


def save_recording(rec_dir: str, keep: int, pcm: np.ndarray, stage: str, text: str) -> None:
    """Write the utterance as 16 kHz mono WAV: <time>_<stage>_<text>.wav; keep only the newest `keep` files."""
    import glob, wave
    os.makedirs(rec_dir, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text)[:40].strip("_") or "empty"
    path = os.path.join(rec_dir, time.strftime("%Y%m%d_%H%M%S") + f"_{stage}_{slug}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    files = sorted(glob.glob(os.path.join(rec_dir, "*.wav")))
    for old in files[:-keep] if len(files) > keep else []:
        os.remove(old)


# ---------------------------------------------------------------- Wyoming handler
class Handler(AsyncEventHandler):
    def __init__(self, cfg: Config, engines: Engines, matcher: HAMatcher, info: Info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg, self.eng, self.matcher, self.info = cfg, engines, matcher, info
        self.stream = None
        self.buf: List[np.ndarray] = []
        self.t_start = None
        self.t_last_chunk = None
        self.in_rate = RATE
        self._live_t, self._live_last = 0.0, ""

    def _stream_level(self, pcm: np.ndarray) -> np.ndarray:
        """(2026-09-28) the streaming model gets the chime ducked and quiet speech raised (running gain from the loudest
        50 ms seen after the chime, 1x…16x), so a voice from across the room shows up in the live transcript."""
        pos = getattr(self, "_lvl_pos", 0)
        if pos == 0:
            self._lvl_peak = 0.0
        self._lvl_pos = pos + len(pcm)
        c = int(RATE * STREAM_CHIME_S)
        if c and pos + len(pcm) <= c:
            return pcm * 0.15
        if c and pos < c:
            pcm = pcm.copy(); pcm[: c - pos] *= 0.15
        fr = RATE // 20
        for i in range(max(0, c - pos), len(pcm) - fr + 1, fr):
            self._lvl_peak = max(self._lvl_peak, float(np.sqrt(np.mean(pcm[i:i + fr] ** 2))))
        gain = min(16.0, max(1.0, 0.1 / max(self._lvl_peak, 1e-4)))
        return np.tanh(pcm * gain * 1.2) / 1.2 if gain > 1.0 else pcm

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Transcribe.is_type(event.type) or AudioStart.is_type(event.type):
            if AudioStart.is_type(event.type):
                self.in_rate = AudioStart.from_event(event).rate
            self.stream = self.eng.online.create_stream()
            self.buf = []
            self.t_start = time.perf_counter()
            self._live_t, self._live_last = 0.0, ""
            self._lvl_pos = 0          # per-turn level state (chime ducking + running gain)
            asyncio.get_running_loop().create_task(self.matcher.fire("casa_live_text", {"text": "", "final": False}))
            return True
        if AudioChunk.is_type(event.type):
            if self.stream is None:
                self.stream = self.eng.online.create_stream(); self.buf = []; self.t_start = time.perf_counter(); self._lvl_pos = 0
            chunk = AudioChunk.from_event(event)
            pcm = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
            if chunk.channels and chunk.channels > 1:
                pcm = pcm.reshape(-1, chunk.channels).mean(axis=1)
            if chunk.rate and chunk.rate != RATE:
                idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * RATE / chunk.rate)).astype(np.int64)
                pcm = pcm[idx]
            self.buf.append(pcm)
            self.stream.accept_waveform(RATE, self._stream_level(pcm))
            while self.eng.online.is_ready(self.stream):
                self.eng.online.decode_stream(self.stream)
            self.t_last_chunk = time.perf_counter()
            if self.t_last_chunk - self._live_t >= 0.25:
                self._live_t = self.t_last_chunk
                partial = self.eng.online.get_result(self.stream).strip()
                if partial and partial != self._live_last:
                    self._live_last = partial
                    asyncio.get_running_loop().create_task(
                        self.matcher.fire("casa_live_text", {"text": partial.capitalize(), "final": False}))
            return True
        if AudioStop.is_type(event.type):
            t_stop = time.perf_counter()
            text, stage, intent, stream_text = "", "stream", None, ""
            audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
            # (2026-09-28) wake-word training capture: while CAPTURE_FLAG exists (its content = a tag, e.g. "office"), every
            # turn is saved to CAPTURE_DIR and answered with a fixed phrase that HA handles silently (intent CasaCapture) -
            # no recognition, no error tone. The orchestrator opens silent sessions on ONE Echo; the others work normally.
            if os.path.exists(CAPTURE_FLAG) and len(audio):
                try:
                    tag = open(CAPTURE_FLAG).read().strip() or "capture"
                    os.makedirs(CAPTURE_DIR, exist_ok=True)
                    save_recording(CAPTURE_DIR, 100000, audio, tag, "amb")
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("capture failed: %s", e)
                self.stream = None; self.buf = []
                await self.write_event(Transcript(text="casa capture sample").event())
                return False
            if self.stream is not None and len(audio):
                self.stream.accept_waveform(RATE, self.eng.tail)
                self.stream.input_finished()
                while self.eng.online.is_ready(self.stream):
                    self.eng.online.decode_stream(self.stream)
                text = clean_text(self.eng.online.get_result(self.stream), self.eng.fixes)
            stream_text = text
            t_stream = time.perf_counter()
            # (2026-10-19) show the whole streamed sentence the moment speech ends; whisper's final replaces it ~1 s later
            if text and text.strip().lower() != (self._live_last or "").strip().lower():
                self._live_last = text
                asyncio.get_running_loop().create_task(
                    self.matcher.fire("casa_live_text", {"text": text[:1].upper() + text[1:], "final": False}))
            intent, how = None, ""

            async def check(t):
                try:
                    return await asyncio.wait_for(self.matcher.is_command(t), 1.2)
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("matcher timeout/error: %s", e)
                    return None

            async def match(t):
                try:
                    return await asyncio.wait_for(self.matcher.match(t), 1.2)
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("matcher timeout/error: %s", e)
                    return None, {}

            async def check_with_repair(t):
                """exact match; else known confusions (~ rules); else fix the unmatched name/area slot with the
                closest house phrase; else vocabulary repair. Returns (text, intent, how)."""
                it, unmatched = await match(t)
                if it:
                    return t, it, "exact"
                tried = {t.lower()}
                # 1) known confusions ("like"->"light", "tone of"->"turn off" ...)
                fixed = t
                for rx, rep in self.eng.repairs:
                    fixed = rx.sub(rep, fixed)
                if fixed.lower() not in tried:
                    tried.add(fixed.lower())
                    it, unmatched2 = await match(fixed)
                    if it:
                        return fixed, it, f"repaired {t!r}"
                    if unmatched2:
                        unmatched = unmatched2  # keep the ORIGINAL text; `fixed` is only a candidate
                # 2) the matcher tells us which words failed to be a device/area name -> closest house phrase
                question = bool(QUESTION_RE.match(t))
                for slot, val in list(unmatched.items())[:2]:
                    if slot == "name" and isinstance(val, str) and val.strip():
                        # a question ("what is Obama's height") must not be bent onto a device name
                        best, score = closest_phrase(val, self.eng.phrases, cutoff=0.85 if question else 0.66)
                        if best:
                            cand = re.sub(re.escape(val.strip()), best, fixed, count=1, flags=re.IGNORECASE)
                            if cand.lower() not in tried:
                                tried.add(cand.lower())
                                it, _ = await match(cand)
                                if it:
                                    return cand, it, f"slot {slot} {val.strip()!r}->{best!r} ({score:.2f})"
                # 3) generic vocabulary repair
                for cand in repair_candidates(t, self.eng.vocab):
                    if cand.lower() in tried:
                        continue
                    tried.add(cand.lower())
                    it, _ = await match(cand)
                    if it:
                        return cand, it, f"repaired {t!r}"
                return t, None, ""

            if text and not self.cfg.always_offline:
                text, intent, how = await check_with_repair(text)
            t_match = time.perf_counter()
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            # (2026-09-27) an Echo whose microphone silently died sends pure digital zeros (the office Show 5 did for hours):
            # tell HA, which flags the Echo that was listening; my Echo watchdog (not in this repo) reboots it once idle
            if len(audio) >= RATE and float(np.max(np.abs(audio))) == 0.0:
                asyncio.get_running_loop().create_task(self.matcher.fire("casa_mic_dead", {"seconds": round(len(audio) / RATE, 1)}))
            # (2026-09-28) the office Show 5 on the beamforming test ROM (lineage 20260927): the HAL closed the capture
            # and VACA kept "listening" with NO audio at all (0.00 s per turn) until the app restarted -> same recovery
            if len(audio) == 0 and time.perf_counter() - (self.t_start or time.perf_counter()) > 3:
                asyncio.get_running_loop().create_task(self.matcher.fire("casa_mic_dead", {"seconds": 0, "no_audio": True}))
            if intent is None and len(audio) >= RATE * 0.3 and rms < 0.001:
                how, text = f"silence (rms {rms:.4f})", ""   # nothing was said: don't let whisper hallucinate
            elif intent is None and len(audio) >= RATE * 0.3:
                loop = asyncio.get_running_loop()
                if self.eng.offline is not None and self.cfg.offline_engine == "hybrid":
                    # (2026-09-24) Parakeet on the CPU (~0.25 s, and no GPU at all) first. When it agrees with the
                    # independent streaming model, two engines heard the same thing -> use it now. Only when they
                    # disagree ask whisper on the iGPU (~1.3 s) - and meanwhile let HA's casa_brain start answering
                    # Parakeet's version (casa_stream_question head start; it keeps that answer only if whisper matches).
                    # whisper starts at the same moment (its answer is only waited for when the two disagree)
                    wtask = loop.create_task(self.eng.whisper_text(audio))
                    wtask.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                    ptxt = clean_text(await loop.run_in_executor(None, self.eng.offline_text, audio), self.eng.fixes)
                    same = agree(ptxt, stream_text)
                    if same or not ptxt and not stream_text:
                        text2, how_eng = ptxt, "parakeet"
                    else:
                        guess = ptxt or stream_text
                        if guess and QUESTION_RE.match(guess) and not HOUSE_RE.search(" " + guess.lower() + " "):
                            loop.create_task(self.matcher.fire("casa_stream_question", {"text": guess}))
                        wtxt = clean_text(await wtask, self.eng.fixes)
                        # 3 engines, 3 opinions: take the one the other two agree with most (whisper first on ties)
                        text2 = vote([wtxt, ptxt, stream_text]) if wtxt else ptxt
                        how_eng = f"vote (whisper {wtxt!r}, parakeet {ptxt!r})"
                    _LOGGER.debug("stage 2: %s", how_eng)
                elif self.eng.offline is not None and self.cfg.offline_engine == "parakeet":
                    text2 = await loop.run_in_executor(None, self.eng.offline_text, audio)
                else:
                    # (2026-09-24) NO head start from the stream text any more: it ran the LLM on the iGPU at the same
                    # moment as whisper (both ~2x slower) and the stream mishears names ("Ironan" -> the LLM answered
                    # the wrong film and casa_brain accepted it as 93 % similar). Measured end-to-end: net loss.
                    # (2026-09-24) questions -> whisper SMALL (0.37 s vs 0.84 s; same text on my questions), house
                    # commands keep MEDIUM (small garbles "countertop light" -> "contact applied")
                    question = bool(stream_text) and bool(QUESTION_RE.match(stream_text)) and not HOUSE_RE.search(" " + stream_text.lower() + " ")
                    text2 = await self.eng.whisper_text(audio, self.cfg.whisper_fast_url if question and self.cfg.whisper_fast_url else None)
                    if question and not text2:
                        text2 = await self.eng.whisper_text(audio)       # fast server down -> the accurate one
                    # whisper hears my "What's the / What is the" as "Watch the / Watch is the"
                    text2 = re.sub(r"^\s*watch(?:'s| is|es)?\s+(?=(?:the|is)\b)", "What is ", text2 or "", flags=re.I)
                    text2 = re.sub(r"^What is is\b", "What is", text2)
                # (2026-09-28) the tail of the wake word leaks into the command ("Hey Coco, what's the weather", "TV cocoa. Set a
                # timer…"): the mic opens right after the blip, so strip a leading wake phrase before the canonicalizer sees it
                if text2:
                    text2 = WAKE_ECHO_RE.sub("", text2, count=1)
                # (2026-09-27) whisper's classic hallucinations on silence / room noise (YouTube outro lines): drop them, so a
                # false wake or a silent follow-up never sends "Thank you for watching" to the assistant
                if text2 and HALLUCINATION_RE.search(text2):
                    _LOGGER.info("dropped whisper hallucination %r", text2)
                    text2 = ""
                if text2:
                    text2 = clean_text(text2, self.eng.fixes)
                    if not self.cfg.always_offline:
                        text2, intent, how = await check_with_repair(text2)
                    # the accurate engine wins unless it returned nothing
                    text, stage = text2, ("parakeet" if self.cfg.offline_engine == "hybrid" and how_eng == "parakeet"
                                          else "whisper" if self.cfg.offline_engine == "hybrid" else self.cfg.offline_engine)
            # Not a command after every repair. Questions / general requests go on to the LLM; fragments and
            # garbled HOUSE commands ("turn off molecule", "turn on turn it like") get an instant local
            # "didn't catch that" instead of 30-60 s of the LLM guessing (and blocking the satellite).
            # (2026-09-28) several commands in one sentence: fine when EVERY part is a known command on its own
            if intent is None and text and not self.cfg.always_offline:
                parts = split_multi(text)
                if len(parts) > 1:
                    fixed = []
                    for pt in parts:
                        f, it_, _ = await check_with_repair(pt)
                        if not it_:
                            break
                        fixed.append(f)
                    else:
                        text, intent, how = " and ".join(fixed), "multi", f"split into {len(parts)}"
            if intent is None and text:
                words = [w for w in re.sub(r"[^\w' ]+", " ", text).split() if any(c.isalpha() for c in w)]
                low = " " + text.lower() + " "
                # "play Moody" / "next song" are real two-word commands - don't treat them as fragments
                if len(words) < 3 and not MUSIC_RE.match(text):
                    how, text = f"discarded fragment {text!r}", ""
                # (2026-09-25/26) only COMMAND-shaped text ("turn/switch/set/dim…") counts as a garbled house command: HOUSE_RE matches
                # everyday words ("on", "all", "cooking", "kitchen"), so "I was cooking bread in the oven and noticed it
                # was hard, what could it be" used to get "didn't catch that". Long free-form sentences go to the LLM.
                elif not QUESTION_RE.match(text) and HOUSE_RE.search(low) and CMD_START_RE.match(text):
                    how, text = f"not understood {text!r}", NOT_UNDERSTOOD
            t_end = time.perf_counter()
            _LOGGER.info("%-8s %5.2fs audio | stream %.2fs | match %.2fs%s | total %.2fs | %r%s%s",
                         stage, len(audio) / RATE, t_stream - t_stop, t_match - t_stream,
                         f" ({intent})" if intent else "", t_end - t_stop, text,
                         f"   (stream heard: {stream_text!r})" if stage != "stream" else "",
                         f"   [{how}]" if how and how != "exact" else "")
            await self.write_event(Transcript(text=text).event())
            if text and text != NOT_UNDERSTOOD:
                asyncio.get_running_loop().create_task(self.matcher.fire("casa_live_text", {"text": text, "final": True}))
            if self.cfg.record_dir and len(audio):
                try:
                    save_recording(self.cfg.record_dir, self.cfg.record_keep, audio, stage, text)
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("recording failed: %s", e)
            self.stream = None
            self.buf = []
            return False
        return True


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default=os.environ.get("CASA_STT_URI", "tcp://0.0.0.0:10310"))
    p.add_argument("--models", default=os.environ.get("CASA_STT_MODELS", "/opt/casa-stt/models"))
    p.add_argument("--stream-model", default="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06")
    p.add_argument("--offline-model", default="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    p.add_argument("--hotwords", default=os.environ.get("CASA_STT_HOTWORDS", "/opt/casa-stt/hotwords.txt"))
    p.add_argument("--hotwords-score", type=float, default=float(os.environ.get("CASA_STT_HOTWORDS_SCORE", "1.5")))
    p.add_argument("--beam", type=int, default=int(os.environ.get("CASA_STT_BEAM", "4")))
    p.add_argument("--stream-threads", type=int, default=int(os.environ.get("CASA_STT_STREAM_THREADS", "4")))
    p.add_argument("--offline-threads", type=int, default=int(os.environ.get("CASA_STT_OFFLINE_THREADS", "8")))
    p.add_argument("--tail-silence", type=float, default=0.8)
    p.add_argument("--ha-url", default=os.environ.get("HA_URL", "http://127.0.0.1:8123"))
    p.add_argument("--ha-token", default=os.environ.get("HA_TOKEN", ""))
    p.add_argument("--always-offline", action="store_true", help="skip the matcher: always answer with Parakeet")
    p.add_argument("--offline-engine", default=os.environ.get("CASA_STT_OFFLINE_ENGINE", "whisper"), choices=["whisper", "parakeet", "hybrid"],
                   help="stage-2 engine: whisper = whisper.cpp server on the iGPU (accurate with accents), parakeet = sherpa-onnx CPU")
    p.add_argument("--whisper-url", default=os.environ.get("CASA_STT_WHISPER_URL", "http://127.0.0.1:10320/inference"))
    p.add_argument("--whisper-fast-url", default=os.environ.get("CASA_STT_WHISPER_FAST_URL", "http://127.0.0.1:10322/inference"),
                   help="whisper SMALL server used for questions ('' = always the main one)")
    p.add_argument("--whisper-timeout", type=float, default=float(os.environ.get("CASA_STT_WHISPER_TIMEOUT", "9")))
    p.add_argument("--record-dir", default=os.environ.get("CASA_STT_RECORD_DIR", ""), help="save every utterance as WAV here (debugging)")
    p.add_argument("--record-keep", type=int, default=int(os.environ.get("CASA_STT_RECORD_KEEP", "100")))
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(a)
    engines = Engines(cfg)
    matcher = HAMatcher(cfg.ha_url, cfg.ha_token)
    info = Info(asr=[AsrProgram(name="casa-stt", description="Kroko streaming + Parakeet TDT (two-stage)", version="1.0",
                                attribution=Attribution(name="sherpa-onnx", url="https://github.com/k2-fsa/sherpa-onnx"),
                                installed=True,
                                models=[AsrModel(name="kroko+parakeet-tdt-0.6b-v3", description="two-stage", version="1.0",
                                                 attribution=Attribution(name="Banafo / NVIDIA", url="https://huggingface.co/Banafo/Kroko-ASR"),
                                                 installed=True, languages=["en"])])])
    server = AsyncServer.from_uri(cfg.uri)
    _LOGGER.info("casa-stt listening on %s", cfg.uri)
    await server.run(lambda *args, **kw: Handler(cfg, engines, matcher, info, *args, **kw))


if __name__ == "__main__":
    asyncio.run(main())
