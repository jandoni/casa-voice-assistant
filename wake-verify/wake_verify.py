#!/usr/bin/env python3
"""casa wake verifier (2026-09-28), voice host :10350. Second stage of the two-stage wake word.

The patched VACA (casa5) POSTs the last 2 s of microphone audio (16 kHz mono PCM16) the moment its on-device
"Hey Coco" model fires: POST /verify?score=0.87 -> "accept" | "reject" (must answer in < 250 ms; the Echo fails open).
Phase 1 (now): record every wake (clip + score + which Echo) and always accept -> real training data for the verifier
model, including the audio of every false wake. Phase 2: a larger model decides.

Phase 2b (2026-10-01): TV / BROADCAST GUARD. Two nights of false wakes in empty rooms were all TV
dialogue from another room (the beamforming ROM's processing levels distant speech up to ~-22 dBFS), often waking
TWO Echos in the same second. (a) A wake is held 100 ms; if another room woke within 1.5 s at a similar level
(< 6 dB apart) it is a broadcast sound -> reject both; with a clear level gap only the louder (nearer) one passes.
(b) While the TV plays (CASA_TV_ENTITY playing, or the soundbar on its TV input), the Echos in the rooms listed in
CASA_TV_GUARD_ROOMS (the rooms that only hear the TV through the walls) need score >= 0.97.

Phase 2a (2026-09-29): VOICING GATE. The night's false wakes were door slams, thuds, bumps, a spoon (25-50 dB
transients), never speech. "Hey Coco" always carries >= 120 ms of voiced (pitched 70-400 Hz) speech in the clip
(86/86 real recordings of two adult voices through the Echo mics; median ~450 ms); 17 of 26 false wakes had < 60 ms.
-> reject when voiced < VOICED_MIN_MS. Fail-open: empty/silent clip, any error, or the kill switch file
/opt/casa-stt/wake_verify.off (touch it to accept everything again). ~3 ms.

Phase 3 (2026-11-01): DID SOMEONE SAY THE WORD? 6 false "Alexa" wakes in the TV room during one film, with
nothing even close to "Alexa" said. The on-device model (microWakeWord alexa, 0.7 there) scores
plenty of dialogue lines > 0.7 and the TV rules above only guard the OTHER rooms. Now the clip is also transcribed by
casa-stt's Kroko streaming zipformer (CPU, "ALEXA" hotword-boosted, ~60-120 ms, run while the 100 ms hold waits):
an "Alexa"-like word -> accept; other words -> reject ("asr"); nothing recognisable -> reject only while the TV plays.
Offline check (eval/asr_eval.py): synthetic "Alexa" clips 360/360, real Echo "Alexa" clips
3/3 clear ones, tonight's TV false wakes 0/9, ~4,500 other clips (room audio, commands to Casa, other wake words) 0.
Fail-open: model not loaded, error, or no answer within ASR_BUDGET_S. Kill switch: touch /opt/casa-stt/wake_asr.off.
Whispers (2026-11-02): a wake the voicing gate rejects is accepted when the word check hears "Alexa" ("whisper";
real whispered wakes: 3 of 4 rescued, 0 of the thud / noise rejects).
If the wake word changes, change WAKE_RE / WAKE_WORD (and the hotword) too.
"""
import concurrent.futures, json, os, re, threading, time, urllib.request, wave
from difflib import SequenceMatcher
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

OUT = "/opt/casa-stt/wakeclips"
OFF = "/opt/casa-stt/wake_verify.off"
ENV = dict(l.strip().split("=", 1) for l in open("/opt/casa-stt/casa-stt.env") if "=" in l and not l.startswith("#"))
# CASA_WAKE_ROOMS="<echo ip>=<room>,<echo ip>=<room>" names each Echo by the address it posts from (see casa-stt.env.example)
ECHOS = dict(p.strip().split("=", 1) for p in ENV.get("CASA_WAKE_ROOMS", "").strip('"').split(",") if "=" in p)
TV_ENTITY = ENV.get("CASA_TV_ENTITY", "media_player.tv").strip('"')
SOUNDBAR_ENTITY = ENV.get("CASA_SOUNDBAR_ENTITY", "").strip('"')
VOICED_MIN_MS = 60
HOLD_S, PAIR_S, PAIR_DB = 0.10, 1.5, 6.0
TV_ROOMS = tuple(r for r in ENV.get("CASA_TV_GUARD_ROOMS", "").strip('"').split(",") if r)
TV_MIN_SCORE = 0.97
RECENT = []            # (monotonic time, room, speech_db) of recent wakes
LOCK = threading.Lock()
TV = {"on": False}
# phase 3: the word check
ASR_MODEL = "/opt/casa-stt/models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
ASR_OFF = "/opt/casa-stt/wake_asr.off"
ASR_BUDGET_S = 0.19        # from the start of the request; VACA stops waiting at 250 ms and then accepts
ASR_HOT_SCORE = 2.0
WAKE_WORD = "alexa"
WAKE_RE = re.compile(r"\b(alexa|alexia|alexis|alexus|a lexa|elexa|lexa|alex)\b")
ASR = {"rec": None}
ASR_LOCK = threading.Lock()
POOL = concurrent.futures.ThreadPoolExecutor(2)


def ha_state(entity):
    req = urllib.request.Request(ENV["HA_URL"] + "/api/states/" + entity, headers={"Authorization": "Bearer " + ENV["HA_TOKEN"]})
    return json.load(urllib.request.urlopen(req, timeout=3))


def tv_poller():
    """Keep TV["on"] fresh off the request path (every 3 s)."""
    while True:
        try:
            atv = ha_state(TV_ENTITY)
            arc = ha_state(SOUNDBAR_ENTITY) if SOUNDBAR_ENTITY else {"state": "off", "attributes": {}}
            TV["on"] = atv["state"] == "playing" or (arc["state"] == "playing" and arc["attributes"].get("source") == "TV"
                                                     and atv["state"] not in ("off", "standby", "unavailable", "unknown"))
        except Exception:  # noqa: BLE001
            TV["on"] = False
        time.sleep(3)
os.makedirs(OUT, exist_ok=True)


def analyse(pcm, sr=16000):
    """(voiced ms, speech dBFS, noise dBFS) - speech = 75th percentile of the voiced frames, noise = 15th percentile
    of all frames. None, None, None = no usable audio."""
    """ms of frames that are both loud (floor + 10 dB, > -62 dBFS) and pitched (autocorr peak > 0.5 at 70-400 Hz).
    None = no usable audio (fail open)."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768
    if len(x) < sr // 2 or np.abs(x).max() < 1e-4:
        return None, None, None
    X = np.fft.rfft(x); X[: int(90 * len(x) / sr)] = 0; X[int(3000 * len(x) / sr):] = 0   # raw mic: DC/rumble
    y = np.fft.irfft(X, len(x)).astype(np.float32)
    N, Hp = 480, 160
    n = (len(y) - N) // Hp
    fr = np.stack([y[i * Hp:i * Hp + N] for i in range(n)]) * np.hanning(N).astype(np.float32)
    rms = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    floor = np.percentile(rms, 15)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * N)) ** 2)[:, :N]
    ac = ac / (ac[:, :1] + 1e-12)
    v = (ac[:, 40:229].max(1) > 0.5) & (rms > floor + 10) & (rms > -62)
    speech = float(np.percentile(rms[v], 75)) if v.sum() >= 5 else None
    return int(v.sum()) * 10, speech, float(floor)


def asr_load():
    try:
        import sherpa_onnx
        hot = "/opt/casa-stt/wake_hot.txt"
        with open(hot, "w") as f:
            f.write("ALEXA\nalexa\nAlexa\n")
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=ASR_MODEL + "/tokens.txt", encoder=ASR_MODEL + "/encoder.onnx", decoder=ASR_MODEL + "/decoder.onnx",
            joiner=ASR_MODEL + "/joiner.onnx", num_threads=2, decoding_method="modified_beam_search", max_active_paths=4,
            hotwords_file=hot, hotwords_score=ASR_HOT_SCORE, modeling_unit="bpe", bpe_vocab="/opt/casa-stt/kroko.bpe.vocab")
        ASR["rec"] = rec
        t = time.perf_counter()
        asr_text(np.zeros(32000, np.int16).tobytes())   # warm-up
        print(json.dumps({"asr": "ready", "warmup_ms": round((time.perf_counter() - t) * 1000)}), flush=True)
    except Exception as e:  # noqa: BLE001
        print("asr load failed (wakes are not word-checked)", e, flush=True)


def asr_text(pcm):
    """Kroko transcript of the clip: high-passed (raw mic rumble/DC), peak-normalised, with silence around it (the
    streaming model needs ~1 s of right context to emit a word that ends the clip). None = no model."""
    rec = ASR["rec"]
    if rec is None:
        return None
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768
    if len(x) < 1600:
        return None
    X = np.fft.rfft(x); X[: int(90 * len(x) / 16000)] = 0
    x = np.fft.irfft(X, len(x)).astype(np.float32)
    x *= min(0.5 / (float(np.abs(x).max()) + 1e-9), 31.6)
    with ASR_LOCK:
        st = rec.create_stream()
        st.accept_waveform(16000, np.zeros(8000, np.float32))
        st.accept_waveform(16000, x)
        st.accept_waveform(16000, np.zeros(19200, np.float32))
        st.input_finished()
        while rec.is_ready(st):
            rec.decode_stream(st)
        return rec.get_result(st).strip()


def said_wake(text):
    t = text.lower()
    if WAKE_RE.search(t):
        return True
    w = re.findall(r"[a-z]+", t)
    return any(SequenceMatcher(None, c, WAKE_WORD).ratio() >= 0.75 for c in w + [a + b for a, b in zip(w, w[1:])])


def tell_ha(room, speech, noise):
    """(2026-09-29) casa_wake_level -> HA sets this Echo's answer volume (far voice / noisy room = louder)."""
    try:
        req = urllib.request.Request(ENV["HA_URL"] + "/api/events/casa_wake_level",
                                     data=json.dumps({"room": room, "speech_db": speech, "noise_db": noise}).encode(),
                                     headers={"Authorization": "Bearer " + ENV["HA_TOKEN"], "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3).read()
    except Exception as e:  # noqa: BLE001
        print("casa_wake_level failed", e, flush=True)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        t0 = time.perf_counter()
        u = urlparse(self.path)
        if u.path != "/verify":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length") or 0)
        pcm = self.rfile.read(n) if n else b""
        score = (parse_qs(u.query).get("score") or ["?"])[0]
        room = ECHOS.get(self.client_address[0], self.client_address[0])
        verdict, vms, sp, nf = "accept", None, None, None
        try:
            vms, sp, nf = analyse(pcm)
            if vms is not None and vms < VOICED_MIN_MS and not os.path.exists(OFF):
                verdict = "reject"
        except Exception as e:  # noqa: BLE001
            print("voicing failed", e, flush=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        why = ""
        asr, said = None, None
        # (2026-11-02) a WHISPERED "Alexa" has no pitched frames, so the voicing gate threw it out (three real
        # whispered wakes scored 0.96 / 0.99 / 0.82 and were all rejected). The word check decides instead: "Alexa" heard ->
        # accept; thuds / doors / noise transcribe as nothing or other words -> still rejected.
        if verdict == "reject" and ASR["rec"] is not None and not os.path.exists(ASR_OFF):
            try:
                asr = asr_text(pcm)
                said = said_wake(asr) if asr is not None else None
            except Exception as e:  # noqa: BLE001
                print("asr failed", e, flush=True)
            if said:
                verdict, why = "accept", "whisper"
        # (b) TV playing: the rooms that only hear it through the walls need a very confident wake
        try:
            sc = float(score)
        except ValueError:
            sc = 1.0
        if verdict == "accept" and TV["on"] and room in TV_ROOMS and sc < TV_MIN_SCORE and not os.path.exists(OFF):
            verdict, why = "reject", "tv"
        # phase 3: transcribe the clip while the checks below run (the 100 ms hold covers most of it)
        fut = None
        if verdict == "accept" and asr is None and ASR["rec"] is not None and not os.path.exists(OFF):
            fut = POOL.submit(asr_text, pcm)
        # (a) the same sound woke another room at the same time -> a broadcast (TV), not a person next to this Echo
        now = time.monotonic()
        with LOCK:
            RECENT[:] = [r for r in RECENT if now - r[0] < 10]
            RECENT.append((now, room, sp))
        if verdict == "accept" and not os.path.exists(OFF):
            time.sleep(HOLD_S)
            with LOCK:
                others = [r for r in RECENT if r[1] != room and abs(r[0] - now) <= PAIR_S]
            for _, oroom, osp in others:
                # (2026-11-01) the other Echo heard NO speech (it only half-heard the person standing here: an Echo
                # in the next room picks up an "Alexa" said here): that is not a broadcast, so it can't veto this wake.
                # A TV line reaches both rooms AS speech at similar levels and is still rejected below.
                if osp is None:
                    continue
                if sp is None or abs(sp - osp) < PAIR_DB or osp > sp:
                    verdict, why = "reject", "pair:" + oroom
                    break
        if fut is not None:
            try:
                asr = fut.result(timeout=max(0.0, ASR_BUDGET_S - (time.perf_counter() - t0)))
            except Exception:  # noqa: BLE001 - too slow / failed: fail open
                asr = None
            if asr is not None:
                said = said_wake(asr)
                # other words = someone (or the TV) said something else; nothing at all only counts while the TV plays
                if verdict == "accept" and not said and (asr or TV["on"]) and not os.path.exists(ASR_OFF):
                    verdict, why = "reject", "asr"
        name = stamp + f"_{room}_{score}_v{vms}_{verdict}" + (f"_{why.replace(':', '-')}" if why else "") + ".wav"

        def save():
            try:
                with wave.open(os.path.join(OUT, name), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
            except Exception as e:  # noqa: BLE001
                print("save failed", e, flush=True)
        threading.Thread(target=save, daemon=True).start()
        if verdict == "accept" and sp is not None:   # off the answer path: the Echo already has its reply below
            threading.Thread(target=tell_ha, args=(room, round(sp, 1), round(nf, 1)), daemon=True).start()
        body = verdict.encode()
        self.send_response(200); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
        print(json.dumps({"room": room, "score": score, "bytes": n, "verdict": verdict, "why": why, "tv": TV["on"], "voiced_ms": vms, "speech_db": sp and round(sp, 1), "noise_db": nf and round(nf, 1), "asr": asr, "said": said, "ms": round((time.perf_counter() - t0) * 1000, 1), "file": name}), flush=True)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=tv_poller, daemon=True).start()
    threading.Thread(target=asr_load, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 10350), H).serve_forever()
