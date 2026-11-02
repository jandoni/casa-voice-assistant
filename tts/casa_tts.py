#!/usr/bin/env python3
"""casa-tts - Kokoro (82M, StyleTTS2) as a Wyoming TTS server for Home Assistant (voice host, :10340). 2026-09-25.

I picked Kokoro "af_bella" by ear out of 13 samples (warmer and more natural than Piper). Speaks sentence by sentence: audio of the first
sentence is sent as soon as it is ready (~0.45 s for a short one on 6 CPU threads), the rest follows while it plays.
A long first sentence is split at its first comma so speech starts sooner. CPU only (the iGPU stays for the LLM).
"""
import argparse, asyncio, logging, re, time
from functools import partial

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger("casa-tts")
RATE = 24000
VOICES = {"af_bella": "Bella (US female)", "af_heart": "Heart (US female)", "am_michael": "Michael (US male)",
          "am_fenrir": "Fenrir (US male)", "bf_emma": "Emma (British female)", "bm_george": "George (British male)"}


def chunks(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if parts and len(parts[0]) > 70 and ", " in parts[0][20:]:
        i = parts[0].index(", ", 20)
        parts = [parts[0][: i + 1], parts[0][i + 2:]] + parts[1:]
    return parts or [text]


class Handler(AsyncEventHandler):
    """Plain `synthesize` (whole text) and the streaming protocol (synthesize-start / -chunk… / -stop): each finished
    sentence is spoken as soon as it is complete, so HA can start playback after the first one."""
    def __init__(self, info, engine, default_voice, speed, *a, **k):
        super().__init__(*a, **k)
        self.info, self.engine, self.default_voice, self.speed = info, engine, default_voice, speed
        self.streaming = False; self.buf = ""; self.voice = default_voice; self.t0 = 0.0; self.first = None; self.said = ""

    async def speak(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        for part in chunks(text):
            audio, sr = await loop.run_in_executor(None, partial(self.engine.create, part, voice=self.voice, speed=self.speed, lang="en-us"))
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
            if self.first is None:
                self.first = time.perf_counter() - self.t0
            step = RATE * 2 // 5
            for i in range(0, len(pcm), step):
                await self.write_event(AudioChunk(rate=RATE, width=2, channels=1, audio=pcm[i:i + step]).event())
        self.said += " " + text

    def pick_voice(self, v) -> str:
        return (v.name if v and v.name in VOICES else None) or self.default_voice

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if SynthesizeStart.is_type(event.type):
            st = SynthesizeStart.from_event(event)
            self.streaming, self.buf, self.voice, self.t0, self.first, self.said = True, "", self.pick_voice(st.voice), time.perf_counter(), None, ""
            await self.write_event(AudioStart(rate=RATE, width=2, channels=1).event())
            return True
        if SynthesizeChunk.is_type(event.type) and self.streaming:
            self.buf += SynthesizeChunk.from_event(event).text
            # speak every complete sentence now; keep the unfinished tail
            m = list(re.finditer(r"[.!?](?:\s+|$)", self.buf))
            if m:
                cut = m[-1].end()
                ready, self.buf = self.buf[:cut], self.buf[cut:]
                await self.speak(ready)
            return True
        if SynthesizeStop.is_type(event.type) and self.streaming:
            if self.buf.strip():
                await self.speak(self.buf)
            self.buf = ""
            await self.write_event(AudioStop().event())
            await self.write_event(SynthesizeStopped().event())
            _LOGGER.info("%s | streamed | first audio %.2fs, total %.2fs | %r", self.voice, self.first or 0, time.perf_counter() - self.t0, self.said.strip()[:80])
            self.streaming = False
            return True
        if Synthesize.is_type(event.type):
            if self.streaming:            # streaming clients also send the whole text for old servers: ignore it
                return True
            syn = Synthesize.from_event(event)
            self.voice, self.t0, self.first, self.said = self.pick_voice(syn.voice), time.perf_counter(), None, ""
            await self.write_event(AudioStart(rate=RATE, width=2, channels=1).event())
            await self.speak(syn.text)
            await self.write_event(AudioStop().event())
            _LOGGER.info("%s | first audio %.2fs, total %.2fs | %r", self.voice, self.first or 0, time.perf_counter() - self.t0, syn.text[:80])
            return False
        return True


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="tcp://0.0.0.0:10340")
    p.add_argument("--model", default="/opt/casa-tts/models/kokoro-v1.0.onnx")
    p.add_argument("--voices", default="/opt/casa-tts/models/voices-v1.0.bin")
    p.add_argument("--voice", default="af_bella")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    engine = Kokoro.from_session(ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"]), a.voices)
    engine.create("Warm up.", voice=a.voice, speed=a.speed, lang="en-us")
    attr = Attribution(name="Kokoro (hexgrad) via kokoro-onnx", url="https://github.com/thewh1teagle/kokoro-onnx")
    info = Info(tts=[TtsProgram(name="casa-tts", description="Kokoro local TTS", attribution=attr, installed=True, version="1.1",
                                supports_synthesize_streaming=True,
                                voices=[TtsVoice(name=v, description=d, attribution=attr, installed=True, version="1.0", languages=["en"])
                                        for v, d in VOICES.items()])])
    _LOGGER.info("casa-tts ready on %s (default voice %s, %d threads)", a.uri, a.voice, a.threads)
    await AsyncServer.from_uri(a.uri).run(partial(Handler, info, engine, a.voice, a.speed))


if __name__ == "__main__":
    asyncio.run(main())
