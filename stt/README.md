# casa-stt: two-stage speech-to-text for the voice pipeline

A Wyoming STT server (port 10310) that Home Assistant uses as `stt.casa_stt` in the "Casa AI" pipeline. It runs on
the voice host under systemd (`casa-stt.service`), from `/opt/casa-stt/` (venv, `casa_stt.py`, `hotwords.txt`,
`casa-stt.env`, `models/`).

## How it works

1. **While you talk** the audio streams into the **Kroko streaming Zipformer** (sherpa-onnx, modified beam search 4,
   **hotword biasing** towards the phrases in `hotwords.txt`: devices, rooms, aliases, playlists, people's names).
   0.8 s of silence is appended at the end so the last word is never clipped. Spoken numbers become digits
   ("twenty percent" to "20 percent"), and the `wrong => right` lines in `hotwords.txt` fix known spellings.
   A partial transcript goes to HA ~4 times a second as `casa_live_text` events (the Echo screen shows your words, and
   `casa_endpoint` uses them to decide when you have finished).
2. **The streaming text is checked against Home Assistant's own sentence matcher**
   (`conversation/agent/homeassistant/debug`, ~10–50 ms). If it is a **complete command** (`match: true`) it is
   returned immediately: **about 0.05–0.2 s after you stop talking**.
3. If it is not a command as heard, casa-stt tries to **repair** it before giving up. Each candidate is validated by
   the HA matcher, so a repair is only accepted when it becomes a real command:
   a. known confusions from the `~ wrong => right` lines ("like/lie/line" to "light", "tone of" to "turn off",
      "the moment" to "the monitor");
   b. the matcher's *unmatched* name or area slot replaced by the closest house phrase (difflib);
   c. generic vocabulary repair of out-of-vocabulary words.
   Example from the log: `'In the monitor lie to 20 percent'` became `dim the monitor light to 20 percent` (0.13 s).
4. **Stage 2, the accurate engine**, for questions and anything that did not match: **whisper.cpp `medium` on the Arc
   iGPU** (`whisper-server.service`, port 10320, Vulkan, `-ac 512 -nf -sns`, vocabulary prompt in `whisper.env`),
   about 1.0–1.5 s per phrase. Unlike Parakeet and Kroko it understands my accent (Parakeet returned nothing or
   nonsense on 6 of 13 real recordings; whisper medium got all 13). When the streaming text already looks like a
   question, whisper `small` answers instead (`whisper-fast.service`, port 10322, 0.37 s vs 0.84 s). The audio
   context is sized per request, so short commands stay fast and turns up to 30 s still fit; longer turns are cut at
   their quietest point. Very quiet audio (RMS < 0.001) is treated as silence so whisper cannot hallucinate on it, and
   whisper's classic hallucinations ("thank you for watching", its own prompt) are dropped.
   Parakeet TDT 0.6B v3 on the CPU is still available (`CASA_STT_OFFLINE_ENGINE=parakeet`), and `hybrid` runs
   Parakeet first and only asks whisper when Parakeet and the streaming model disagree (three-way vote).
5. **Not understood: instant local reply.** If nothing matches after repair and the text is command-shaped
   ("turn/switch/set/dim…") and looks like a house command, casa-stt returns the sentinel sentence
   `casa did not understand`, which HA answers locally in 0.1 s ("Sorry, I didn't catch that. Could you say it
   again?"; `custom_sentences/en/casa_voice.yaml` + `homeassistant/packages/casa_voice.yaml`). Fragments under 3 words
   are dropped. Everything else, including long free-form sentences that happen to contain "on" or "kitchen", goes on
   to the LLM.
6. **Several commands in one sentence** ("turn off the monitor light and set the thermostat to 20") are accepted when
   every part is a known command on its own; the patched canonicalizer then runs the parts in parallel.

Measured (`ai-eval/stt_bench.py`, 64 clips, audio sent in real time): latency avg 0.27 s clean / 0.33 s noisy
(commands 0.05–0.15 s, questions 0.55–0.7 s), **WER 0.5 % clean / 1.6 % noisy** (Whisper small in the HA add-on was
1.1–1.5 s and 4–9 %; Parakeet alone 0.6 s and 2.4–3.8 %). Over two days of real use, 757 turns: stream fast path
median 0.03 s; whisper path median 1.04 s, p90 1.77 s.

## Far-field audio

Every Echo turn starts with the Echo's own wake chime recorded by its own mic (about −15 dBFS), far louder than a
voice from across the room (−40 to −50 dBFS). Scaling by the loudest peak meant scaling by the chime. `normalize()`
ducks the chime when it is clearly louder than the rest and brings the loud frames of what remains (the speech) to
about −20 dBFS (at most +30 dB, soft limiter); the streaming model gets a running gain instead. Simulated sofa distance
(voice 20–28 dB below a full-level chime): 4 of 6 understood before, 6 of 6 after (5 of them instantly from the stream).

## Files

| File | What |
|---|---|
| `casa_stt.py` | the server |
| `make_hotwords.py` | builds the plain phrases of `hotwords.txt` from HA's entity names (+ optional alias files) |
| `hotwords.txt.example` | the file format: phrases with scores, `wrong => right` fixes, `~ wrong => right` repair rules |
| `casa-stt.env.example`, `whisper.env.example` | environment files (copy without `.example`, fill in, never commit) |
| `casa-stt.service`, `whisper-server.service`, `whisper-fast.service` | systemd units |
| `ollama_guard.py`, `ollama-guard.service` | proxy in front of Ollama (:11435 to :11434) that caps `num_predict` at 400, so a degenerate generation can never freeze a satellite for minutes |
| `ollama-health.sh`, `.service`, `.timer` | every 5 min: ask Ollama for "pong"; two bad replies in a row restart it (after an iGPU hang it answers 200 with empty content). Skips while a real request is open |
| `ollama-keepwarm.sh` | after an Ollama restart, asks HA to re-warm the Casa prompt (a cold house prompt costs ~40 s) |

## Running it

```bash
# on the voice host
python3 -m venv /opt/casa-stt/venv
/opt/casa-stt/venv/bin/pip install sherpa-onnx wyoming numpy aiohttp
# models from the sherpa-onnx asr-models GitHub release, unpacked into /opt/casa-stt/models/:
#   sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06 (70 MB)
#   sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (650 MB, only for --offline-engine parakeet|hybrid)
# whisper.cpp built with -DGGML_VULKAN=1 in /opt/whisper.cpp, models ggml-medium-q5_0.bin and ggml-small-q5_1.bin
cp casa_stt.py /opt/casa-stt/; cp hotwords.txt.example /opt/casa-stt/hotwords.txt
cp casa-stt.env.example /opt/casa-stt/casa-stt.env; cp whisper.env.example /opt/casa-stt/whisper.env   # then edit both
cp *.service *.timer /etc/systemd/system/ && systemctl daemon-reload
systemctl enable --now whisper-server whisper-fast casa-stt
journalctl -u casa-stt -f          # one line per utterance: stage, timings, text, what the stream heard
```

In HA: add a Wyoming integration entry pointing at `<voice host>:10310` and pick `stt.casa_stt` in the pipeline.
Regenerate the hotword phrases after renaming devices or playlists:
`set -a; . ./config.env; set +a; python3 stt/make_hotwords.py`, then add your `=>` and `~` lines back.
Benchmark: `python3 ai-eval/stt_bench.py run <voice host>:10310 --realtime --set all` (without `--realtime` the
streaming stage looks slower than it is).

Tunables (`casa-stt.env`): `CASA_STT_HOTWORDS_SCORE` (2.0), `CASA_STT_BEAM` (4), `CASA_STT_STREAM_THREADS` (4),
`CASA_STT_OFFLINE_THREADS` (8), `CASA_STT_OFFLINE_ENGINE` (whisper). `--always-offline` disables stage 1.

## Gotchas

- The Kroko tarball has no `bpe.vocab`, so `casa_stt.py` derives one from `tokens.txt` for the hotword encoder
  (`modeling_unit=bpe`).
- The HA matcher's `match` flag must be true: a bare `HassGetState` or an unmatched-slot result is how questions look.
- Sending audio faster than real time makes the streaming stage look slow (use `--realtime` in the bench).
- Don't benchmark against the live server while someone is using the voice assistant: the bench loads all 12 cores.
- A slot repair must not bend a question onto a device name ("what is Obama's height" is not a light): questions use
  a stricter similarity cutoff (0.85 instead of 0.66).
- `CASA_STT_RECORD_DIR` keeps the last N utterances as WAV for debugging. Those are recordings of your home; they stay
  on the voice host.
