# Speech-to-text / text-to-speech benchmarks (voice pipeline)

Register of every engine I tried for the local voice pipeline, so new candidates can be compared against the same
clips. Rows only get added, never deleted.

## How the benchmark works

- Tools: `ai-eval/stt_bench.py`, `ai-eval/tts_bench.py` (need `pip install wyoming`; a venv works:
  `python3 -m venv venv && venv/bin/pip install wyoming`).
- Clips: `ai-eval/stt_clips/{clean,noisy}/p*.wav`, generated with `stt_bench.py gen`: 16 phrases × 2 macOS voices
  (Samantha, Daniel) = 32 clips, each with 0.8 s trailing silence. *noisy* = −8 dB, room echo, pink noise
  (roughly the Echo Show's far-field mic). The clips are not in the repo; `gen` rebuilds them in a minute. Two of the
  16 sentences in the published list differ from the measured set (the originals named a person and my city), so a
  rerun is close to, not identical with, the numbers below.
- **Latency** = seconds from the last audio byte sent until the transcript arrives, which is what you feel after you
  stop talking. (The HA pipeline's "stt" timing also counts the seconds you are speaking and the end-of-speech
  silence wait, so it is always larger.)
- **WER** = word error rate after normalising case, punctuation, `%` to percent and a few number words.
- Targets (2026-09-20): under 1 s, ideally under 0.5 s, without losing accuracy.
- Whisper add-on (`core_whisper`) presets are switched by the script through the Supervisor API
  (`stt_bench.py config '<json>'`; `compare` runs every preset in `PRESETS`). The add-on's Wyoming ports (10300 STT,
  10200 TTS) were published on the HA host for this.

## STT results: HA VM (4 vCPU of the Core Ultra mini PC), core_whisper 3.5.3, 2026-09-20

| Engine / preset | Latency avg (p90) clean | Latency avg (p90) noisy | WER clean | WER noisy | Notes |
|---|---|---|---|---|---|
| **parakeet-tdt-0.6b-v3** (`model: auto`, `stt_library: sherpa`), in use at the time | 0.60 s (0.72) | 0.64 s (0.78) | 2.4 % | 3.8 % | Best balance. Writes "20%" and misspells our names (aliases cover them). Ignores beam/initial_prompt. |
| zipformer streaming (`sherpa_streaming: true`, Kroko 2025) | 0.17 s (0.22) | 0.20 s (0.23) | 12.3 % | 11.4 % | Transcribes while you talk. Numbers as words ("two thousand and twenty six"), garbles names ("plague shall out"), sometimes drops the last word. |
| whisper small-int8, beam 1 (previous default) | 1.13 s (1.49) | 1.05 s (1.07) | 3.8 % | 8.9 % | With beam 3 + long initial_prompt: 1.3 s. |
| whisper base-int8 | 0.43 s (0.45) | 0.45 s (0.46) | 7.0 % | 10.0 % | Fast but "Berraker Bama", "TANU F5". |
| whisper distil-small.en | 1.71 s (1.78) | 1.71 s (1.76) | 7.1 % | 7.6 % | Slower than small here. |
| parakeet + `vad_clip: true` | 0.92 s | 1.06 s | – | – | The extra VAD pass costs more than it saves. Off. |
| **Kroko zipformer streaming, `beam_size: 4`** (modified beam search) | 0.22 s (0.25) | 0.21 s (0.24) | 7.1 % | 7.3 % | Same model as above with beam search and spoken numbers scored correctly: the remaining misses are proper nouns ("Chillout" to "shall out", "Barack Obama") and an occasionally clipped last word ("kitch"). |
| NVIDIA FastConformer streaming transducer 480 ms (`custom_model: sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8`) | 0.59 s | 0.66 s | 29 % | 39 % | Drops the first word of most phrases in the add-on's streaming path ("is the c e o of nvidia"). Not usable as shipped. |
| NVIDIA FastConformer streaming 80 ms | 1.62 s | 1.73 s | 25 % | 36 % | Same problem, slower. |
| NVIDIA Parakeet-unified 0.6B streaming 240 ms (`…parakeet-unified-en-0.6b-int8-streaming-240ms`) | **38 s** | – | 16 % | – | 0.6B parameters cannot keep up with real time on 4 CPU threads; times out. |
| NVIDIA Nemotron Speech streaming 0.6B 160 ms (2026-04) | – | – | – | – | Fails to load in add-on 3.5.3 ("Protobuf parsing failed": newer ONNX than its onnxruntime). |

## casa-stt (my own two-stage server on the voice host, 12 cores), 2026-09-21

| Version | Latency avg (p90) clean | Latency avg (p90) noisy | WER clean | WER noisy | Notes |
|---|---|---|---|---|---|
| v1: Kroko beam 4 + hotwords 1.5 + Parakeet fallback (audio sent faster than real time) | 0.46 s (0.77) | 0.42 s (0.72) | 4.2 % | 4.0 % | matcher accepted partial matches; no name fixes |
| v2: + `match: true` only, number words to digits, name fixes, hotwords 2.0 | 0.61 s (1.17) | 0.62 s (1.13) | 0.5 % | 2.2 % | bench sent audio 10× real time, so the streaming stage lagged |
| **v3: + gain-normalised Parakeet input, measured with `--realtime`** | **0.27 s (0.54)** | **0.33 s (0.69)** | **0.5 %** | **1.6 %** | commands 0.05–0.15 s, questions 0.55–0.7 s; end to end through HA: a pendant on in 0.09 s |

Real voices changed the second stage. On 13 real recordings of my own (accented) voice, Kroko streaming got 8 of 13
wrong, Parakeet TDT returned nothing or nonsense on 6 of 13 ("What is Obama's height" came back empty), Whisper small
was wrong on the hard ones, and **whisper.cpp medium (q5_0) got all 13**. On the Arc iGPU with Vulkan it encodes a full
30 s context in 1.8 s, and in 0.3 s with `-ac 512`, so about 1.0–1.3 s per phrase; `-nf` (no temperature fallback)
avoids 10–30 s decoder loops. Since 2026-09-21 casa-stt's stage 2 is whisper.cpp medium; questions go to whisper small
(0.37 s instead of 0.84 s, the same text on my questions) and house commands keep medium (small turned "countertop
light" into "contact applied"). On 74 voice-lab captures: medium 0.26 WER, small 0.29, large-v3-turbo 0.51.

Not tested or not possible: Whisper on the Arc iGPU via OpenVINO inside the HA VM (needs `/dev/dri` there; the iGPU
belongs to the voice host); Speech-to-Phrase (fastest but closed vocabulary, so it cannot take questions); CUDA builds
(no NVIDIA). The idea that became casa-stt: run sherpa-onnx myself as a Wyoming server with **hotword biasing**
(device, room, playlist and people's names via sherpa-onnx `hotwords_file` + modified beam search) and proper tail
padding, plus a **two-stage rule**: return the streaming text at once when it matches a known Assist command
(HA `conversation/agent/homeassistant/debug`, ~50 ms), otherwise run the accurate engine on the buffered audio.

## TTS results: Piper add-on 2.5.2, 2026-09-20

| Voice | Time to first audio | Total (5.5 s of speech) | Speed | Notes |
|---|---|---|---|---|
| en_US-lessac-medium, in use at the time | 0.06–0.17 s (0.66 s cold) | 0.17 s | 24× realtime | Not a bottleneck; voices differ in taste, not speed. `tts_bench.py compare` tries the list in `VOICES`. |

## TTS: Kokoro "Bella" (casa-tts), 2026-09-25

Chosen by ear out of 13 voice samples. Kokoro 82M through kokoro-onnx on 6 CPU threads (the
fastest setting on this CPU; the iGPU stays free for the LLM and whisper): first audio about 0.45–0.5 s after the
answer text for a short sentence, streamed sentence by sentence, so a long answer starts as fast as a short one.
Piper was about 0.2 s.

## Pipeline budget after these changes (Echo Show 5 to HA, local command)

end-of-speech wait 0.25 s (VACA `finished_speaking_detection: aggressive`; default 0.7, relaxed 1.25)
+ STT 0.05–0.2 s for commands (casa-stt streaming) / 0.6 s for questions + local intent 0–0.3 s + TTS 0.1 s
+ playback start ≈ **0.5–0.8 s** after you stop talking for a command (was 1.0–1.3 s with Parakeet alone,
2.5 s with Whisper small and the default VAD). The aggressive end-of-speech setting was later reverted (it clipped
short commands); `custom_components/casa_endpoint` now picks the wait from the words heard so far.
LLM answers add the model time (see LOCAL_AI_FINDINGS.txt).
