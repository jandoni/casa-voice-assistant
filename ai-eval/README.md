# ai-eval: benchmarks and checks

Everything here talks to Home Assistant or to the Wyoming / Ollama servers. Export `config.env` first
(`set -a; . ./config.env; set +a`; see `config.env.example`). Wyoming scripts need `pip install wyoming`; the
websocket ones need `pip install websockets`.

| Script | What it measures |
|---|---|
| `stt_bench.py` | Wyoming STT servers: latency after the end of audio and WER on 32 clean + 32 far-field clips. `gen` builds the clips with macOS `say` + ffmpeg (they are not committed), `run HOST:PORT --realtime --set all`, `config` / `compare` switch the HA Whisper add-on presets over ssh |
| `tts_bench.py` | Wyoming TTS servers: time to first audio, total time, × realtime; `compare` walks a list of Piper voices |
| `llm_bench.py MODEL` | Ollama speed on a ~2.7k-token HA-like prompt: cold prompt reading, warm generation, and a trick question |
| `tool_probe.py MODEL…` | Tool calling with HA-style tool schemas on 6 requests, including one that must not call a tool |
| `ha_agent_eval.py AGENT` | 10 real commands through HA's conversation API; checks the light really changed, then restores it. Set `EVAL_LIGHT_CEILING` / `EVAL_LIGHT_PENDANT` to two lights in an empty room |
| `ha_pipeline_trace.py "text"` | Runs the pipeline from the intent stage and prints the model's raw output per turn: text, tool calls, tool results, thinking |
| `match_test.py` | Dry-runs sentences through HA's matcher (`conversation/agent/homeassistant/debug`): which intent and slots, nothing executed |
| `voice_selftest.py` | End to end without a human: a speaker next to the Echo plays "wake word + command" (macOS `say`), then the script checks what the pipeline heard, what it said, and whether the device changed. Edit `TESTS` for your devices |
| `voice_log.py N` | The last N voice turns: what the Echo heard, who answered (local or LLM + tools), what it said, stage timings |
| `assist_learn.py poll\|report` | The learning loop: every phrase that still reached the LLM, with what the LLM did, most frequent first. Each one becomes an alias or a sentence template |
| `make_chat_model.py` | Builds the "chat" Ollama preset: gemma3:4b with brevity rules injected into the first turn |
| `ha_ws.py` | Helper: run raw HA websocket commands. It maps `data` to `service_data` in `call_service`; HA silently ignores `data`, which made my first threshold tests change nothing |
