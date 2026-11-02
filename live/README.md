# Casa Live: Gemini Live as the voice

One realtime speech-to-speech model instead of the chain casa-stt, canonicalizer, Claude, Kokoro: open conversation,
a natural voice, better with accents. Since 2026-10-24 all three Echos use this pipeline; the older "Casa AI" chain is
still installed and one command away per room.

## Pieces

| Piece | Where | What |
|---|---|---|
| `gemini_live` v1.0.11 | [matt123p/ha-gemini-live](https://github.com/matt123p/ha-gemini-live) (MIT), plus `gemini_live-1.0.11-casa.patch` | STT + conversation + TTS entities that stream the Echo's mic to Gemini Live and play its voice back |
| `casa_live` | `custom_components/casa_live/` | My own small LLM API, "Casa house tools", given to Gemini instead of HA's big "Assist" API: `house` (any home command, run through HA's LOCAL agent with the Echo's device id, so timers ring on that Echo, room-relative commands, recipes, music and weather cards keep working), plus chore and grocery tools. The prompt carries the time, the room you talk from and a ~150-token live snapshot (inside temperature, thermostat, lights on, what's playing, timers). Also tracks the monthly cost from Gemini's usage metadata |
| `casa_live_prompt.txt` | here | Persona and rules (short answers, English only, never act on a fragment, silent on TV talk). Placeholders in `<>` |
| `setup_gemini_live.py` | here | Creates or updates the Gemini Live entry (model gemini-3.8-live, voice Aoede, transcripts on, 30 s session timeout, tools = casa, search grounding off) and the "Casa Live" pipeline. `--assign <room>` / `--revert <room>` move one Echo between Casa Live and Casa AI |

Why my own tool API: Assist sends every exposed entity and ~20 intent tools with every request (thousands of tokens,
re-billed on each turn of a Live session) and bypasses the local logic I had already built.

## Measured

- First reply audio **0.05–0.25 s after the audio ends** for questions; about **1.6 s for a house action** (the tool
  call comes first).
- With Gemini's own end-of-speech and the "replied" signal below, the turn ends **1.4 s** after the last word of a
  question ("What is the capital of Australia?"), 1.8 s for a tool question. Before that fix every Echo turn sat in
  "listening" for the 10 s safety net (9–23 s per turn in the history).
- Cost estimate: **$7–10 a month** at our usage; all testing up to the first week cost about $0.24.
- Bedroom Echo end to end with synthetic question clips, while the manual-activity patch was in: 7 of 10 complete and
  correct answers, 3 cut at a deliberate 1.6 s mid-sentence pause, 0 wrong actions.

## Local patches (`gemini_live-1.0.11-casa.patch`)

Apply on top of upstream v1.0.11: `cd <config> && patch -p1 < gemini_live-1.0.11-casa.patch`.

1. `runtime.py`: `AUDIO_STREAM_MAX_BUFFER_BYTES` 6400 to 2,000,000. With the tiny buffer the receive loop paused after
   ~0.13 s of reply audio while HA 2026.9 waited for more text before reading the audio, so every short answer stopped
   after the first word and hung 30 s.
2. `stt.py`: Gemini's input transcription goes out as `casa_live_text` events (live words on the Echo screen).
3. `stt.py`: mic closed and no words heard for 2 s: end the turn now with a silent marker ("…") and SUCCESS instead of
   ERROR (an ERROR became `stt-stream-failed`, which VACA answers with an error sound and message).
4. `stt.py`: `casa_gemini_replied` on Gemini's first tool call, audio or text of a turn (Gemini's own end-of-speech has
   fired); `casa_endpoint` closes the mic on it.
5. `stt.py`: the first ≤0.5 s of a turn is silenced when it is speech followed by a quiet gap: the tail of the wake word
   plus the pause before the question made Gemini answer early and cut the turn mid-sentence.
6. `conversation.py`: re-open the mic only when the answer ends with "?" (the integration always continued, leaving a
   30 s listening window that swallowed the next request); not even then on a follow-up turn or with the TV on when
   the "question" is just "could you repeat?".
7. `conversation.py`: drop a turn silently when it is not meant for us: another script or Spanish punctuation, more
   than half unmistakably Spanish words, "shut up / never mind / not you", or a single non-command word while the TV
   plays. A false wake on Korean TV dialogue once chained four turns of "could you repeat?"; this ended that.
8. `gemini.py`: a `USAGE_HOOK` for token usage (casa_live turns it into `sensor.gemini_live_cost`). Notes in the code
   record what was tried and removed: `language_code: en-US` made the native-audio model deaf (8 of 10 runs without a
   transcript), and manual activity detection with a fixed 1 s pause cut long questions; Google's default automatic
   activity detection is what runs now.

Also in `casa_live`: `assist_pipeline.pipeline.STREAM_RESPONSE_CHARS = 0`, so HA starts speaking on the first words of
a reply instead of after 60 characters or the whole answer.

## Gotchas

- Free-tier keys have no quota for the live model ("You exceeded your current quota"): billing must be on. Set a
  budget alert.
- Google Search grounding must stay off for now: with it on, audio turns are cut (ha-gemini-live issue #13).
- Re-apply the patch after every `gemini_live` update.
- One specific synthetic recording repeatedly made Google's server fail with 1011; other wordings of the same question
  were fine.
