# Voice lab

Two ways to test the voice assistant without a person: replay synthetic clips at a real Echo through a speaker next to
it and score wake + transcript, or run wake-word models offline over WAV files. In lab mode
(`input_boolean.casa_voice_lab`) the patched canonicalizer only reports what it heard (`casa_lab_result`) and does
nothing, so a lab run never switches a light.

## Scripts

| Script | Where it runs | What |
|---|---|---|
| `make_clips.py` | voice host, casa-tts venv | 12 phrases × 6 Kokoro voices (US m/f, UK, Spanish, Italian, Indian accents) × 4 backgrounds (none, keyboard, fan, TV chatter at ~10 dB SNR) = 288 clips, each wake word + phrase ("\|" = a 1.6 s pause mid-question), plus `false_reel.mp3` (3.7 min of TV-style talk with near-miss words and no wake word). Copy the output to HA's `/config/www/voicelab/` |
| `run_lab.py` | laptop, `config.env` exported | Plays the clips on the speaker, waits for the satellite to go "listening" and for the lab result, scores WER and timings, appends to `results/results.jsonl`. `--match`, `--volume`, `--gain` / `--ns` (VACA mic gain / noise suppression), `--false-reel`, `--gap`, `--wake-only`. Rooms come from `VOICELAB_ROOMS` |
| `make_wake_clips.py` | voice host | Offline wake-word set, 16 kHz WAVs: `pos_<word>__<voice>__<speed>__<noise>__<n>.wav` (10 Kokoro voices × 3 speeds × 3 phrasings × none/fan/tv/keyboard) and `neg_*.wav` (everyday sentences full of near-miss words, ~15 min) |
| `eval_wake.py` | laptop venv with pymicro-wakeword (the same engine VACA runs) | Streams the WAVs like the Echo does: hit rate per threshold and false wakes with a 2 s cool-down |
| `make_coco_train.py` | voice host | Training data for a custom "Hey Coco" model: 47 confusables ("cocoa", "coconut", "kokomo", "echo", "hey cool", "okay nabu", "toucan", "taco"…) × 29 training-only Kokoro voices × 3 carrier sentences as hard negatives (~4k clips), and accent-matched positives. "hey cocoa" is deliberately not a negative: it sounds the same |

## What the lab found

| Test | Result |
|---|---|
| Baseline, quiet, UK + Spanish-accent voices | 23/24 woke, **0/23 exact transcripts**, mean WER 0.47: the first 1–2 words were always missing |
| Cause | VACA muted the mic from the wake word until the blip ended + 200 ms (≈1–1.2 s) |
| VACA patch casa2 (no blip mute) | 24/24 woke, **15/24 exact**, mean WER **0.11** |
| Pause mid-question (1.6 s) | cut at the pause before `casa_endpoint`; complete after |
| Keyboard / TV noise at ~10 dB SNR | wake word mostly fails with TTS voices; fan noise fine |
| False wakes, 3.7 min of TV talk | 0 |
| Whisper offline on 74 lab captures | medium 0.26 WER (kept) · small 0.29 · large-v3-turbo 0.51 |
| Wake threshold 0.7 vs 0.6 (32 clips, 4 accents, quiet + fan) | 0.7: 25–27/32 (Indian-accent voice 4/8) · **0.6: 32/32**, 0 false wakes in 2 × 3.7 min of TV talk |
| Long real questions (12 s / 15 s) | were cut to 8 s by casa-stt, a ~10 s whisper window and HA's 15 s limit: full text after the fix |
| US Kokoro voices | never trigger "Okay Nabu" (a TTS artefact): use UK or accented voices |
| Back-to-back wakes ~15 s apart | the on-device detector goes deaf for minutes; with 45 s gaps 27/32 woke. Test with `--gap 45` |
| A newer LineageOS build with the Echo's beamforming and echo cancellation | woke 11/12 while the Echo itself played music vs **0/12** on the old build; 20/20 at normal volume vs 17/20 |

## Wake-word models, offline

| Model | hits @0.6 / 0.7 | false wakes in 15.4 min of talk @0.6 / 0.7 / 0.9 |
|---|---|---|
| okay_nabu (stock) | 80 % / 74 % (TV noise hardest) | 1 / 0 / 0 |
| a downloaded community "hey_coco" | 100 % / 100 % | 63 / 58 / 39: fires on "cocoa", "echo", "go go", "hey cool", "Kokomo", "coffee", "toucan" |

So I trained my own "Hey Coco" with microWakeWord (TaterTotterson's Apple-silicon trainer), seven versions over four
days. Hard negatives were the main lever against false wakes; recordings through the Echo's own microphone, also
attenuated by 8 and 14 dB, were the lever for distance.

| Test (never seen in training), threshold 0.6 | v2 | v3 | v4 | **v5** | okay_nabu |
|---|---|---|---|---|---|
| Echo-mic held-out real clips (26) | 15 | 24 | 20 | **25** | – |
| real held-out clips (10) | 10 | 10 | 10 | **10** | – |
| synthetic other voices (360) | 305 | 359 | 325 | 242 | 306 (its own 360) |
| false wakes: everyday talk 15 min + 300 tricky clips | 0 | 31 | 5 | **0** | 9 |
| false wakes: Echo-mic tricky reel + TV reel (95) | 2 | 8 | 0 | **0** | – |

v5 was tuned to our two voices: near-perfect at every distance with zero false wakes, but other (synthetic) accents
dropped to 45–65 %. v7 added 1,620 silent-room captures as negatives: false wakes on held-out room audio 4 vs 19
(253 clips) at the same recall on our voices. Later the Echos went back to "Okay Nabu" and, since 2026-11-01, to
microWakeWord's official "Alexa" model, with the word check in `wake-verify/` removing its TV false wakes.

The models, the clip lists and the results files are not published: the training and test sets include recordings of
real people in our home.
