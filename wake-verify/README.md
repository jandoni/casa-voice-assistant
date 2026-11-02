# Wake verifier

Second stage of the wake word. The on-device engine (microWakeWord in VACA) is small and fast but fooled by door
slams and TV dialogue. The patched VACA (`vaca-patch/CasaVerify.java`) keeps the last 2 s of microphone audio in a
ring buffer and, the moment its wake word fires, POSTs it here and waits for `accept` or `reject`
(150 ms connect, 250 ms read; on any error or timeout the Echo goes ahead, so the verifier can only remove wakes).

```
POST /verify?score=0.87   body: 2 s of 16 kHz mono PCM16   ->   "accept" | "reject"
```

## The checks, in the order they were added

| Check | What it rejects | Measured |
|---|---|---|
| Voicing gate (2026-09-29) | clips with under 60 ms of voiced (pitched 70–400 Hz, loud enough) speech: door slams, thuds, a spoon | 0 of 86 real wakes rejected; 17 of 26 false wakes had under 60 ms; ~3 ms per clip |
| Room-pair / TV guard (2026-10-01) | the same sound waking two rooms within 1.5 s at similar levels (under 6 dB apart) is a broadcast; with a clear gap only the louder Echo answers. While the TV plays, the rooms that only hear it through the walls need a score of 0.97 | stopped two nights of TV-dialogue false wakes in empty rooms |
| Word check (2026-11-01) | transcribes the clip with casa-stt's Kroko streaming model ("ALEXA" hotword, about 80 ms, run during the 100 ms hold) and wants an "Alexa"-like word; other words reject, no words reject only while the TV plays | synthetic "Alexa" 360 of 360 accepted, real "Alexa" 3 of 3, the TV false wakes from one film 6 of 6 rejected (0 of 9 accepted in the offline set), 0 false hits in about 4,500 other clips |
| Whisper rescue (2026-11-02) | (inverse) a wake the voicing gate rejected is accepted when the word check hears "Alexa": whispered wakes have no pitch | 3 of 4 real whispered wakes rescued, 0 of the thud and noise rejects |

Every accepted wake also reports its speech and noise level to HA (`casa_wake_level`), which raises that Echo's answer
volume for a far or noisy speaker. Every clip is saved with its verdict and reason in the file name
(`<utc>_<room>_<score>_v<voiced ms>_<verdict>[_<why>].wav`): that folder became the training and evaluation data.
Those are recordings from inside the home, so they stay on the voice host.

Kill switches: `touch /opt/casa-stt/wake_verify.off` accepts every wake again; `touch /opt/casa-stt/wake_asr.off`
turns off only the word check. If the wake word changes, update `WAKE_RE`, `WAKE_WORD` and the hotword.

## Running it

It runs in casa-stt's venv and reads `/opt/casa-stt/casa-stt.env` (HA token, `CASA_WAKE_ROOMS`, `CASA_TV_ENTITY`,
`CASA_SOUNDBAR_ENTITY`, `CASA_TV_GUARD_ROOMS`; see `stt/casa-stt.env.example`). The word check reuses the Kroko model
and the `kroko.bpe.vocab` that casa-stt writes on its first start.

```bash
cp wake_verify.py /opt/casa-stt/ && cp wake-verify.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now wake-verify
journalctl -u wake-verify -f      # one JSON line per wake: room, score, verdict, why, voiced ms, levels, transcript
```

## Offline evaluation (`eval/`)

- `kws_eval.py`: would a dedicated keyword spotter (sherpa-onnx KWS zipformer, gigaspeech 3.3M) do better? Runs it
  over positive and negative sets for a grid of boost scores and thresholds; prints recall, false alarms and ms per
  clip. It also defines the sets the other script uses.
- `asr_eval.py`: the approach that shipped. Transcribes every clip with Kroko (or Parakeet) and counts "Alexa"-like
  words.

The evaluation sets, described rather than published: synthetic "Alexa" clips (10 Kokoro voices × 3 speeds × 3
phrasings × 4 noise conditions), a handful of real "Alexa" wakes, the TV false wakes of one night, older verifier clips,
silent-room captures from three rooms, synthetic sound-alikes ("Alex", "Alexander", "Lexus", "election"), everyday
sentences, other wake phrases, and real commands recorded by the Echos. About 4,500 negative clips in all.

I also compared a vendor's proprietary on-device wake engine on the same recordings. Nothing from that comparison is
in this repo.
