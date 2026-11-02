#!/usr/bin/env python3
"""Patch a decompiled VACA (apktool output dir) so SoundEffectsPlayer.preload() is a no-op.

Why (2026-09-25): VACA keeps ONE prepared ExoPlayer per sound effect and replays it (seekTo(0)+play()). On the Echo
Shows a replay that reuses that player's existing AudioTrack is SILENT (logcat: "Started wake word sound" but no new
`AF::TrackHandle OpPlayAudio: track:N usage:5`) - even 14 s after a preload - so the first blip after a quiet spell
was lost; a blip with a freshly created track is always heard. With preload() a no-op nothing is cached and
SoundEffectsPlayer.play() takes its ad-hoc path: a new ExoPlayer (= new AudioTrack) per play, released at the end.

Usage: patch_preload.py <apktool-dir>
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1])
hits = list(root.glob("smali*/com/msp1974/vacompanion/players/SoundEffectsPlayer.smali"))
if len(hits) != 1:
    sys.exit(f"SoundEffectsPlayer.smali not found exactly once: {hits}")
p = hits[0]
s = p.read_text()
sig = ".method public final preload(Landroid/net/Uri;Lkotlin/coroutines/Continuation;)Ljava/lang/Object;"
if "CASA PATCH" in s:
    sys.exit("already patched")  # (re-run on a fresh apktool dir)
a = s.index(sig)
b = s.index(".end method", a)
s = s[:a] + sig + """
    .registers 3
    # CASA PATCH: never cache a prepared ExoPlayer for sound effects (stale AudioTrack = silent first blip)
    sget-object v0, Lkotlin/Unit;->INSTANCE:Lkotlin/Unit;
    return-object v0
""" + s[b:]
p.write_text(s)
print(f"patched {p}")


def patch_no_blip_mute(root):
    """v2 (2026-09-26, voice lab): VACA muted the mic from the wake word until the blip ended + 200 ms and streamed the
    audio before that at 10 % volume -> the first 1-2 words after "Okay Nabu" were lost (lab: 0/23 exact transcripts ->
    15/24 after this patch). Remove both: suppressMicFor(0) and silenceAudioBefore = 1 (stream everything)."""
    sat = next(root.glob("smali*/com/msp1974/vacompanion/satellite/Satellite.smali"))
    s = sat.read_text()
    m = ".method public final playWakeWordDetectionSound("
    a = s.index(m); b = s.index(".end method", a)
    body = s[a:b]
    assert body.count("const-wide/16 v6, 0xfa0") == 1
    s = s[:a] + body.replace("const-wide/16 v6, 0xfa0", "const-wide/16 v6, 0x0  # CASA PATCH v2: no mic mute during the blip") + s[b:]
    sat.write_text(s)
    cont = next(root.glob("smali*/com/msp1974/vacompanion/satellite/Satellite$playWakeWordDetectionSound$2.smali"))
    c = cont.read_text()
    assert c.count("const-wide/16 v1, 0xc8") == 1
    c = c.replace("const-wide/16 v1, 0xc8", "const-wide/16 v1, 0x0  # CASA PATCH v2: no trailing mic mute")
    old = "    invoke-static {}, Ljava/lang/System;->currentTimeMillis()J\n\n    move-result-wide v3\n\n    add-long/2addr v3, v1\n"
    assert c.count(old) == 1
    c = c.replace(old, "    const-wide/16 v3, 0x1  # CASA PATCH v2: stream all audio at full volume\n")
    cont.write_text(c)
    print("patched (v2) no blip mute")


patch_no_blip_mute(root)

def patch_barge_in(root):
    """v3 (2026-09-26): a wake word said while the previous exchange is still answering was ignored (VACA only starts
    a new pipeline once the old one is ENDED) -> "I have to wait ~3 s before it hears me again". Now any stage from
    STARTING_TTS (ordinal 8) on starts a new request; stop() of the old pipeline also stops its voice (barge-in)."""
    sat = next(root.glob("smali*/com/msp1974/vacompanion/satellite/Satellite.smali"))
    s = sat.read_text()
    a = s.index(".method public final handleWakeWordDetection("); b = s.index(".end method", a); body = s[a:b]
    old = ("    :goto_1\n    sget-object v4, Lcom/msp1974/vacompanion/satellite/PipelineStage;->ENDED:Lcom/msp1974/vacompanion/satellite/PipelineStage;\n\n"
           "    if-ne v0, v4, :cond_3\n")
    assert body.count(old) == 1
    new = ("    :goto_1\n    # CASA PATCH v3 (barge-in)\n    if-eqz v0, :cond_3\n\n"
           "    invoke-virtual {v0}, Lcom/msp1974/vacompanion/satellite/PipelineStage;->ordinal()I\n\n    move-result v4\n\n"
           "    const/16 v0, 0x8\n\n    if-lt v4, v0, :cond_3\n")
    sat.write_text(s[:a] + body.replace(old, new, 1) + s[b:])
    print("patched (v3) barge-in")


patch_barge_in(root)
