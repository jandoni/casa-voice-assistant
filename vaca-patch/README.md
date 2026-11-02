# VACA patches

The Echo Shows run LineageOS and the [View Assist Companion App](https://github.com/msp1974/ViewAssistCompanionApp)
(VACA, Apache-2.0) as Home Assistant satellites. These patches change a decompiled VACA build (apktool smali) plus two
small Java helpers of mine. Each build is named "casaN" and contains all the earlier ones. Nothing here ships VACA
itself: you patch the APK you already have.

| Build | Change | Why, and what it measured |
|---|---|---|
| casa1 (`patch_preload.py`) | `SoundEffectsPlayer.preload()` is a no-op | The first wake blip after a quiet spell was silent. VACA keeps one prepared ExoPlayer per sound and replays it; a replay that reuses that player's AudioTrack produced no sound on these devices (logcat: "Started wake word sound" with no new `OpPlayAudio … usage:5` track, even 14 s after a preload). With no preload every blip gets a fresh player (~0.1 s extra). First blip after ~5 h idle: heard |
| casa2 (in `patch_preload.py`) | no mic mute during the blip | VACA muted the mic from the wake word until the blip ended + 200 ms and sent the audio before that at 10 % volume, so the first 1–2 words were lost. Voice lab: exact transcripts **0 of 23 to 15 of 24**, mean WER **0.47 to 0.11** |
| casa3 (in `patch_preload.py`) | barge-in | A wake word said while the Echo was still answering was ignored until the old pipeline ended. Now any stage from STARTING_TTS on starts a new request and the old voice stops |
| casa4 | AGC noise gate | For far-field speech, VACA's software AGC only boosts audio above a gate of clamp(floor + 6 dB, −65, −20 dBFS); a voice across the room arrived below −65 dBFS and stayed at hiss level. casa4 moved the limits to −85 / −60 dBFS. It helped the Echo Show 8 but made a Show 5 on a beamforming ROM boost room hum into false wakes, so it stayed on the Show 8 only. (Two constants in the AGC constructor; no script here) |
| casa5 (`CasaVerify.java`) | wake verifier hook | `MicrophoneInput.readBytes()` feeds `CasaVerify.pushShorts()` (2 s ring buffer); `onWakeWordDetected()` calls `CasaVerify.verify(score)`, which POSTs the clip to `wake-verify/` and drops the wake on "reject". Fail-open, 150 ms connect / 250 ms read. Detection to pipeline start went from ~0.10 to 0.15 s; the verifier answers in 10–14 ms |
| casa6 (`CasaVolume.java`) | volume keys reported instantly | VACA watches `Settings.System`, which Android writes 0.5–1 s after a change, so the volume keys reached HA ~1.3 s late. A `VOLUME_CHANGED_ACTION` receiver runs the same observer at once, only for a real change of the music stream (reacting to every broadcast double-counted presses). Key to HA: under 10 ms |
| casa7 (`patch_norevoke.py`) | never revoke the HA login | When a token refresh failed for any reason (a Wi-Fi blip), VACA revoked its own still-valid refresh token and the Echo fell back to HA's login page for good. `safeRevokeSession()` is now a no-op; a token HA really rejects (`invalid_grant`) is still cleared |

## Build (redo after every VACA update)

```bash
brew install apktool openjdk
ADB=~/Library/Android/sdk/platform-tools/adb; BT=~/Library/Android/sdk/build-tools/36.0.0
$ADB -s <echo>:5555 pull $($ADB -s <echo>:5555 shell pm path com.msp1974.vacompanion | cut -d: -f2 | tr -d '\r') orig.apk
apktool d -f -o src orig.apk
python3 vaca-patch/patch_preload.py src          # casa1-3
python3 vaca-patch/patch_norevoke.py src         # casa7
# casa5/6: set URL_BASE in CasaVerify.java, then
#   javac --release 8 -cp <android-sdk>/platforms/android-36/android.jar CasaVerify.java CasaVolume.java
#   d8 --min-api 26 com/casa/*.class; apktool d the resulting dex; copy smali/com/casa/*.smali into src/smali/com/casa/
#   and add the invoke-static hooks described in the table (readBytes, onWakeWordDetected, VolumeObserver.register)
apktool b -o unsigned.apk src
$BT/zipalign -f -p 4 unsigned.apk aligned.apk
$BT/apksigner sign --ks ~/.android/debug.keystore --ks-pass pass:android --out vaca-casa.apk aligned.apk
```

## Install (keeps pairing and settings)

The patched APK has a different signature from the official one, so the first install needs an uninstall; later
casa builds share one signature and install with `adb install -r`.

1. On the Echo: Settings, Security, Device admin apps, View Assist Companion, Deactivate (otherwise uninstall fails with
   `DELETE_FAILED_DEVICE_POLICY_MANAGER`).
2. Back up the app's data (the app is debuggable, so `run-as` works):
   `adb shell "run-as com.msp1974.vacompanion sh -c 'cd /data/data/com.msp1974.vacompanion && tar cf - --exclude=./lib --exclude=./cache --exclude=./code_cache .'" > data.tar`
3. `adb uninstall com.msp1974.vacompanion && adb install -g vaca-casa.apk`
4. Restore: push `data.tar` to `/data/local/tmp`, `run-as … tar xf`, delete the tar.
5. Re-apply: `appops set … WRITE_SETTINGS / REQUEST_INSTALL_PACKAGES / TURN_ON_SCREEN allow`,
   `dumpsys deviceidle whitelist +com.msp1974.vacompanion`,
   `dpm set-active-admin com.msp1974.vacompanion/.VACADeviceAdminReceiver`,
   `cmd package set-home-activity com.msp1974.vacompanion/.MainActivity`, then start `.MainActivity`.
6. Check: HA shows `assist_satellite.vaca_*` idle, logcat has "Starting MICROWAKEWORD wake word engine".

An official VACA update can't install over a patched build (signature): rebuild from the new release, or drop a patch
once upstream fixes the same thing.

## Gotchas

- VACA caches a custom wake-word model by name on the device and does not re-download it when HA's file changes: push
  the new `.tflite`/`.json` with `adb push` + `run-as … cp` into `files/custom/wakewords/microwakeword/`, then switch
  the wake word away and back.
- Never run `dumpsys media.audio_flinger` on these Echos: it killed audioserver and left the speaker silent until a
  reboot.
- The Echo's speaker amplifier sleeps on digital silence. A quiet 25 Hz keep-alive track on the media stream and a
  0.2 s lead-in on the wake sounds stop the first syllable of an answer from being lost.
