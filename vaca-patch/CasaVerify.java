package com.casa;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Two-stage wake word (casa5, 2026-09-28). VACA feeds every microphone frame to push(); on a detection, verify() posts
 * the last 2 s (16 kHz mono PCM16) to the local verifier on the mini PC, which answers "accept" or "reject".
 * Fail-open: any error / timeout (150 ms connect, 250 ms read) -> accept, so the Echo behaves exactly as without it.
 * Set URL_BASE to your verifier (wake-verify/wake_verify.py) before compiling.
 */
public final class CasaVerify {
    private static final String URL_BASE = "http://VOICE_HOST:10350/verify";   // placeholder: the verifier's address
    private static final int N = 16000 * 2;
    private static final short[] RING = new short[N];
    private static int pos = 0;

    private CasaVerify() {}

    public static void push(byte[] b) {
        if (b == null) return;
        synchronized (RING) {
            for (int i = 0; i + 1 < b.length; i += 2) {
                RING[pos] = (short) ((b[i] & 0xff) | (b[i + 1] << 8));
                pos = (pos + 1) % N;
            }
        }
    }

    /** Every microphone read of the wake-word loop (MicrophoneInput.readBytes). */
    public static void pushShorts(short[] a) {
        if (a == null) return;
        synchronized (RING) {
            for (short v : a) {
                RING[pos] = v;
                pos = (pos + 1) % N;
            }
        }
    }

    private static byte[] snapshot() {
        byte[] out = new byte[N * 2];
        synchronized (RING) {
            for (int i = 0; i < N; i++) {
                short s = RING[(pos + i) % N];
                out[2 * i] = (byte) s;
                out[2 * i + 1] = (byte) (s >> 8);
            }
        }
        return out;
    }

    /** true = go ahead with the wake; false = the verifier rejected it (a false wake). */
    public static boolean verify(float score) {
        final boolean[] ok = {true};
        final byte[] pcm = snapshot();
        Thread t = new Thread(() -> {
            HttpURLConnection c = null;
            try {
                c = (HttpURLConnection) new URL(URL_BASE + "?score=" + score).openConnection();
                c.setConnectTimeout(150);
                c.setReadTimeout(250);
                c.setDoOutput(true);
                c.setRequestMethod("POST");
                c.setRequestProperty("Content-Type", "application/octet-stream");
                OutputStream os = c.getOutputStream();
                os.write(pcm);
                os.close();
                if (c.getResponseCode() == 200) {
                    InputStream in = c.getInputStream();
                    ByteArrayOutputStream bo = new ByteArrayOutputStream();
                    byte[] buf = new byte[256];
                    int n;
                    while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
                    if (bo.toString("UTF-8").trim().startsWith("reject")) ok[0] = false;
                }
            } catch (Throwable ignored) {
                // fail open
            } finally {
                if (c != null) c.disconnect();
            }
        });
        t.start();
        try {
            t.join(400);
        } catch (InterruptedException ignored) {
        }
        return ok[0] || t.isAlive();
    }
}
