package com.casa;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.ContentObserver;

/** casa6 (2026-09-29): report the Echo's volume keys to HA at once. VACA's VolumeObserver watches Settings.System,
 *  which Android only writes ~0.5-1 s after the change (the dashboard slider lagged ~1.3 s). Android's
 *  VOLUME_CHANGED_ACTION broadcast fires immediately -> run the same observer callback then too. */
public final class CasaVolume {
    private CasaVolume() {}

    public static void register(Context ctx, final ContentObserver obs) {
        try {
            ctx.getApplicationContext().registerReceiver(new BroadcastReceiver() {
                @Override public void onReceive(Context c, Intent i) {
                    try {
                        // only a real change of the MUSIC stream (= the volume keys). The assistant-voice changes and
                        // HA's own reset of the music stream also broadcast; reacting to those re-reported stale values
                        // and HA counted them as extra key presses (double steps).
                        int type = i.getIntExtra("android.media.EXTRA_VOLUME_STREAM_TYPE", -1);
                        int v = i.getIntExtra("android.media.EXTRA_VOLUME_STREAM_VALUE", -1);
                        int pv = i.getIntExtra("android.media.EXTRA_PREV_VOLUME_STREAM_VALUE", -2);
                        if (type != 3 || v == pv) return;
                        obs.onChange(false, null);
                    } catch (Throwable t) { /* never break VACA */ }
                }
            }, new IntentFilter("android.media.VOLUME_CHANGED_ACTION"));
        } catch (Throwable t) { /* fall back to the settings observer only */ }
    }
}
