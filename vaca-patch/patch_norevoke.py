#!/usr/bin/env python3
"""casa7 (2026-10-23): VACA must never revoke its own Home Assistant login.

Why: the Show 8 woke up on HA's login page (2026-10-23). VACA serves the dashboard's tokens ("external auth");
when a token refresh fails for ANY reason - AuthenticationService wraps every exception (timeout, connection reset,
Wi-Fi hiccup) in AuthenticationException - CustomWebView.requestAuthorisation() calls safeRevokeSession(), which
POSTs a revoke for the still-valid refresh token, so HA deletes it, and then reloads the page -> login screen.
The Show 8 had a two-minute Wi-Fi drop (HA lost its satellite) exactly then.

With safeRevokeSession() a no-op a failed refresh just reloads the page and asks again (works once the network is
back). A refresh token HA really rejects (invalid_grant) is still cleared by AuthenticationManager, which then shows
the login page - unchanged. Applies on top of the casa builds (run on a fresh apktool dir of the installed APK).

Usage: patch_norevoke.py <apktool-dir>
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1])
hits = list(root.glob("smali*/com/msp1974/vacompanion/utils/CustomWebView.smali"))
if len(hits) != 1:
    sys.exit(f"CustomWebView.smali not found exactly once: {hits}")
p = hits[0]
s = p.read_text()
sig = ".method private final safeRevokeSession(Lkotlin/coroutines/Continuation;)Ljava/lang/Object;"
if "CASA PATCH casa7" in s:
    sys.exit("already patched")
a = s.index(sig)
b = s.index(".end method", a)
s = s[:a] + sig + """
    .registers 3
    # CASA PATCH casa7: never revoke the HA refresh token (a network blip used to log the Echo out for good)
    sget-object v0, Lkotlin/Unit;->INSTANCE:Lkotlin/Unit;
    return-object v0
""" + s[b:]
p.write_text(s)
print(f"patched {p}")
