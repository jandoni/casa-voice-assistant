#!/bin/sh
# Ollama health check (voice host, every 5 min via ollama-health.timer).
# After an Arc iGPU hang (dmesg "GPU HANG" / "Fence expiration") Ollama keeps answering, but with EMPTY content,
# a 500 ("vk::Queue::submit") or pure garbage, until it is restarted - HA's Casa agent then says "Unable to get
# response" or speaks nonsense. This asks for one known word and restarts Ollama when the answer is not that word
# twice in a row.
STATE=/run/ollama-health.fails
# (2026-09-24) num_ctx MUST match the HA agent + chat pipe (16384): Ollama reloads granite whenever num_ctx changes, so a
# 4096 health ping reloaded the model every 5 min (+~3 s on the next voice question, extra GPU churn).
# (2026-09-24) never add a GPU job on top of a real request: two jobs at once halve the speed of the one someone is
# waiting for and make long batches (= i915 hangs) more likely. Busy = a client connection open to Ollama right now.
if [ "$(ss -Htn state established '( sport = :11434 )' 2>/dev/null | wc -l)" -gt 0 ]; then
    exit 0
fi
OUT=$(curl -s -m 60 http://127.0.0.1:11434/api/chat -d '{"model":"granite4.2:3b","stream":false,"think":false,"keep_alive":"24h","messages":[{"role":"user","content":"Reply with the single word pong."}],"options":{"num_predict":8,"num_ctx":16384}}' 2>/dev/null)
CONTENT=$(printf '%s' "$OUT" | python3 -c 'import sys,json
try: print(json.load(sys.stdin)["message"]["content"].strip())
except Exception: print("")' 2>/dev/null)
case "$(printf '%s' "$CONTENT" | tr 'A-Z' 'a-z')" in
    *pong*)
        rm -f "$STATE"
        exit 0
        ;;
esac
N=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$STATE"
logger -t ollama-health "bad reply from Ollama ($N in a row): $(printf '%s' "$OUT" | head -c 200)"
if [ "$N" -ge 2 ]; then
    logger -t ollama-health "restarting ollama (bad replies; likely iGPU hang)"
    rm -f "$STATE"
    systemctl restart ollama
fi
exit 0
