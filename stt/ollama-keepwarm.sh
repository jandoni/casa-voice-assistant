#!/bin/sh
# after every Ollama (re)start, ask HA to re-warm the Casa prompt (cold house prompt = 40 s)
sleep 25
. /opt/casa-stt/casa-stt.env
curl -s -m 10 -X POST -H "Authorization: Bearer $HA_TOKEN" -H "Content-Type: application/json"   -d "{\"entity_id\":\"automation.assist_ai_keep_casa_warm\",\"skip_condition\":true}"   "$HA_URL/api/services/automation/trigger" >/dev/null 2>&1 || true
exit 0
