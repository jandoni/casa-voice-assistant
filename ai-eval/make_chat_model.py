import json, os, urllib.request, time, base64, sys
AI=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
RULES = ("Rules for every reply: answer directly, leading with the answer. No greetings, small talk, "
         "follow-up questions, offers of more help, or disclaimers about being an AI. No emojis, no markdown, "
         "no bold, no headers. Do not mention the date, time, platform or device unless asked. Length: one "
         "short sentence for simple questions and facts; for recipes, instructions, lists, comparisons or "
         "explanations give the complete content compactly, e.g. ingredients then numbered steps, with no filler. "
         "For a photo: one or two sentences saying what it shows. Reply in the language the user writes in.")
# Same as gemma3's template, but the rules are always injected into the first turn - even when the
# app sends its own system prompt (Reins adds date/platform context), the rules stay.
TEMPLATE = """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ if eq $i 0 }}""" + RULES + """

{{ end }}{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""
body = {"model":"chat","from":"gemma3:4b","template":TEMPLATE,
        "parameters":{"num_predict":400,"temperature":0.7,"stop":["<end_of_turn>"]},"stream":False}
r = urllib.request.urlopen(urllib.request.Request(AI+"/api/create", data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}), timeout=120)
print("create:", json.loads(r.read()))
