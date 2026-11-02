#!/usr/bin/env python3
"""
ollama-guard - tiny HTTP proxy in front of Ollama (voice host, :11435 -> :11434) for Home Assistant.
Its one job: cap every /api/chat generation (options.num_predict, default 400) so a degenerate model
can never run for minutes and freeze the voice satellite. Everything else is streamed through untouched.
(A derived `ollama create` model would do this too, but deriving from granite4.2 loses its built-in
chat renderer and the copy emits garbage - see LOCAL_AI_FINDINGS 2026-09-21.)
"""
import asyncio, json, logging, os
import aiohttp
from aiohttp import web

UPSTREAM = os.environ.get("OLLAMA_UPSTREAM", "http://127.0.0.1:11434")
PORT = int(os.environ.get("OLLAMA_GUARD_PORT", "11435"))
NUM_PREDICT = int(os.environ.get("OLLAMA_GUARD_NUM_PREDICT", "400"))
log = logging.getLogger("ollama-guard")


async def handle(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")}
    if request.method == "POST" and request.path in ("/api/chat", "/api/generate") and body:
        try:
            j = json.loads(body)
            opts = j.setdefault("options", {}) or {}
            if not isinstance(opts.get("num_predict"), int) or opts["num_predict"] <= 0 or opts["num_predict"] > NUM_PREDICT:
                opts["num_predict"] = NUM_PREDICT
            j["options"] = opts
            body = json.dumps(j).encode()
            msgs = j.get("messages") or []
            log.info("chat: model=%s stream=%s think=%s options=%s tools=%d msgs=%d chars=%d last=%r", j.get("model"), j.get("stream"), j.get("think"),
                     opts, len(j.get("tools") or []), len(msgs), sum(len(json.dumps(m)) for m in msgs), (msgs[-1].get("content") or "")[:80] if msgs else "")
            if os.environ.get("OLLAMA_GUARD_DUMP"):
                open("/tmp/last_chat_request.json", "w").write(json.dumps(j))
        except Exception as e:  # noqa: BLE001
            log.warning("could not patch body: %s", e)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as sess:
        async with sess.request(request.method, UPSTREAM + request.path_qs, data=body, headers=headers) as up:
            resp = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handle)
    log.info("ollama-guard :%d -> %s (num_predict cap %d)", PORT, UPSTREAM, NUM_PREDICT)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)


if __name__ == "__main__":
    main()
