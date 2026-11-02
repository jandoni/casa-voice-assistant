"""casa_weather - fast, local weather answers for Assist (no LLM): met.no forecast (weather.forecast_home) cached in memory.

Sentences: custom_sentences/en/casa_weather.yaml. Every answer is built from the cached hourly (48 h) + daily (6 days)
forecast, refreshed every 15 min and whenever the weather entity changes - a question costs ~1 ms.
Replaces HA's HassGetWeather handler (adds today's high/low + rain) and adds:
  CasaWeatherDay {day}   "what's the weather tomorrow / on Saturday / tonight", "forecast for tomorrow"
  CasaWeatherWeek {span} "weather this week / this weekend"
  CasaRain {day}         "will it rain", "is it raining", "do I need an umbrella", "any storms tomorrow"
  CasaSnow {day}         "will it snow"
  CasaTempOut            "what's the temperature outside", "how cold is it outside"
  CasaFeel {feel} {day}  "is it cold outside", "will it be hot tomorrow", "do I need a jacket"
  CasaWind {day}         "is it windy"
  CasaHighLow {which}    "what's the high today", "how cold will it get tonight"
  CasaSun {event}        "when is sunset", "what time does the sun rise"
  CasaHumidOut           "how humid is it outside"
Enabled by `casa_weather:` in configuration.yaml.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_registry as er, intent
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)
DOMAIN = "casa_weather"
CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)
WEATHER = "weather.forecast_home"

COND = {"clear-night": "clear", "cloudy": "cloudy", "exceptional": "unusual weather", "fog": "foggy", "hail": "hail",
        "lightning": "thunderstorms", "lightning-rainy": "thunderstorms", "partlycloudy": "partly cloudy", "pouring": "heavy rain",
        "rainy": "rain", "snowy": "snow", "snowy-rainy": "sleet", "sunny": "sunny", "windy": "windy", "windy-variant": "windy"}
WET = {"rainy", "pouring", "lightning-rainy", "lightning", "hail", "snowy-rainy"}
SNOW = {"snowy", "snowy-rainy"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _t(v) -> str:
    return f"{round(v)}" if v is not None else "?"


def _hour(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{h} {'AM' if dt.hour < 12 else 'PM'}"


async def async_setup(hass: HomeAssistant, config) -> bool:
    cache = {"at": 0.0, "hourly": [], "daily": []}

    async def refresh(_=None) -> None:
        try:
            out = {}
            for kind in ("hourly", "daily"):
                r = await hass.services.async_call("weather", "get_forecasts", {"entity_id": WEATHER, "type": kind},
                                                   blocking=True, return_response=True)
                fc = (r.get(WEATHER) or {}).get("forecast") or []
                for f in fc:
                    f["_dt"] = dt_util.as_local(dt_util.parse_datetime(f["datetime"]))
                out[kind] = fc
            cache.update(out, at=time.monotonic())
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning("casa_weather: forecast refresh failed: %s", e)

    async def fc():
        if time.monotonic() - cache["at"] > 900 or not cache["hourly"]:
            await refresh()
        return cache["hourly"], cache["daily"]

    def unit() -> str:
        return "degrees"

    def now_state():
        return hass.states.get(WEATHER)

    # ---- time windows ----
    def window(day: str | None):
        """-> (label, start, end, weekday-or-None) in local time."""
        now = dt_util.now()
        today0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
        d = (day or "today").lower()
        if d in ("today", "now"):
            return "today", now, today0 + timedelta(days=1), None
        if d in ("this afternoon",):
            return "this afternoon", max(now, today0 + timedelta(hours=12)), today0 + timedelta(hours=18), None
        if d in ("this evening", "tonight"):
            return "tonight", max(now, today0 + timedelta(hours=18)), today0 + timedelta(days=1, hours=6), None
        if d == "tomorrow":
            return "tomorrow", today0 + timedelta(days=1, hours=6), today0 + timedelta(days=2), None
        if d == "tomorrow night":
            return "tomorrow night", today0 + timedelta(days=1, hours=18), today0 + timedelta(days=2, hours=6), None
        if d == "tomorrow morning":
            return "tomorrow morning", today0 + timedelta(days=1, hours=6), today0 + timedelta(days=1, hours=12), None
        if d == "weekend":
            ahead = (5 - now.weekday()) % 7
            sat = today0 + timedelta(days=ahead)
            return "this weekend", max(now, sat), sat + timedelta(days=2), ahead
        if d in DAYS:
            ahead = (DAYS.index(d) - now.weekday()) % 7
            if ahead == 0:
                return "today", now, today0 + timedelta(days=1), None
            start = today0 + timedelta(days=ahead)
            return ("tomorrow" if ahead == 1 else f"on {d.capitalize()}"), start + timedelta(hours=6), start + timedelta(days=1), ahead
        return "today", now, today0 + timedelta(days=1), None

    def hours_in(hourly, start, end):
        return [h for h in hourly if start <= h["_dt"] < end]

    def daily_for(daily, date):
        return next((d for d in daily if d["_dt"].date() == date), None)

    def summary(label, start, end, hourly, daily):
        if (end - start) > timedelta(days=1, hours=1):
            parts = []
            d0 = start
            while d0 < end:
                d1 = min(end, (d0 + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
                p_ = summary1(label, d0, d1, hourly, daily)
                if p_:
                    parts.append(p_)
                d0 = d1
            if not parts:
                return None
            wet = [p_ for p_ in parts if p_["wet"]]
            conds = [p_["cond"] for p_ in parts]
            return {"cond": max(set(conds), key=conds.count), "hi": max(p_["hi"] for p_ in parts), "lo": min(p_["lo"] for p_ in parts),
                    "rain_mm": sum(p_["rain_mm"] for p_ in parts), "rain_at": None, "wet": bool(wet),
                    "snow": any(p_["snow"] for p_ in parts), "wind": max(p_["wind"] for p_ in parts), "from": "multi"}
        return summary1(label, start, end, hourly, daily)

    def summary1(label, start, end, hourly, daily):
        """-> dict(cond, hi, lo, rain_mm, rain_at, snow, wind) for a window, from hourly when we have it, else daily."""
        hs = hours_in(hourly, start, end)
        if hs:
            temps = [h["temperature"] for h in hs if h.get("temperature") is not None]
            conds = [h["condition"] for h in hs]
            main = max(set(conds), key=conds.count)
            wet = [h for h in hs if (h.get("precipitation") or 0) >= 0.2 or h["condition"] in WET]
            return {"cond": main, "hi": max(temps), "lo": min(temps), "rain_mm": sum(h.get("precipitation") or 0 for h in hs),
                    "rain_at": wet[0]["_dt"] if wet else None, "wet": bool(wet), "snow": any(h["condition"] in SNOW for h in hs),
                    "wind": max(h.get("wind_speed") or 0 for h in hs), "from": "hourly"}
        d = daily_for(daily, start.date())
        if not d:
            return None
        return {"cond": d["condition"], "hi": d.get("temperature"), "lo": d.get("templow"), "rain_mm": d.get("precipitation") or 0,
                "rain_at": None, "wet": (d.get("precipitation") or 0) >= 0.5 or d["condition"] in WET,
                "snow": d["condition"] in SNOW, "wind": d.get("wind_speed") or 0, "from": "daily"}

    # ---- phrasing: short but complete, natural sentences (Jon 2026-09-24: "the temperature tomorrow is going to be
    # between 17 and 19 degrees", not "17 to 19 degrees") ----
    def cap(t: str) -> str:
        return t[:1].upper() + t[1:]

    def poss(label: str) -> str:
        """today -> today's, on Saturday -> Saturday's, this weekend -> this weekend's"""
        return (label[3:] if label.startswith("on ") else label) + "'s"

    def when(label: str) -> str:
        """label as a sentence start: 'Tomorrow', 'On Saturday', 'Tonight'"""
        return cap(label)

    def rain_phrase(s, label) -> str:
        if not s["wet"]:
            return f"No rain is expected {label}."
        word = "Snow" if s["snow"] else ("Thunderstorms are" if s["cond"] in ("lightning", "lightning-rainy") else "Rain")
        verb = "" if word.endswith(" are") else " is"
        at = f", starting around {_hour(s['rain_at'])}" if s.get("rain_at") and label not in ("tonight",) else ""
        heavy = " It could be heavy at times." if s["rain_mm"] >= 10 else ""
        return f"{word}{verb} likely {label}{at}.{heavy}"

    # ---- answers ----
    async def ans_now() -> str:
        st = now_state()
        hourly, daily = await fc()
        if not st:
            return "Sorry, I can't reach the weather service right now."
        a = st.attributes
        lbl, s0, e0, _ = window("today")
        s = summary(lbl, s0, e0, hourly, daily)
        txt = f"Right now it's {_t(a.get('temperature'))} {unit()} and {COND.get(st.state, st.state)}."
        if s:
            hi = max(s['hi'], a.get('temperature') or s['hi']); lo = min(s['lo'], a.get('temperature') or s['lo'])
            txt += f" Today's high is {_t(hi)} and the low is {_t(lo)}. {rain_phrase(s, 'today')}"
        return txt

    async def ans_day(day) -> str:
        hourly, daily = await fc()
        lbl, s0, e0, ahead = window(day)
        if ahead is not None and ahead > 5:
            return "I only have the forecast for the next few days."
        s = summary(lbl, s0, e0, hourly, daily)
        if not s:
            return "I don't have the forecast that far out yet."
        c = COND.get(s['cond'], s['cond'])
        if lbl in ("tonight", "tomorrow night"):
            return f"{when(lbl)} will be {c}, going down to {_t(s['lo'])} {unit()}. {rain_phrase(s, lbl)}"
        return f"{when(lbl)} will be {c}, with a high of {_t(s['hi'])} and a low of {_t(s['lo'])} {unit()}. {rain_phrase(s, lbl)}"

    async def ans_week(span) -> str:
        _, daily = await fc()
        now = dt_util.now()
        if span == "weekend":
            days = [d for d in daily if d["_dt"].weekday() in (5, 6)][:2]
            if not days:
                return "I don't have the weekend forecast yet."
            intro = "This weekend: "
        else:
            days = daily[:6]
            intro = "Here's the week: "
        parts = []
        for d in days:
            name = "today" if d["_dt"].date() == now.date() else ("tomorrow" if d["_dt"].date() == (now + timedelta(days=1)).date() else d["_dt"].strftime("%A"))
            wet = " with some rain" if ((d.get("precipitation") or 0) >= 0.5 or d["condition"] in WET) and d["condition"] not in WET else ""
            parts.append(f"{name} {COND.get(d['condition'], d['condition'])}{wet}, around {_t(d.get('temperature'))} {unit()}")
        return intro + "; ".join(parts) + "."

    async def ans_rain(day, snow=False) -> str:
        hourly, daily = await fc()
        st = now_state()
        lbl, s0, e0, ahead = window(day)
        if not day and st and st.state in WET | SNOW:
            return f"Yes, it's {'snowing' if st.state in SNOW else 'raining'} right now."
        s = summary(lbl, s0, e0, hourly, daily)
        if not s:
            return "I don't have the forecast that far out yet."
        if snow:
            return f"Yes, snow is likely {lbl}." if s["snow"] else f"No, it isn't expected to snow {lbl}."
        if not s["wet"]:
            return f"No, it isn't expected to rain {lbl}."
        return "Yes. " + rain_phrase(s, lbl)

    async def ans_umbrella(day) -> str:
        hourly, daily = await fc()
        lbl, s0, e0, _ = window(day)
        s = summary(lbl, s0, e0, hourly, daily)
        if not s:
            return "I don't have the forecast that far out yet."
        return (f"Yes, take an umbrella. {rain_phrase(s, lbl)}" if s["wet"]
                else f"No, you won't need an umbrella {lbl}. No rain is expected.")

    async def ans_temp_out() -> str:
        st = now_state()
        if not st:
            return "Sorry, I can't reach the weather service right now."
        a = st.attributes
        return f"It's {_t(a.get('temperature'))} {unit()} outside right now, and {COND.get(st.state, st.state)}."

    def feel_word(t) -> str:
        return ("freezing" if t < 0 else "cold" if t < 10 else "chilly" if t < 16 else "mild" if t < 22 else "warm" if t < 28 else "hot")

    async def ans_feel(feel, day) -> str:
        hourly, daily = await fc()
        if not day:
            st = now_state()
            t = st.attributes.get("temperature") if st else None
            if t is None:
                return "Sorry, I can't reach the weather service right now."
            w = feel_word(t)
            if feel in ("jacket", "coat", "sweater"):
                return (f"Yes, I'd bring a jacket. It's {_t(t)} {unit()} outside and {w}." if t < 16
                        else f"You probably don't need a jacket. It's {_t(t)} {unit()} outside and {w}.")
            yes = {"cold": t < 12, "chilly": t < 16, "hot": t >= 27, "warm": t >= 21, "nice": 16 <= t < 27}.get(feel, False)
            return (f"Yes, it's {w} outside, {_t(t)} {unit()}." if yes
                    else f"Not really. It's {_t(t)} {unit()} outside, which is {w}.")
        lbl, s0, e0, _ = window(day)
        s = summary(lbl, s0, e0, hourly, daily)
        if not s:
            return "I don't have the forecast that far out yet."
        hi, lo = s["hi"], s["lo"]
        rng = f"between {_t(lo)} and {_t(hi)} {unit()}"
        if feel in ("jacket", "coat", "sweater"):
            return (f"Yes, bring a jacket. It will get down to {_t(lo)} {unit()} {lbl}." if lo < 16
                    else f"You probably won't need one. {when(lbl)} will be {rng}.")
        ref = lo if feel in ("cold", "chilly") else hi
        yes = {"cold": ref < 12, "chilly": ref < 16, "hot": ref >= 27, "warm": ref >= 21, "nice": 16 <= hi < 28 and not s["wet"]}.get(feel, False)
        return f"{'Yes' if yes else 'Not really'}. {when(lbl)} will be {rng}, and {COND.get(s['cond'], s['cond'])}."

    async def ans_wind(day) -> str:
        hourly, daily = await fc()
        if not day:
            st = now_state()
            w = st.attributes.get("wind_speed") if st else None
            if w is None:
                return "Sorry, I can't reach the weather service right now."
            kind = "It's windy" if w >= 30 else "It's a little breezy" if w >= 15 else "It's calm"
            return f"{kind} right now, with wind at {_t(w)} kilometers an hour."
        lbl, s0, e0, _ = window(day)
        s = summary(lbl, s0, e0, hourly, daily)
        if not s:
            return "I don't have the forecast that far out yet."
        kind = "windy" if s['wind'] >= 30 else "a little breezy" if s['wind'] >= 15 else "calm"
        return f"{when(lbl)} will be {kind}, with wind up to {_t(s['wind'])} kilometers an hour."

    async def ans_highlow(which, day) -> str:
        hourly, daily = await fc()
        lbl, s0, e0, _ = window(day)
        s = summary(lbl, s0, e0, hourly, daily)
        if not s:
            return "I don't have the forecast that far out yet."
        if which == "low":
            return f"{cap(poss(lbl))} low will be {_t(s['lo'])} {unit()}."
        if which == "high":
            return f"{cap(poss(lbl))} high will be {_t(s['hi'])} {unit()}."
        return f"The temperature {lbl} will be between {_t(s['lo'])} and {_t(s['hi'])} {unit()}."

    async def ans_sun(event) -> str:
        sun = hass.states.get("sun.sun")
        if not sun:
            return "Sorry, I don't know that right now."
        key = "next_setting" if event == "sunset" else "next_rising"
        t = dt_util.as_local(dt_util.parse_datetime(sun.attributes[key]))
        day = "today" if t.date() == dt_util.now().date() else "tomorrow"
        return f"{'Sunset' if event == 'sunset' else 'Sunrise'} is at {t.strftime('%-I:%M %p')} {day}."

    async def ans_humid() -> str:
        st = now_state()
        h = st.attributes.get("humidity") if st else None
        return f"The humidity outside is {_t(h)} percent." if h is not None else "Sorry, I can't reach the weather service right now."

    # ---- on-screen card (2026-09-26): the asking Echo shows the forecast for ~15 s (echo-home.js renders it from the
    #      casa_show event - no dashboard re-render, appears as the answer is spoken) ----
    WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def focus_date(day):
        today = dt_util.now().date()
        if not day or day in ("today", "now", "tonight", "this morning", "this afternoon", "this evening"):
            return today
        if day == "tomorrow" or day.startswith("tomorrow"):
            return today + timedelta(days=1)
        for i, w in enumerate(WEEKDAYS):
            if w in day:
                return today + timedelta(days=(i - today.weekday()) % 7)
        if "weekend" in day:
            return today + timedelta(days=(5 - today.weekday()) % 7)
        return today

    def vaca_of(device_id):
        if not device_id:
            return None
        for ent in er.async_entries_for_device(er.async_get(hass), device_id):
            if ent.entity_id.startswith("assist_satellite.vaca_"):
                return ent.entity_id.split("vaca_", 1)[1]
        return None

    async def show_card(io, day, spoken, dev=None):
        dev = dev or vaca_of(io.device_id if io else None)
        if not dev:
            return
        hourly, daily = await fc()
        st = hass.states.get(WEATHER)
        focus = focus_date(day)
        days = []
        for d in daily[:6]:
            if len(days) == 5:
                break
            days.append({"date": d["_dt"].date().isoformat(), "cond": d.get("condition"),
                         "hi": round(d.get("temperature")) if d.get("temperature") is not None else None,
                         "lo": round(d.get("templow")) if d.get("templow") is not None else None,
                         "rain": round(d.get("precipitation") or 0, 1)})
        now_h = [{"t": h["_dt"].strftime("%-I %p").lower(), "cond": h.get("condition"),
                  "temp": round(h["temperature"]) if h.get("temperature") is not None else None}
                 for h in hourly if h["_dt"] >= dt_util.now() - timedelta(minutes=30)][:6]
        hass.bus.async_fire("casa_show", {
            "dev": dev, "card": "weather", "secs": 15, "spoken": spoken,
            "data": {"focus": focus.isoformat(), "today": dt_util.now().date().isoformat(),
                     "now": {"temp": st.attributes.get("temperature") if st else None, "cond": st.state if st else None,
                             "humidity": st.attributes.get("humidity") if st else None},
                     "days": days, "hours": now_h}})

    # ---- intent handlers ----
    def slot(io, name):
        v = (io.slots.get(name) or {}).get("value")
        return str(v).lower() if v not in (None, "") else None

    def make(intent_type, fn):
        class H(intent.IntentHandler):
            def __init__(self):
                self.intent_type = intent_type

            @property
            def slot_schema(self):
                return None

            async def async_handle(self, io):
                t0 = time.perf_counter()
                try:
                    text = await fn(io)
                except Exception as e:  # noqa: BLE001
                    _LOGGER.warning("casa_weather: %s failed: %s", intent_type, e)
                    text = "Sorry, I couldn't get the weather."
                _LOGGER.debug("casa_weather: %s in %.1f ms", intent_type, (time.perf_counter() - t0) * 1000)
                try:
                    await show_card(io, slot(io, "day") or ("weekend" if slot(io, "span") == "weekend" else None), text)
                except Exception as e:  # noqa: BLE001
                    _LOGGER.debug("casa_weather: card failed: %s", e)
                r = io.create_response()
                r.async_set_speech(text)
                return r
        return H()

    handlers = []
    for itype, fn in (
        ("HassGetWeather", lambda io: ans_day(slot(io, "day")) if slot(io, "day") not in (None, "today", "now") else ans_now()),
        ("CasaWeatherDay", lambda io: ans_day(slot(io, "day"))),
        ("CasaWeatherWeek", lambda io: ans_week(slot(io, "span") or "week")),
        ("CasaRain", lambda io: ans_rain(slot(io, "day"))),
        ("CasaUmbrella", lambda io: ans_umbrella(slot(io, "day"))),
        ("CasaSnow", lambda io: ans_rain(slot(io, "day"), snow=True)),
        ("CasaTempOut", lambda io: ans_temp_out()),
        ("CasaFeel", lambda io: ans_feel(slot(io, "feel") or ("jacket" if slot(io, "garment") else "cold"), slot(io, "day"))),
        ("CasaWind", lambda io: ans_wind(slot(io, "day"))),
        ("CasaHighLow", lambda io: ans_highlow(slot(io, "which") or "both", slot(io, "day"))),
        ("CasaSun", lambda io: ans_sun(slot(io, "event") or "sunset")),
        ("CasaHumidOut", lambda io: ans_humid()),
    ):
        handlers.append(make(itype, fn))

    def _register(_=None):
        # HA's weather integration registers its own HassGetWeather while loading intents: (re)register ours after start
        for h in handlers:
            intent.async_register(hass, h)
    _register()
    from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _register)

    @callback
    def _changed(_event) -> None:
        cache["at"] = 0.0          # the entity updated: next question (or the refresh below) re-reads the forecast
        hass.async_create_task(refresh())

    async def _start(_=None):
        await refresh()
    hass.async_create_task(_start())
    async_track_state_change_event(hass, [WEATHER], _changed)
    @callback
    def _tick(_now) -> None:       # a plain lambda runs in a worker thread -> HA's thread-safety error
        hass.async_create_task(refresh())
    async_track_time_interval(hass, _tick, timedelta(minutes=15))

    # (2026-09-28) the Living tab's weather chip: casa_weather.show {dev} draws today's card on that Echo, silently
    async def _show(call) -> None:
        dev = str(call.data.get("dev") or "")
        if dev:
            await show_card(None, "today", "", dev=dev)
    hass.services.async_register(DOMAIN, "show", _show)
    return True
