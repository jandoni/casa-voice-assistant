"""Casa Brain - a conversation agent that routes voice requests the way the chat does.

House-related requests -> the Home Assistant "Casa" agent (Ollama + Assist tools).
Everything else (facts, web search, recipes, general questions) -> the Open WebUI "casa" pipe
(SearXNG + Wikipedia + the LLM brain) in voice mode, answer trimmed to plain speech.

configuration.yaml:
  casa_brain:
    house_agent: conversation.casa
    chat_url: http://<open-webui host>:8080/api/chat/completions
    chat_model: casa
    chat_api_key: !secret openwebui_api_key
"""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, SOURCE_IMPORT
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

DOMAIN = "casa_brain"
_LOGGER = logging.getLogger(__name__)
PLATFORMS = [Platform.CONVERSATION]

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional("house_agent", default="conversation.casa"): cv.string,
                vol.Required("chat_url"): cv.string,
                vol.Optional("chat_model", default="casa"): cv.string,
                vol.Required("chat_api_key"): cv.string,
                vol.Optional("chat_timeout", default=25): cv.positive_int,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    conf = config.get(DOMAIN)
    if not conf:
        return True
    hass.data.setdefault(DOMAIN, {})["yaml"] = dict(conf)
    if not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_IMPORT}, data={})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
