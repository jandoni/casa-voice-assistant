"""Config flow for Casa Brain (settings live in configuration.yaml; the entry only hosts the entity)."""
from __future__ import annotations

from homeassistant.config_entries import ConfigFlow

from . import DOMAIN


class CasaBrainConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_import(self, import_data):
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        return self.async_create_entry(title="Casa Brain", data={})

    async def async_step_user(self, user_input=None):
        return await self.async_step_import({})
