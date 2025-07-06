import voluptuous as vol
import logging
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import config_validation as cv
from .const import DOMAIN
from .pairing import DanfossPairing

_LOGGER = logging.getLogger(__name__)

class DanfossConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Danfoss Heating."""
    
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    def __init__(self):
        """Initialize the config flow."""
        self.discovered_devices = []

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        return self.async_show_form(
            step_id="user",
            description_placeholders={
                "url": f"http://{self.hass.config.api.host}:8123/api/danfoss_heating/configreceiver.html"
            }
        )

    async def async_step_select(self, user_input=None):
        """Handle the device selection step."""
        errors = {}
        
        if user_input is not None:
            _LOGGER.debug("User selected device: %s", user_input["device"])
            # Find the selected device
            device = next((d for d in self.discovered_devices if d["name"] == user_input["device"]), None)
            
            if device:
                _LOGGER.debug("Creating config entry for device: %s", device)
                return self.async_create_entry(title=device["name"], data=device)
            else:
                _LOGGER.error("Device not found: %s", user_input["device"])
                errors["base"] = "device_not_found"

        device_names = [d["name"] for d in self.discovered_devices]
        
        data_schema = vol.Schema({
            vol.Required("device"): vol.In(device_names),
        })
        
        return self.async_show_form(
            step_id="select",
            data_schema=data_schema,
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return DanfossOptionsFlow(config_entry)

class DanfossOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Danfoss Heating."""
    
    def __init__(self, config_entry):
        self.config_entry = config_entry
        
    async def async_step_init(self, user_input=None):
        """Manage the options."""
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({})
        )
