import voluptuous as vol
import logging
from homeassistant import config_entries
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from .const import DOMAIN
from .web_server import DanfossWebServer

_LOGGER = logging.getLogger(__name__)

class DanfossConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Danfoss Heating."""
    
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_POLL

    def __init__(self):
        """Initialize the config flow."""
        self.web_server = None
        self.discovered_devices = []

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        _LOGGER.debug("Starting config flow")
        if not self.web_server:
            self.web_server = DanfossWebServer(self.hass)
            self.web_server.start()

        return self.async_show_form(
            step_id="user",
            description_placeholders={
                "url": f"http://{self.hass.config.api.host}:8080/custom_components/danfoss_heating/configreceiver.html"
            }
        )

    async def async_step_discovery_complete(self, user_input=None):
        """Handle the discovery complete step."""
        _LOGGER.debug("Discovery complete")
        if self.web_server:
            self.web_server.stop()
            self.web_server = None
            
        # In a real implementation, the web server would have populated
        # self.discovered_devices. For now, we'll assume it has.
        
        return await self.async_step_select()

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
