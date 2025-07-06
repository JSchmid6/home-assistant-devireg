import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from .const import DOMAIN, CONF_PEER_ID, CONF_DEVICE_TYPE, CONF_ROOM_NUMBER, CONF_HOST
from .discovery import DanfossDiscovery

class DanfossConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Danfoss Heating."""
    
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_POLL

    def __init__(self):
        """Initialize the config flow."""
        self.discovered_devices = []

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        return await self.async_step_discover()

    async def async_step_discover(self, user_input=None):
        """Handle the discovery step."""
        errors = {}
        
        if user_input is not None:
            discovery = DanfossDiscovery(self.hass)
            self.discovered_devices = await discovery.discover_devices(user_input["otp"])
            
            if self.discovered_devices:
                return await self.async_step_select()
            else:
                errors["base"] = "no_devices_found"
                
        data_schema = vol.Schema({
            vol.Required("otp"): str,
        })
        
        return self.async_show_form(
            step_id="discover",
            data_schema=data_schema,
            errors=errors
        )

    async def async_step_select(self, user_input=None):
        """Handle the device selection step."""
        errors = {}
        
        if user_input is not None:
            # Find the selected device
            device = next((d for d in self.discovered_devices if d["name"] == user_input["device"]), None)
            
            if device:
                # In a real implementation, we would also get the host from discovery
                device[CONF_HOST] = "192.168.1.1" # Placeholder
                return self.async_create_entry(title=device["name"], data=device)
            else:
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
