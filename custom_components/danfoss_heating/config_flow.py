import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from .const import DOMAIN

CONF_USERNAME = "username"
CONF_PRIVATE_KEY = "private_key"
CONF_PUBLIC_KEY = "public_key"

DEFAULT_USERNAME = "HomeAssistant"

class DanfossConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Danfoss Heating."""
    
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}
        
        if user_input is not None:
            # Validate input here (in real implementation would validate keys)
            return self.async_create_entry(
                title=f"Danfoss Heating - {user_input[CONF_USERNAME]}",
                data=user_input
            )
            
        data_schema = vol.Schema({
            vol.Required(CONF_USERNAME, default=DEFAULT_USERNAME): str,
            vol.Required(CONF_PRIVATE_KEY): str,
            vol.Required(CONF_PUBLIC_KEY): str,
        })
        
        return self.async_show_form(
            step_id="user",
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
