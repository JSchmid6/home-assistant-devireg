import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from .const import DOMAIN, DEVICE_TYPE_DEVISMART, DEVICE_TYPE_ICON_ROOM, CONF_PEER_ID, CONF_DEVICE_TYPE, CONF_ROOM_NUMBER, CONF_HOST

class DanfossConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Danfoss Heating."""
    
    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_LOCAL_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}
        
        if user_input is not None:
            return self.async_create_entry(
                title=f"Danfoss {user_input[CONF_DEVICE_TYPE]}",
                data=user_input
            )
            
        data_schema = vol.Schema({
            vol.Required(CONF_HOST): str,
            vol.Required(CONF_PEER_ID): str,
            vol.Required(CONF_DEVICE_TYPE, default=DEVICE_TYPE_DEVISMART): vol.In([DEVICE_TYPE_DEVISMART, DEVICE_TYPE_ICON_ROOM]),
            vol.Optional(CONF_ROOM_NUMBER): cv.positive_int,
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
