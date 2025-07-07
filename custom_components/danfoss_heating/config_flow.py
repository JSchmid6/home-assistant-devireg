import voluptuous as vol
import logging
from homeassistant import config_entries
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv
from .const import (
    DOMAIN,
    CONF_PEER_ID,
    CONF_DEVICE_TYPE,
    DEVICE_TYPE_DEVISMART,
    DEVICE_TYPE_ICON_ROOM,
    CONF_ROOM_NUMBER,
    CONF_HOST,
    CONF_CONNECTION_TYPE,
    CONNECTION_TYPE_LOCAL,
    CONNECTION_TYPE_CLOUD,
    CONF_OTP,
)
from .pysdg_cloud.connector import DanfossCloudConnector

_LOGGER = logging.getLogger(__name__)


class DanfossConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Danfoss Heating."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if user_input is not None:
            if user_input[CONF_CONNECTION_TYPE] == CONNECTION_TYPE_CLOUD:
                return await self.async_step_cloud()
            else:
                return await self.async_step_local()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CONNECTION_TYPE, default=CONNECTION_TYPE_LOCAL
                    ): vol.In([CONNECTION_TYPE_LOCAL, CONNECTION_TYPE_CLOUD])
                }
            ),
        )

    async def async_step_local(self, user_input=None):
        """Handle the local connection setup."""
        errors = {}

        if user_input is not None:
            _LOGGER.debug("Creating local config entry for: %s", user_input)
            return self.async_create_entry(
                title=f"Danfoss {user_input[CONF_DEVICE_TYPE]} (Local)",
                data={**user_input, CONF_CONNECTION_TYPE: CONNECTION_TYPE_LOCAL},
            )

        data_schema = vol.Schema(
            {
                vol.Required(CONF_HOST): str,
                vol.Required(CONF_PEER_ID): str,
                vol.Required(
                    CONF_DEVICE_TYPE, default=DEVICE_TYPE_DEVISMART
                ): vol.In([DEVICE_TYPE_DEVISMART, DEVICE_TYPE_ICON_ROOM]),
                vol.Optional(CONF_ROOM_NUMBER): cv.positive_int,
            }
        )

        return self.async_show_form(
            step_id="local", data_schema=data_schema, errors=errors
        )

    async def async_step_cloud(self, user_input=None):
        """Handle the cloud connection setup."""
        errors = {}

        if user_input is not None:
            otp = user_input[CONF_OTP]
            user_name = user_input["user_name"]
            connector = DanfossCloudConnector(
                otp, user_name, self.hass.config.path()
            )
            try:
                devices = await connector.get_devices()

                if devices:
                    return self.async_create_entry(
                        title=devices["houseName"],
                        data={
                            CONF_CONNECTION_TYPE: CONNECTION_TYPE_CLOUD,
                            "devices": devices,
                        },
                    )
                else:
                    errors["base"] = "invalid_otp"
            except NotImplementedError:
                errors["base"] = "not_implemented"

        return self.async_show_form(
            step_id="cloud",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_OTP): str,
                    vol.Required("user_name", default="Home Assistant"): str,
                }
            ),
            errors=errors,
        )

    async def async_step_discovery(self, discovery_info):
        """Handle discovery flow."""
        return self.async_create_entry(
            title=f"Danfoss {discovery_info[CONF_DEVICE_TYPE]}",
            data=discovery_info
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
