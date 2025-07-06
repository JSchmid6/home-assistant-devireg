import logging
from homeassistant.components.select import SelectEntity
from .const import DOMAIN
from .pysdg import DeviReg

_LOGGER = logging.getLogger(__name__)

CONTROL_MODES = ["Manual", "Schedule", "Vacation", "Pause", "Off", "Override"]

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Danfoss select entities."""
    device = hass.data[DOMAIN][config_entry.entry_id]
    
    if isinstance(device, DeviReg):
        async_add_entities([ControlModeSelect(device)], True)

class ControlModeSelect(SelectEntity):
    """Representation of a control mode select entity."""

    def __init__(self, device: DeviReg):
        self._device = device
        self._name = f"{device.get_peer_id()} Control Mode"
        self._unique_id = f"{device.get_peer_id()}_control_mode"

    @property
    def name(self):
        """Return the name of the select entity."""
        return self._name

    @property
    def unique_id(self):
        """Return a unique ID for this entity."""
        return self._unique_id

    @property
    def options(self):
        """Return the list of available options."""
        return CONTROL_MODES

    @property
    def current_option(self):
        """Return the currently selected option."""
        return self._device.get_control_mode()

    async def async_select_option(self, option: str):
        """Change the selected option."""
        await self._device.set_control_mode(option)

    async def async_update(self):
        """Fetch new state data for the select entity."""
        await self._device.update()
