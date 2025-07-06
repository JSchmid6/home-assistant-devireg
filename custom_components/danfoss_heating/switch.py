import logging
from homeassistant.components.switch import SwitchEntity
from .const import DOMAIN
from .pysdg import DeviReg

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Danfoss switch entities."""
    device = hass.data[DOMAIN][config_entry.entry_id]
    
    if isinstance(device, DeviReg):
        async_add_entities([ScreenLockSwitch(device)], True)

class ScreenLockSwitch(SwitchEntity):
    """Representation of a screen lock switch."""

    def __init__(self, device: DeviReg):
        self._device = device
        self._name = f"{device.get_peer_id()} Screen Lock"
        self._unique_id = f"{device.get_peer_id()}_screen_lock"

    @property
    def name(self):
        """Return the name of the switch."""
        return self._name

    @property
    def unique_id(self):
        """Return a unique ID for this entity."""
        return self._unique_id

    @property
    def is_on(self):
        """Return the state of the switch."""
        return self._device.get_screen_lock()

    async def async_turn_on(self, **kwargs):
        """Turn the switch on."""
        await self._device.set_screen_lock(True)

    async def async_turn_off(self, **kwargs):
        """Turn the switch off."""
        await self._device.set_screen_lock(False)

    async def async_update(self):
        """Fetch new state data for the switch."""
        await self._device.update()
