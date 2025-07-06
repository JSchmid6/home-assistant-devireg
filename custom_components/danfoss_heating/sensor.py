import logging
from homeassistant.components.sensor import SensorEntity
from homeassistant.const import UnitOfTemperature
from .const import DOMAIN
from .pysdg import DeviReg

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Danfoss sensor entities."""
    device = hass.data[DOMAIN][config_entry.entry_id]
    
    if isinstance(device, DeviReg):
        async_add_entities([FloorTemperatureSensor(device)], True)

class FloorTemperatureSensor(SensorEntity):
    """Representation of a floor temperature sensor."""

    def __init__(self, device: DeviReg):
        self._device = device
        self._name = f"{device.get_peer_id()} Floor Temperature"
        self._unique_id = f"{device.get_peer_id()}_floor_temp"

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def unique_id(self):
        """Return a unique ID for this entity."""
        return self._unique_id

    @property
    def native_unit_of_measurement(self):
        """Return the unit of measurement."""
        return UnitOfTemperature.CELSIUS

    @property
    def native_value(self):
        """Return the state of the sensor."""
        return self._device.get_floor_temperature()

    async def async_update(self):
        """Fetch new state data for the sensor."""
        await self._device.update()
