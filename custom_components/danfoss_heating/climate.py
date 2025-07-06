import logging
import asyncio
from homeassistant.components.climate import ClimateEntity, ClimateEntityFeature, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from .const import (
    DOMAIN,
    DEVICE_TYPE_DEVISMART,
    DEVICE_TYPE_ICON_ROOM,
    CONF_PEER_ID,
    CONF_ROOM_NUMBER
)
from .pysdg import SDGPeerConnector, DeviReg, IconRoom

_LOGGER = logging.getLogger(__name__)

# Map Danfoss modes to Home Assistant HVAC modes
HVAC_MODES = [HVACMode.HEAT, HVACMode.OFF]

# Preset modes for different setpoints
PRESET_MODES = {
    "comfort": "At Home",
    "economy": "Away",
    "manual": "Manual",
    "away": "Vacation",
    "antifreeze": "Frost Protection"
}

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the Danfoss climate entities."""
    device = hass.data[DOMAIN][config_entry.entry_id]
    
    async_add_entities([DanfossClimate(hass, config_entry, device.name, device)], True)

class DanfossClimate(ClimateEntity):
    """Representation of a Danfoss Heating climate entity."""
    
    def __init__(self, hass, config_entry, name, device):
        """Initialize the climate entity."""
        self._hass = hass
        self._config_entry = config_entry
        self._name = name
        self._device = device
        
        # Default values
        self._current_temp = None
        self._target_temp = None
        self._floor_temp = None
        self._hvac_mode = HVACMode.HEAT
        self._preset_mode = "comfort"
        self._window_open = False
        self._heating_on = False
        
    @property
    def unique_id(self):
        """Return a unique ID for this entity."""
        return self._device.get_peer_id()
        
    @property
    def name(self):
        """Return the name of the entity."""
        return self._name
        
    @property
    def temperature_unit(self):
        """Return the unit of measurement."""
        return UnitOfTemperature.CELSIUS
        
    @property
    def current_temperature(self):
        """Return the current temperature."""
        return self._current_temp
        
    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._target_temp
        
    @property
    def hvac_mode(self):
        """Return current hvac operation mode."""
        return self._hvac_mode
        
    @property
    def hvac_modes(self):
        """Return the list of available hvac operation modes."""
        return HVAC_MODES
        
    @property
    def preset_mode(self):
        """Return the current preset mode."""
        return self._preset_mode
        
    @property
    def preset_modes(self):
        """Return the list of available preset modes."""
        return list(PRESET_MODES.keys())
        
    @property
    def supported_features(self):
        """Return the list of supported features."""
        return ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE
        
    @property
    def extra_state_attributes(self):
        """Return additional state attributes."""
        attrs = {
            "floor_temperature": self._floor_temp,
            "window_open": self._window_open,
            "heating_on": self._heating_on,
            "peer_id": self._device.get_peer_id()
        }
        return attrs
        
    async def async_set_temperature(self, **kwargs):
        """Set new target temperature."""
        temp = kwargs.get(ATTR_TEMPERATURE)
        if temp is None:
            return
            
        _LOGGER.debug("Setting temperature for %s to %s", self.name, temp)
        await self._device.set_temperature(temp)
        
    async def async_set_hvac_mode(self, hvac_mode):
        """Set new hvac mode."""
        _LOGGER.debug("Setting HVAC mode for %s to %s", self.name, hvac_mode)
        # This needs to be implemented in the device classes
        
    async def async_set_preset_mode(self, preset_mode):
        """Set new preset mode."""
        _LOGGER.debug("Setting preset mode for %s to %s", self.name, preset_mode)
        # This needs to be implemented in the device classes
        
    async def async_update(self):
        """Fetch new state data for the sensor."""
        await self._device.update()
        self._current_temp = self._device.get_current_temperature()
        self._target_temp = self._device.get_target_temperature()
        
        if isinstance(self._device, DeviReg):
            self._floor_temp = self._device.get_floor_temperature()
            self._window_open = self._device.get_window_open()
            self._heating_on = self._device.get_heating_on()
            
        self._hvac_mode = self._device.get_hvac_mode()
        self._preset_mode = self._device.get_preset_mode()
        _LOGGER.debug("Updated climate entity: %s", self.name)
