import logging
from homeassistant.components.climate import ClimateEntity, ClimateEntityFeature, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from .const import (
    DOMAIN,
    DEVICE_TYPE_DEVISMART,
    DEVICE_TYPE_ICON_ROOM,
    CONF_PEER_ID,
    CONF_ROOM_NUMBER
)

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

class DanfossClimate(ClimateEntity):
    """Representation of a Danfoss Heating climate entity."""
    
    def __init__(self, hass, config_entry, name, device_type, peer_id, room_number=None):
        """Initialize the climate entity."""
        self._hass = hass
        self._config_entry = config_entry
        self._name = name
        self._device_type = device_type
        self._peer_id = peer_id
        self._room_number = room_number
        
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
        return f"{self._peer_id}_{self._room_number}" if self._room_number else self._peer_id
        
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
            "device_type": self._device_type,
            "peer_id": self._peer_id
        }
        if self._room_number:
            attrs["room_number"] = self._room_number
        return attrs
        
    async def async_set_temperature(self, **kwargs):
        """Set new target temperature."""
        temp = kwargs.get(ATTR_TEMPERATURE)
        if temp is None:
            return
            
        # In a real implementation, this would send the temperature to the device
        _LOGGER.info("Setting temperature for %s to %s", self.name, temp)
        self._target_temp = temp
        self.async_write_ha_state()
        
    async def async_set_hvac_mode(self, hvac_mode):
        """Set new hvac mode."""
        # In a real implementation, this would control the device
        _LOGGER.info("Setting HVAC mode for %s to %s", self.name, hvac_mode)
        
        # Convert string to HVACMode enum if needed
        if isinstance(hvac_mode, str):
            hvac_mode = HVACMode(hvac_mode)
            
        self._hvac_mode = hvac_mode
        self.async_write_ha_state()
        
    async def async_set_preset_mode(self, preset_mode):
        """Set new preset mode."""
        # In a real implementation, this would control the device
        _LOGGER.info("Setting preset mode for %s to %s", self.name, preset_mode)
        self._preset_mode = preset_mode
        self.async_write_ha_state()
        
    async def async_update(self):
        """Fetch new state data for the sensor."""
        # In a real implementation, this would query the device
        # For now, we'll just set some dummy values
        self._current_temp = 22.0
        self._floor_temp = 24.0
        self._target_temp = 21.5
        self._window_open = False
        self._heating_on = True
