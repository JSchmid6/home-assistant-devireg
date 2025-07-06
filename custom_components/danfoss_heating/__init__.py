"""The Danfoss Heating integration."""
import asyncio

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_PEER_ID, CONF_DEVICE_TYPE, DEVICE_TYPE_DEVISMART, DEVICE_TYPE_ICON_ROOM, CONF_ROOM_NUMBER, CONF_HOST
from .pysdg import SDGPeerConnector, DeviReg, IconRoom

PLATFORMS = ["climate", "sensor", "switch"]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Danfoss Heating from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    peer_id = entry.data[CONF_PEER_ID]
    device_type = entry.data[CONF_DEVICE_TYPE]
    host = entry.data[CONF_HOST]
    
    connector = SDGPeerConnector(peer_id, host)
    await connector.connect()
    
    if not connector.is_connected:
        return False
        
    if device_type == DEVICE_TYPE_DEVISMART:
        device = DeviReg(connector)
    elif device_type == DEVICE_TYPE_ICON_ROOM:
        room_number = entry.data[CONF_ROOM_NUMBER]
        device = IconRoom(connector, room_number)
    else:
        return False
        
    hass.data[DOMAIN][entry.entry_id] = device
    
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        
    return unload_ok
