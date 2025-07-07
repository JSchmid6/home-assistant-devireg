"""The Danfoss Heating integration."""
import asyncio
import logging
import json

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
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
)
from .pysdg import SDGPeerConnector, DeviReg, IconRoom
from .pysdg_cloud.connector import DanfossCloudConnector

PLATFORMS = ["climate", "sensor", "switch", "select"]

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Danfoss Heating from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    connection_type = entry.data.get(CONF_CONNECTION_TYPE, CONNECTION_TYPE_LOCAL)

    if connection_type == CONNECTION_TYPE_CLOUD:
        devices_info = entry.data["devices"]
        hass.data[DOMAIN][entry.entry_id] = {"devices": devices_info}
    else:
        peer_id = entry.data[CONF_PEER_ID]
        device_type = entry.data[CONF_DEVICE_TYPE]
        host = entry.data[CONF_HOST]

        connector = SDGPeerConnector(peer_id, host)
        await connector.connect()

        if not connector.is_connected:
            _LOGGER.error("Failed to connect to device")
            return False

        _LOGGER.debug("Creating device of type: %s", device_type)
        if device_type == DEVICE_TYPE_DEVISMART:
            device = DeviReg(connector)
        elif device_type == DEVICE_TYPE_ICON_ROOM:
            room_number = entry.data[CONF_ROOM_NUMBER]
            device = IconRoom(connector, room_number)
        else:
            return False

        hass.data[DOMAIN][entry.entry_id] = {"device": device}

    _LOGGER.debug("Forwarding setup to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading config entry: %s", entry.entry_id)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        
    return unload_ok

async def async_setup_from_discovery(hass: HomeAssistant, config: dict):
    """Set up the Danfoss Heating component from discovery."""
    with open("danfoss_devices.json", "r") as f:
        devices = json.load(f)

    for device in devices:
        # For now, we'll assume all discovered devices are DeviSmart
        device[CONF_DEVICE_TYPE] = DEVICE_TYPE_DEVISMART
        
        # Check if the device is already configured
        if not any(
            entry.data[CONF_PEER_ID] == device[CONF_PEER_ID]
            for entry in hass.config_entries.async_entries(DOMAIN)
        ):
            hass.async_create_task(
                hass.config_entries.flow.async_init(
                    DOMAIN, context={"source": "discovery"}, data=device
                )
            )
