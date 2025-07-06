# Danfoss Heating Integration for Home Assistant

This integration allows you to control Danfoss heating devices through Home Assistant, including DeviReg Smart thermostats and Icon Room controllers.

## Features
- Control multiple thermostat types (DeviReg Smart and Icon Room).
- Set target temperatures.
- Basic monitoring of device state.
- *Note: This integration is currently under development. More features will be added in the future.*

## Installation

### HACS Installation (Recommended)
1. Open HACS in Home Assistant.
2. Go to Integrations.
3. Click "+ Explore & Download Repositories".
4. Search for "Danfoss Heating".
5. Click "Download".
6. Restart Home Assistant.

### Manual Installation
1. Download the `danfoss_heating` folder from the `home_assistant_integration/custom_components` directory.
2. Copy it to your `custom_components` directory in your Home Assistant configuration folder.
3. Restart Home Assistant.

## Configuration
1. Go to **Settings** → **Devices & Services**.
2. Click **+ Add Integration**.
3. Search for "Danfoss Heating".
4. Enter the required information:
   - **Peer ID**: This is the MAC address of your Danfoss device. You can usually find this on a sticker on the device itself, or in your router's list of connected devices. It should be in the format `XX:XX:XX:XX:XX:XX`.
   - **Device Type**: Select whether you are adding a `DeviSmart` or an `IconRoom` device.
   - **Room Number** (for IconRoom only): If you are adding an `IconRoom` device, you must specify the room number.

5. Click **Submit**.

## Usage
After configuration, your thermostat will appear as a climate entity. You can control it through the Home Assistant UI or automations.

## Troubleshooting
If your device doesn't appear or is unavailable:
1.  **Verify the Peer ID:** Double-check that the Peer ID (MAC address) is correct.
2.  **Check Network Connectivity:** Ensure that your Home Assistant instance and the Danfoss device are on the same network. The integration communicates locally with the device.
3.  **Check Home Assistant Logs:** Look for any errors related to the `danfoss_heating` integration in **Settings** → **System** → **Logs**.

## Support
For issues or feature requests, please open an issue on [GitHub](https://github.com/JSchmid6/home-assistant-devireg).

## License
This integration is licensed under the Eclipse Public License 2.0.

---
*Home Assistant integration implemented by Cline (AI assistant) based on the openHAB binding by Sonic-Amiga*
