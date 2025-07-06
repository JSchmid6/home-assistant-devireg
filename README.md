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
This integration uses a discovery process to find your Danfoss devices. You will need to get a One-Time Password (OTP) from your Danfoss mobile app to start the discovery.

**Getting the OTP:**
1. Open the Danfoss mobile app on your phone.
2. Go to the settings for your home.
3. Select "Add user" or "Share home".
4. The app will generate a code (the OTP). This code is valid for a short time.

**Adding the Integration:**
1. Go to **Settings** → **Devices & Services**.
2. Click **+ Add Integration**.
3. Search for "Danfoss Heating".
4. Enter the OTP you generated from the mobile app.
5. The integration will discover your devices. Select the device you want to add from the list.

## Usage
After configuration, your thermostat will appear as a climate entity. You can control it through the Home Assistant UI or automations.

## Troubleshooting
If your devices are not discovered:
1.  **Check the OTP:** Ensure you have entered the correct OTP and that it has not expired.
2.  **Check Network Connectivity:** Ensure that your Home Assistant instance and your phone are on the same network.
3.  **Check Home Assistant Logs:** Look for any errors related to the `danfoss_heating` integration in **Settings** → **System** → **Logs**.

## Support
For issues or feature requests, please open an issue on [GitHub](https://github.com/JSchmid6/home-assistant-devireg).

## License
This integration is licensed under the Eclipse Public License 2.0.

---
*Home Assistant integration implemented by Cline (AI assistant) based on the openHAB binding by Sonic-Amiga*
