# Danfoss Heating Integration for Home Assistant

This integration allows you to control Danfoss heating devices through Home Assistant, including DeviReg Smart thermostats and Icon Room controllers.

![Integration Preview](https://via.placeholder.com/800x400?text=Danfoss+Integration+Preview)

## Features
- Control multiple thermostat types
- Set target temperatures
- Switch between HVAC modes (heat/off)
- Use preset modes (comfort, economy, etc.)
- Monitor floor temperature
- Window detection status
- Heating activity indicator

## Installation

### HACS Installation (Recommended)
1. Open HACS in Home Assistant
2. Go to Integrations
3. Click "+ Explore & Download Repositories"
4. Search for "Danfoss Heating"
5. Click "Download"
6. Restart Home Assistant

### Manual Installation
1. Download the `danfoss_heating` folder
2. Copy it to your `custom_components` directory
3. Restart Home Assistant

## Configuration
1. Go to **Settings** → **Devices & Services**
2. Click **+ Add Integration**
3. Search for "Danfoss Heating"
4. Enter your credentials:
   - Username: Your Danfoss account name
   - Private Key: 64-character hex key
   - Public Key: 64-character hex key
5. Click **Submit**

## Usage
After configuration, your thermostats will appear as climate entities. You can control them through the Home Assistant UI or automations.

### Services
- `danfoss_heating.set_temperature`: Set target temperature
- `danfoss_heating.set_mode`: Change HVAC mode
- `danfoss_heating.set_preset`: Activate preset mode

## Troubleshooting
If devices don't appear:
1. Verify your keys are correct
2. Check network connectivity
3. Ensure devices are online in the Danfoss app
4. Check Home Assistant logs for errors

## Support
For issues or feature requests, please open an issue on [GitHub](https://github.com/Sonic-Amiga/org.openhab.binding.devireg)

## License
This integration is licensed under the Eclipse Public License 2.0

---
*Home Assistant integration implemented by Cline (AI assistant) based on the openHAB binding by Sonic-Amiga*
