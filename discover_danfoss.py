import json
import socket
from scapy.all import ARP, Ether, srp

def discover_danfoss_devices(network="192.168.1.0/24"):
    """
    Discovers Danfoss devices on the network by scanning for known MAC address prefixes.
    """
    print(f"Scanning network {network} for Danfoss devices...")
    
    # Danfoss MAC address prefixes
    danfoss_prefixes = ["00:03:56"]

    # Create an ARP request packet
    arp = ARP(pdst=network)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether/arp

    # Send the packet and receive the responses
    result = srp(packet, timeout=3, verbose=0)[0]

    devices = []
    for sent, received in result:
        mac_address = received.hwsrc
        ip_address = received.psrc
        
        for prefix in danfoss_prefixes:
            if mac_address.startswith(prefix):
                print(f"Found potential Danfoss device: {ip_address} ({mac_address})")
                
                # Attempt to connect to the device to confirm it's a Danfoss heating controller
                try:
                    with socket.create_connection((ip_address, 80), timeout=2):
                        print(f"Successfully connected to {ip_address}. Adding to list.")
                        devices.append({
                            "host": ip_address,
                            "peer_id": mac_address
                        })
                except (socket.timeout, ConnectionRefusedError):
                    print(f"Could not connect to {ip_address}. Skipping.")
                
                break # Move to the next device

    return devices

if __name__ == "__main__":
    discovered_devices = discover_danfoss_devices()
    
    if discovered_devices:
        with open("danfoss_devices.json", "w") as f:
            json.dump(discovered_devices, f, indent=4)
        print(f"\nDiscovered {len(discovered_devices)} devices. Saved to danfoss_devices.json")
    else:
        print("\nNo Danfoss devices found.")
