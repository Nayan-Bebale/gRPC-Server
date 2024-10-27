import socket
import requests
import psutil
import struct
from ipwhois import IPWhois
from pprint import pprint
import json
import os

# Get local IP address and subnet mask
def get_local_ip_info():
    net_info = psutil.net_if_addrs()

    for interface, addrs in net_info.items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                return addr.address, addr.netmask  # Return local IP and subnet mask
    return None, None

# Calculate subnet address from local IP and subnet mask
def calculate_subnet(ip, netmask):
    try:
        ip_binary = struct.unpack('!I', socket.inet_aton(ip))[0]
        mask_binary = struct.unpack('!I', socket.inet_aton(netmask))[0]
        subnet_binary = ip_binary & mask_binary
        subnet = socket.inet_ntoa(struct.pack('!I', subnet_binary))
        return subnet
    except Exception as e:
        print("Error calculating subnet address:", e)
        return None

# Calculate CIDR notation from subnet mask
def calculate_cidr(netmask):
    mask_binary = struct.unpack('!I', socket.inet_aton(netmask))[0]
    prefix_length = bin(mask_binary).count('1')  # Count the number of 1s in binary representation
    return prefix_length

# Get public IP using an external service (ipify)
def get_public_ip():
    try:
        public_ip = requests.get('https://api.ipify.org').text
        return public_ip
    except Exception as e:
        print("Error getting public IP address:", e)
        return None

# Get ISP and location details using ipinfo.io
def get_ip_info(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json")
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching IP information:", e)
        return None

# Get ASN using ipwhois
def get_asn_info(ip):
    try:
        obj = IPWhois(ip)
        results = obj.lookup_rdap()
        return results['asn'], results['asn_description']
    except Exception as e:
        print("Error fetching ASN information:", e)
        return None, None

# Fetch and display network information
def load_data():
    # Get local network IP and subnet mask
    local_ip, subnet_mask = get_local_ip_info()
    ip_loc_data = {
        "service_provider": None,
        "city": None,
        "region": None,
        "country": None,
        "geo_location_coordinates": None,
        "latitude": None,  # Added field for latitude
        "longitude": None,  # Added field for longitude
        "asn": None,
        "asn_description": None,
        "local_ip": local_ip,
        "public_ip": None,
        "subnet_mask": subnet_mask,
        "subnet": None
    }

    if local_ip and subnet_mask:
        
        # Calculate the subnet address from the IP and subnet mask
        subnet = calculate_subnet(local_ip, subnet_mask)
        if subnet:
            # Calculate CIDR notation
            prefix_length = calculate_cidr(subnet_mask)
            cidr_notation = f"{subnet}/{prefix_length}"
            ip_loc_data["subnet"] = cidr_notation
            # print(f"Calculated Subnet Address in CIDR Notation: {cidr_notation}")
    else:
        print("Could not retrieve local network IP information.")
    
    # Fetch public IP and ISP information
    public_ip = get_public_ip()
    ip_loc_data["public_ip"] = public_ip
    
    if public_ip:
        
        ip_info = get_ip_info(public_ip)
        
        
        if ip_info:
            # Print IP details
            ip_loc_data["service_provider"] = ip_info.get('org', 'Not available')
            ip_loc_data["city"] = ip_info.get('city', 'Not available')
            ip_loc_data["region"] = ip_info.get('region', 'Not available')
            ip_loc_data["country"] = ip_info.get('country', 'Not available')
            ip_loc_data["geo_location_coordinates"] = ip_info.get('loc', 'Not available') 

            # Extract latitude and longitude from the geo_location_coordinates
            if ip_loc_data["geo_location_coordinates"] != 'Not available':
                coordinates = ip_loc_data["geo_location_coordinates"].split(',')
                if len(coordinates) == 2:
                    ip_loc_data["latitude"] = float(coordinates[0])  # First part is latitude
                    ip_loc_data["longitude"] = float(coordinates[1])  # Second part is longitude
        else:
            print("Could not retrieve public IP information.")
        
        # Fetch ASN information using ipwhois
        asn, asn_description = get_asn_info(public_ip)
        asn = 'AS'+asn
        if asn and asn_description:
            # print(f"Autonomous System (ASN): {asn}")
            # print(f"ASN Description: {asn_description}")
            ip_loc_data["asn"] = asn
            ip_loc_data["asn_description"] = asn_description
        else:
            print("ASN information not available.")
    else:
        print("Could not retrieve public IP address.")

    return ip_loc_data



def display_ip_info():
    file_path = "server_data_cache.json"  # Path to the cached file
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load data from the file
        with open(file_path, 'r') as file:
            cached_data = json.load(file)
        return cached_data
    
    # If file does not exist, collect the data
    server_data = load_data()
    # Save the collected data to the file
    with open(file_path, 'w') as file:
        json.dump(server_data, file)
    
    return server_data
