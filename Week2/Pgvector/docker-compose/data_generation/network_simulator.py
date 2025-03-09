import requests
import random
import time
import socket
import string
import threading
from scapy.all import ARP, Ether, srp 

# Get the container's hostname and IP (Docker assigns internal IPs)
HOSTNAME = socket.gethostname()
MY_IP = socket.gethostbyname(HOSTNAME)

# Network settings
PORT_TCP = 5000
PORT_UDP = 5001
NETWORK_PREFIX = "172.20.0."  # Change based on Availabilty of Docker network

# List of sample websites
websites = ["https://www.google.com", "https://www.youtube.com", "https://x.com", "https://www.microsoft.com/en-in"]

# List of DNS queries
dns_queries = ["google.com", "yahoo.com", "bing.com", "cloudflare.com"]

# Function to generate a random string
def random_message(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# ARP request function to detect network devices
def send_arp_request():
    target_ip = "192.168.1.1/24"  # Adjust this based on your network
    arp = ARP(pdst=target_ip)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether / arp
    answered, _ = srp(packet, timeout=2, verbose=False)
    
    devices = [f"MAC: {resp.hwsrc}, IP: {resp.psrc}" for _, resp in answered]
    print("[ARP] Devices found:", devices)

# Function to perform HTTP requests
def send_http_request():
    url = random.choice(websites)
    try:
        response = requests.get(url, timeout=3)
        print(f"[HTTP] {url} -> {response.status_code}")
    except requests.RequestException as e:
        print(f"[HTTP] Error: {e}")

# Function to perform DNS queries
def send_dns_query():
    domain = random.choice(dns_queries)
    try:
        ip = socket.gethostbyname(domain)
        print(f"[DNS] {domain} -> {ip}")
    except socket.gaierror:
        print(f"[DNS] Failed to resolve {domain}")

# Function to get a random IP in the Docker network
def get_random_host():
    while True:
        random_ip = f"{NETWORK_PREFIX}{random.randint(2, 16)}"
        if random_ip != MY_IP:
            return random_ip

# Function to randomly choose and execute a network operation with weighted probability
def simulate_network_traffic():
    operations = ["http"] * 10 + ["dns"] * 50 + ["arp"] * 40
    while True:
        action = random.choice(operations)
        if action == "http":
            send_http_request()
        elif action == "dns":
            send_dns_query()
        elif action == "arp":
            send_arp_request()
        
        time.sleep(random.uniform(15, 20))  # Increased delay and adjust it accordingly

# Start network traffic simulation
simulate_network_traffic()
