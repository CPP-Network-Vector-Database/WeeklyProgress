import psycopg2
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import time

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="your_database_name",     # Replace with your database name
    user="your_username",       # Replace with your username
    password="your_password",  # Replace with your password
    host="localhost",          # Replace with your host, e.g., localhost or IP
    port="5432"                # Default PostgreSQL port
)

# Create a cursor object to interact with the database
cur = conn.cursor()

# Create table for storing packet information including metadata
cur.execute("""
    CREATE TABLE IF NOT EXISTS packet_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_ip VARCHAR(50),
        dest_ip VARCHAR(50),
        protocol VARCHAR(10),
        source_port INT,
        dest_port INT,
        packet_size INT,
        ip_flags VARCHAR(50),
        tcp_flags VARCHAR(50),
        ttl INT,
        raw_data BYTEA
    );
""")
conn.commit()

# Function to store captured packet data into the database
def store_packet_data(packet):
    try:
        if IP in packet:
            source_ip = packet[IP].src
            dest_ip = packet[IP].dst
            protocol = packet[IP].proto
            packet_size = len(packet)  # Total size of the packet
            ttl = packet[IP].ttl       # Time to live (TTL) for the IP packet
            ip_flags = str(packet[IP].flags)  # IP flags (e.g., DF, MF)

            # Handling TCP packets
            if TCP in packet:
                source_port = packet[TCP].sport
                dest_port = packet[TCP].dport
                tcp_flags = packet[TCP].flags  # TCP flags (e.g., SYN, ACK)
            # Handling UDP packets
            elif UDP in packet:
                source_port = packet[UDP].sport
                dest_port = packet[UDP].dport
                tcp_flags = None  # No TCP flags for UDP
            else:
                source_port = None
                dest_port = None
                tcp_flags = None
            
            # Storing raw packet data for further analysis
            raw_data = bytes(packet)  # Convert the packet to raw byte data

            # Insert packet information into the database
            cur.execute("""
                INSERT INTO packet_data (source_ip, dest_ip, protocol, source_port, dest_port, 
                                         packet_size, ip_flags, tcp_flags, ttl, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (source_ip, dest_ip, protocol, source_port, dest_port,
                  packet_size, ip_flags, tcp_flags, ttl, raw_data))
            conn.commit()
            print(f"Stored packet: {source_ip} -> {dest_ip}, Protocol: {protocol}, Size: {packet_size} bytes")
    
    except Exception as e:
        print(f"Error storing packet data: {e}")

# Function to start sniffing packets and call the store function
def start_sniffing():
    print("Starting to sniff packets...")
    sniff(prn=store_packet_data, store=0, filter="ip", count=100) 

# Start packet sniffing
start_sniffing()

# Close the cursor and connection
cur.close()
conn.close()
