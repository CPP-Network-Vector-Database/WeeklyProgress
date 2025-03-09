import psycopg2
from scapy.all import sniff, IP, TCP, UDP
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="your_database_name",     # Replace with your database name
    user="your_username",       # Replace with your username
    password="your_password",  # Replace with your password
    host="localhost",          # Replace with your host, e.g., localhost or IP
    port="5432"                # Default PostgreSQL port
)


# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create the table to store packet data and embeddings if it doesn't exist
cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS packet_data (
        id SERIAL PRIMARY KEY,
        src_ip TEXT,
        dst_ip TEXT,
        protocol TEXT,
        payload TEXT,
        embedding VECTOR(384)
    );
""")
conn.commit()

# Function to process each captured packet
def process_packet(packet):
    if IP in packet:
        # Extract useful packet data
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto

        # Handle TCP/UDP payloads (this is just an example, you might want to adjust based on your needs)
        if TCP in packet:
            payload = str(packet[TCP].payload)
        elif UDP in packet:
            payload = str(packet[UDP].payload)
        else:
            payload = ""

        # Generate embedding for the extracted data (src_ip, dst_ip, protocol, payload)
        data_to_embed = f"From {src_ip} to {dst_ip} using protocol {protocol}. Payload: {payload[:100]}"  # Shortened payload for embedding
        embedding = model.encode([data_to_embed])[0]  # Get the embedding

        # Insert the packet data and embedding into the PostgreSQL database
        cursor.execute("""
            INSERT INTO packet_data (src_ip, dst_ip, protocol, payload, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (src_ip, dst_ip, protocol, payload, embedding.tolist()))
        conn.commit()

# Start sniffing packets on a specific interface (e.g., 'eth0' or 'wlan0') or all packets
# You can also limit the number of packets, like sniff(count=10)
print("Starting packet capture...")
sniff(prn=process_packet, store=0, count=100)  # Adjust 'count' to control how many packets to capture
print("Packet capture finished.")

# Close the database connection when done
cursor.close()
conn.close()
