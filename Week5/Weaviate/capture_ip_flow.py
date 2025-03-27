from scapy.all import sniff, IP
import csv
from datetime import datetime

csv_filename = "ip_flows.csv"

def process_packet(packet):
    if IP in packet:
        flow_data = [
            packet[IP].src,
            packet[IP].dst,
            packet.proto,
            len(packet),
            datetime.now().isoformat()
        ]
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(flow_data)


with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["source_ip", "destination_ip", "protocol", "packet_size", "timestamp"])


print("capturing IP flows... press Ctrl+C to stop.")
sniff(prn=process_packet, store=False)
