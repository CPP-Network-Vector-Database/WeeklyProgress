from scapy.all import rdpcap, IP

def extract_pcap_data(pcap_file):
    packets = []
    
    try:
        pcap = rdpcap(pcap_file)
        for pkt in pcap:
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                protocol = pkt[IP].proto  # IP protocol number
                
                packets.append(f"{src_ip} {dst_ip} {protocol}")

        print(f"Extracted {len(packets)} packets.")
        print("Sample packets:", packets[:5])

    except Exception as e:
        print("Error reading PCAP file:", str(e))

    return packets
