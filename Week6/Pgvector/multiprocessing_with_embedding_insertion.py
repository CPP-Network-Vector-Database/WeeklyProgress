import pyshark
import psycopg2
import time
import psutil
import os
import multiprocessing
import subprocess
import asyncio
from sentence_transformers import SentenceTransformer

# Load the embedding model globally to avoid reloading it in each process
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# PostgreSQL connection details

DB_NAME = "database_name" # Change it
DB_USER = "admin" # Change it
DB_PASSWORD = "password" # Change it
DB_HOST = "localhost"
DB_PORT = "5432"       

def get_split_filenames():
    """
    Uses a shell command to list files containing 'split' in their names.
    Only returns filenames that end with '.pcapng'.
    """
    command = "ls -ila | grep split | awk '{print $NF}'"
    try:
        result = subprocess.check_output(command, shell=True, text=True)
        filenames = result.splitlines()
        filenames = [f for f in filenames if f.endswith('.pcapng')]
        return filenames
    except subprocess.CalledProcessError as e:
        print("Error fetching filenames:", e)
        return []

def extract_packet_data(packet, packet_no):
    """Extracts features from a single packet."""
    try:
        row = {
            "no": packet_no,
            "time": str(packet.sniff_time),
            "src": getattr(packet.ip, "src", "N/A"),
            "dst": getattr(packet.ip, "dst", "N/A"),
            "protocol": packet.highest_layer,
            "length": packet.length,
            "info": str(packet)[:1000]  # Limiting packet info length
        }
        return row
    except AttributeError:
        return None  # Skip malformed packets

def generate_embedding(text):
    """Generates an embedding for the given packet info using all-MiniLM-L6-v2."""
    return embedding_model.encode(text).tolist()

def insert_into_db(conn, row, vector):
    """Inserts extracted data and vector into PostgreSQL (pgvector)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pcap_embeddings (no, time, src, dst, protocol, length, info, vector) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (row["no"], row["time"], row["src"], row["dst"], row["protocol"], row["length"], row["info"], vector)
        )
        conn.commit()

def process_pcap_file(pcap_file):
    """
    Worker function to process a single pcap file:
      - Creates a new asyncio event loop for this process.
      - Establishes its own database connection.
      - Processes packets from the assigned pcap file.
      - Uses real embeddings for packet data.
      - Prints the number of packets processed.
    """
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    print(f"Processing {pcap_file} ...")
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    
    processed_packets = 0
    packet_no = 1
    
    try:
        cap = pyshark.FileCapture(pcap_file)
        for packet in cap:
            row = extract_packet_data(packet, packet_no)
            if row:
                vector = generate_embedding(row["info"])
                insert_into_db(conn, row, vector)
                processed_packets += 1
            packet_no += 1
    finally:
        cap.close()
        del cap
        conn.close()
    
    print(f"Finished processing {pcap_file}: Processed {processed_packets} packets.")

if __name__ == "__main__":
    pcap_files = get_split_filenames()
    if not pcap_files:
        print("No split files found. Exiting.")
        exit(1)

    print("Found the following split files:")
    for file in pcap_files:
        print(file)

    num_workers = max(multiprocessing.cpu_count() - 2, 1)
    num_workers = min(num_workers, len(pcap_files))

    overall_start_time = time.time()
    process_obj = psutil.Process(os.getpid())
    cpu_times_before = process_obj.cpu_times()

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_pcap_file, pcap_files)

    overall_end_time = time.time()
    cpu_times_after = process_obj.cpu_times()

    total_execution_time = overall_end_time - overall_start_time
    cpu_time_before = cpu_times_before.user + cpu_times_before.system
    cpu_time_after = cpu_times_after.user + cpu_times_after.system
    delta_cpu_time = cpu_time_after - cpu_time_before
    overall_cpu_usage = (delta_cpu_time / total_execution_time) * 100 if total_execution_time > 0 else 0

    print("Processing complete!")
    print(f"Total Execution Time: {total_execution_time:.6f}s")
    print(f"Overall CPU Utilization: {overall_cpu_usage:.2f}%")
