import pyshark
import psycopg2
from sentence_transformers import SentenceTransformer
import time
import psutil
import os

model = SentenceTransformer("all-MiniLM-L6-v2") # 384 Dimensions

# PostgreSQL connection details

DB_NAME = "database_name" # Change it
DB_USER = "admin" # Change it
DB_PASSWORD = "password" # Change it
DB_HOST = "localhost"
DB_PORT = "5432"      

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
            "info": str(packet)[:1000]
        }
        return row
    except AttributeError:
        return None  # Skip malformed packets

def generate_embedding(row):
    """Converts extracted features into a vector embedding."""
    text_data = f"{row['protocol']} {row['src']} {row['dst']} {row['length']} {row['info']}"
    return model.encode([text_data])[0].tolist()

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

if __name__ == "__main__":
    pcap_file = "dataset.pcapng"

    # Establishing database connection
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    cap = pyshark.FileCapture(pcap_file)

    total_insert_time = 0
    packet_count = 0

    # Getting process CPU times before starting (user + system times)
    process = psutil.Process(os.getpid())
    cpu_times_before = process.cpu_times()
    overall_start_time = time.time()

    for i, packet in enumerate(cap):
        row = extract_packet_data(packet, i + 1)
        if row:
            vector = generate_embedding(row)
            insert_start_time = time.time()
            insert_into_db(conn, row, vector)
            insert_end_time = time.time()
            
            total_insert_time += (insert_end_time - insert_start_time)
            packet_count += 1

    overall_end_time = time.time()
    cpu_times_after = process.cpu_times()

    conn.close()

    total_execution_time = overall_end_time - overall_start_time
    avg_insert_time = total_insert_time / packet_count if packet_count > 0 else 0

    # Calculate overall CPU utilization for the process:
    # (Total CPU time used during insertion / Total elapsed time) * 100
    cpu_time_before = cpu_times_before.user + cpu_times_before.system
    cpu_time_after = cpu_times_after.user + cpu_times_after.system
    delta_cpu_time = cpu_time_after - cpu_time_before
    overall_cpu_usage = (delta_cpu_time / total_execution_time) * 100 if total_execution_time > 0 else 0

    print(f"Successfully inserted {packet_count} packets into pgvector!")
    print(f"Total Execution Time: {total_execution_time:.6f}s")
    print(f"Average Insertion Time per Packet: {avg_insert_time:.6f}s")
    print(f"Overall CPU Utilization: {overall_cpu_usage:.2f}%")