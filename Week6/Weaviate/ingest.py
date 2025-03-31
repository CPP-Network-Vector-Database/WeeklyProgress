import csv
import weaviate
import time
from sentence_transformers import SentenceTransformer

client = weaviate.Client("http://localhost:8080")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_ip_flow_embedding(flow_data):
    flow_text = f"{flow_data['source_ip']} {flow_data['destination_ip']} {flow_data['protocol']} {flow_data['packet_size']} {flow_data['timestamp']}"
    return embed_model.encode(flow_text).tolist()

def insert_ip_flows(csv_file):
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        start_time = time.time()
        for row in reader:
            data_object = {
                "source_ip": row["source_ip"],
                "destination_ip": row["destination_ip"],
                "protocol": row["protocol"],
                "packet_size": int(row["packet_size"]),
                "timestamp": row["timestamp"]
            }
            vector_embedding = create_ip_flow_embedding(data_object)
            client.data_object.create(data_object, "IPFlow", vector=vector_embedding)
        duration = time.time() - start_time
    print(f"Ingestion Time: {duration:.4f} seconds")
    print("IP Flows ingested successfully with vector embeddings!")


