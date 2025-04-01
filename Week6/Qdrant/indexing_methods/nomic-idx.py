import uuid
import time
import psutil
import numpy as np
import torch
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoModel, AutoTokenizer

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
NUM_RECORDS = 1000
BATCH_SIZE = 32  # Reduced for CPU memory constraints
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=768, distance=Distance.COSINE)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=768, distance=Distance.COSINE)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)}
]

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Generate sample network data
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, np.random.choice(protocols, NUM_RECORDS))]

def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = (outputs.last_hidden_state * inputs.attention_mask.unsqueeze(-1)).sum(dim=1)
    embeddings = embeddings / inputs.attention_mask.sum(dim=1, keepdim=True)
    return embeddings.numpy()

vectors = []
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    vectors.extend(generate_embeddings(batch))

def benchmark(op_name, operation):
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = operation()
    
    latency = (time.time() - start_time) * 1000
    cpu_usage = process.cpu_percent(interval=None) - cpu_before
    mem_usage = (process.memory_info().rss / (1024 ** 2)) - mem_before
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    return latency, cpu_usage, mem_usage, throughput

def run_benchmarks():
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
    
    for config in CONFIGURATIONS:
        print(f"\n{'='*40}\nRunning {config['name']} configuration\n{'='*40}")
        collection_name = f"benchmark_{config['name'].lower().replace(' ', '_')}"
        
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(collection_name=collection_name, vectors_config=config["vector_config"])
        
        def insert():
            points = [PointStruct(id=ids[i], vector=vector.tolist(), payload={"source": s, "destination": d, "protocol": p})
                      for i, (vector, s, d, p) in enumerate(zip(vectors, sources, destinations, protocols))]
            client.upsert(collection_name=collection_name, points=points)
            return points
        
        def search():
            return client.search(collection_name=collection_name, query_vector=vectors[0], limit=1)
        
        def update():
            updated_texts = [f"Updated text {i}" for i in range(100)]
            updated_vectors = generate_embeddings(updated_texts)
            points = [PointStruct(id=ids[i], vector=vec.tolist(), payload={"source": "updated", "destination": "updated", "protocol": "UPDATED"})
                      for i, vec in enumerate(updated_vectors)]
            client.upsert(collection_name=collection_name, points=points)
            return points
        
        def delete():
            delete_ids = ids[:100]
            client.delete(collection_name=collection_name, points_selector=delete_ids)
            return delete_ids
        
        results["INSERT"].append((config["name"], *benchmark("INSERT", insert)))
        results["SEARCH"].append((config["name"], *benchmark("SEARCH", search)))
        results["UPDATE"].append((config["name"], *benchmark("UPDATE", update)))
        results["DELETE"].append((config["name"], *benchmark("DELETE", delete)))
        
        client.delete_collection(collection_name)
    
    headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
    table_data = []
    for op, metrics in results.items():
        row = [op]
        for metric in metrics:
            row.append(f"Latency: {metric[1]:.2f}ms\nCPU: {metric[2]:.2f}%\nMemory: {metric[3]:.2f}MB\nThroughput: {metric[4]:.2f} ops/s")
        table_data.append(row)
    
    print("\n\nBenchmark Results Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    run_benchmarks()
