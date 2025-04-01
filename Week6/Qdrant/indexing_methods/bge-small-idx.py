import uuid
import time
import psutil
import numpy as np
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
NUM_RECORDS = 1000
BATCH_SIZE = 100
MODEL_NAME = "BAAI/bge-small-en"

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=384, distance=Distance.COSINE)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=384, distance=Distance.COSINE, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=384, distance=Distance.COSINE)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=384, distance=Distance.COSINE, on_disk=True)}
]

# Initialize components
embedder = TextEmbedding(model_name=MODEL_NAME)
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Generate sample network data
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, np.random.choice(protocols, NUM_RECORDS))]

# Generate embeddings
vectors = []
for batch in [documents[i:i+BATCH_SIZE] for i in range(0, NUM_RECORDS, BATCH_SIZE)]:
    vectors.extend(list(embedder.embed(batch)))

def benchmark(operation):
    """Benchmarking utility without prints"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = operation()
    
    latency = (time.time() - start_time) * 1000
    cpu_usage = process.cpu_percent(interval=None) - cpu_before
    mem_usage = (process.memory_info().rss / (1024 ** 2)) - mem_before
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    return (latency, cpu_usage, mem_usage, throughput)

def run_benchmarks():
    results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
    
    for config in CONFIGURATIONS:
        collection_name = f"network_benchmark_{config['name'].lower().replace(' ', '_')}"
        
        # Collection management
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=config["vector_config"]
        )

        # Define operations
        def insert():
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vec.tolist(),
                    payload={"source": s, "destination": d, "protocol": p}
                ) for i, (s, d, p, vec) in enumerate(zip(sources, destinations, protocols, vectors))
            ]
            client.upsert(collection_name=collection_name, points=points)
            return points

        def search():
            return client.search(
                collection_name=collection_name,
                query_vector=vectors[0].tolist(),
                limit=1
            )

        def update():
            update_ids = ids[:100]
            new_docs = [f"Updated {doc}" for doc in documents[:100]]
            new_vectors = list(embedder.embed(new_docs))
            points = [
                PointStruct(
                    id=uid,
                    vector=vec.tolist(),
                    payload={"source": "updated", "destination": "updated", "protocol": "UPDATED"}
                ) for uid, vec in zip(update_ids, new_vectors)
            ]
            client.upsert(collection_name=collection_name, points=points)
            return points

        def delete():
            delete_ids = ids[:100]
            client.delete(collection_name=collection_name, points_selector=delete_ids)
            return delete_ids

        # Run benchmarks
        results["INSERT"].append((config["name"], *benchmark(insert)))
        results["SEARCH"].append((config["name"], *benchmark(search)))
        results["UPDATE"].append((config["name"], *benchmark(update)))
        results["DELETE"].append((config["name"], *benchmark(delete)))

        client.delete_collection(collection_name)

    # Format results
    headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
    table_data = []
    for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
        row = [op]
        for metric in results[op]:
            row.append(
                f"Latency: {metric[1]:.2f}ms\n"
                f"CPU: {metric[2]:.2f}%\n"
                f"Memory: {metric[3]:.2f}MB\n"
                f"Throughput: {metric[4]:.2f} ops/s"
            )
        table_data.append(row)
    
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    run_benchmarks()