import uuid
import time
import psutil
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "network_benchmark"
NUM_RECORDS = 1000
BATCH_SIZE = 100
MODEL_NAME = "BAAI/bge-small-en"

# Initialize components
embedder = TextEmbedding(model_name=MODEL_NAME)
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create/recreate collection
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # bge-small-en has 384-dim embeddings
)

# Generate sample network data and store UUIDs
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, np.random.choice(protocols, NUM_RECORDS))]

# Generate embeddings in batches using FastEmbed
vectors = []
for batch in [documents[i:i+BATCH_SIZE] for i in range(0, NUM_RECORDS, BATCH_SIZE)]:
    vectors.extend(list(embedder.embed(batch)))

def benchmark(operation_name, function):
    """Benchmarking utility function"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = function()
    
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)
    latency = (time.time() - start_time) * 1000
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    print(f"🔹 {operation_name}")
    print(f"   🕒 Latency: {latency:.2f} ms")
    print(f"   🔥 CPU Usage: {cpu_after - cpu_before:.2f}%")
    print(f"   🧠 Memory Used: {mem_after - mem_before:.2f} MB")
    print(f"   ⚡ Throughput: {throughput:.2f} ops/sec")
    print("-" * 50)
    return result

# CRUD Operations
def insert_vectors():
    points = [
        PointStruct(
            id=ids[i],
            vector=vectors[i].tolist(),
            payload={"source": s, "destination": d, "protocol": p}
        ) for i, (s, d, p) in enumerate(zip(sources, destinations, protocols))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return points

def search_vectors(): #high.. switch with old code
    results = []
    for vec in vectors[:10]:  # Search first 10 vectors
        results.extend(client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec.tolist(),
            limit=5
        ))
    return results

def update_vectors(): #high.. switch with old code
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
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return points

def delete_vectors(): #check n see
    delete_ids = ids[:100]
    client.delete(collection_name=COLLECTION_NAME, points_selector=delete_ids)
    return delete_ids

# Run benchmarks
print("Starting benchmarks...\n")
benchmark("INSERT (CREATE)", insert_vectors)
benchmark("SEARCH (READ)", search_vectors)
benchmark("UPDATE", update_vectors)
benchmark("DELETE", delete_vectors)