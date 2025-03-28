import uuid
import time
import psutil
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoModel, AutoTokenizer

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "nomic_benchmark"
NUM_RECORDS = 1000
BATCH_SIZE = 32  # Reduced for CPU memory constraints
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Initialize components
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create/recreate collection
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Nomic uses 768-dim embeddings
)

# Generate sample network data
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, np.random.choice(protocols, NUM_RECORDS))]

# Generate embeddings with Nomic's special handling
def generate_embeddings(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = (outputs.last_hidden_state * inputs.attention_mask.unsqueeze(-1)).sum(dim=1)
    embeddings = embeddings / inputs.attention_mask.sum(dim=1, keepdim=True)
    return embeddings.numpy()

# Batch embedding generation
vectors = []
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    vectors.extend(generate_embeddings(batch))

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
    
    print(f" {operation_name}")
    print(f"    Latency: {latency:.2f} ms")
    print(f"    CPU Usage: {cpu_after - cpu_before:.2f}%")
    print(f"    Memory Used: {mem_after - mem_before:.2f} MB")
    print(f"    Throughput: {throughput:.2f} ops/sec")
    print("-" * 50)
    return result

# CRUD Operations
def insert_vectors():
    points = [
        PointStruct(
            id=ids[i],
            vector=vector.tolist(),
            payload={"source": s, "destination": d, "protocol": p}
        ) for i, (vector, s, d, p) in enumerate(zip(vectors, sources, destinations, protocols))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return points

# 2️⃣ Search Operation
def search_vectors():
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vectors[0],  # Search using the first vector
        limit=1
    )

# 3️⃣ Update Operation (fixed)
def update_vectors():
    updated_texts = [f"Updated text {i}" for i in range(100)]
    updated_vectors = generate_embeddings(updated_texts)
    
    # Create points list and return it
    points = [
        PointStruct(
            id=ids[i],  # Use original UUIDs instead of index 'i'
            vector=vec.tolist(),
            payload={"source": "updated", "destination": "updated", "protocol": "UPDATED"}
        ) for i, vec in enumerate(updated_vectors)
    ]
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return points  # This fixes the NoneType error

def delete_vectors(): #fluctuates.. prev code got lowest (9ms)
    delete_ids = ids[:100]
    client.delete(collection_name=COLLECTION_NAME, points_selector=delete_ids)
    return delete_ids

# Run benchmarks
print("Starting Nomic AI benchmarks...\n")
benchmark("INSERT (CREATE)", insert_vectors)
benchmark("SEARCH (READ)", search_vectors)
benchmark("UPDATE", update_vectors)
benchmark("DELETE", delete_vectors)
