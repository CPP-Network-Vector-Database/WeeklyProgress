import time
import psutil
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, VectorParams, Distance
from transformers import AutoModel, AutoTokenizer
import torch

# Define Qdrant settings
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "benchmark_vectors"

# Connect to Qdrant
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create collection with proper existence check
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# Initialize Nomic embedding model
tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

# Generate embeddings function
def generate_embeddings(texts):
    inputs = tokenizer(texts, 
                      padding=True, 
                      return_tensors="pt", 
                      max_length=512, 
                      truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = (outputs.last_hidden_state * inputs.attention_mask.unsqueeze(-1)).sum(dim=1)
    embeddings = embeddings / inputs.attention_mask.sum(dim=1, keepdim=True)
    return embeddings.numpy()

# Generate sample data
texts = [f"Sample text {i}" for i in range(1000)]
vectors = generate_embeddings(texts)

# Benchmarking utility
def benchmark(operation_name, function):
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)

    result = function()

    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)
    latency = (time.time() - start_time) * 1000
    memory_used = max(mem_after - mem_before, 0)  # Prevent negative values
    throughput = len(vectors)/(latency/1000) if latency > 0 else 0

    print(f"🔹 {operation_name}")
    print(f"   🕒 Latency: {latency:.2f} ms")
    print(f"   🔥 CPU Usage: {cpu_after - cpu_before:.2f}%")
    print(f"   🧠 Memory Used: {memory_used:.2f} MB")
    print(f"   ⚡ Throughput: {throughput:.2f} ops/sec")
    print("-" * 50)
    return result

# 1️⃣ Insert Operation (working - kept as-is)
def insert_vectors():
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i, vector=vec.tolist())  # Convert numpy to list
            for i, vec in enumerate(vectors)
        ]
    )

# 2️⃣ Search Operation (updated to modern API)
def search_vectors():
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vectors[0].tolist(),  # Convert to list
        limit=1
    )

# 3️⃣ Update Operation (fixed)
def update_vectors():
    updated_texts = [f"Updated text {i}" for i in range(100)]
    updated_vectors = generate_embeddings(updated_texts)  # Use existing function
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i, vector=vec.tolist())  # Convert to list
            for i, vec in enumerate(updated_vectors)
        ]
    )

# 4️⃣ Delete Operation (working - kept as-is)
def delete_vectors():
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="id",
                    match=MatchValue(value=5)
                )
            ]
        )
    )

# Run benchmarks
benchmark("Insert Operation", insert_vectors)
benchmark("SEARCH (READ)", search_vectors)
benchmark("Update Operation", update_vectors)
benchmark("Delete Operation", delete_vectors)