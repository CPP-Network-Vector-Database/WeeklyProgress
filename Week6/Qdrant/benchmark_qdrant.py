import time
import psutil
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, UpdateResult
from fastembed.embedding import FlagEmbedding
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Define Qdrant settings
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "benchmark_vectors"

# Connect to Qdrant
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create a collection if not exists
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Adjust size as per embeddings
)

# Initialize FastEmbed model
embedder = FlagEmbedding(model_name="BAAI/bge-small-en")  # Change model if needed

# Generate some sample text data
texts = [f"Sample text {i}" for i in range(1000)]
vectors = list(embedder.embed(texts))

# Benchmarking utility function
def benchmark(operation_name, function):
    """Measures CPU, latency, memory, and throughput for an operation."""
    process = psutil.Process()
    
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    result = function()

    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)

    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to ms
    cpu_usage = cpu_after - cpu_before
    memory_used = mem_after - mem_before
    throughput = len(vectors) / (latency / 1000)  # Operations per second

    print(f"🔹 {operation_name}")
    print(f"   🕒 Latency: {latency:.2f} ms")
    print(f"   🔥 CPU Usage: {cpu_usage:.2f}%")
    print(f"   🧠 Memory Used: {memory_used:.2f} MB")
    print(f"   ⚡ Throughput: {throughput:.2f} ops/sec\n")

# 1️⃣ Insert Operation
def insert_vectors():
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i, vector=vec) for i, vec in enumerate(vectors)
        ]
    )

benchmark("Insert Operation", insert_vectors)

# 2️⃣ Search Operation
def search_vectors():
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vectors[0],  # Search using the first vector
        limit=1
    )

benchmark("Search Operation", search_vectors)

# 3️⃣ Update Operation
def update_vectors():
    updated_vectors = list(embedder.embed([f"Updated text {i}" for i in range(100)]))  # New embeddings
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i, vector=updated_vectors[i]) for i in range(100)
        ]
    )

benchmark("Update Operation", update_vectors)

# 4️⃣ Delete Operation
def delete_vectors():
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="id",
                    match=MatchValue(value=5)  # Adjust as needed
                )
            ]
        )
    )

benchmark("Delete Operation", delete_vectors)