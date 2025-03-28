import time
import psutil
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance

# Initialize Qdrant client with timeout settings
client = QdrantClient(
    "localhost",
    port=6333,
    timeout=30,  # Increase timeout to 30 seconds
    prefer_grpc=True  # Try gRPC protocol instead of REST
)

# Verify connection
try:
    client.get_collections()
except Exception as e:
    print(f"❌ Failed to connect to Qdrant: {e}")
    print("1. Make sure Qdrant is running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
    print("2. Check firewall settings allowing port 6333")
    exit(1)

# Load model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def create_collection():
    try:
        if not client.collection_exists("my_collection"):
            client.create_collection(
                collection_name="my_collection",
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                timeout=60  # Give more time for collection creation
            )
    except Exception as e:
        print(f"🚨 Collection creation failed: {e}")
        exit(1)

def generate_embedding(text):
    return model.encode(text, normalize_embeddings=True).tolist()

def measure_resources(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_cpu = process.cpu_percent()
        start_mem = process.memory_info().rss

        result = func(*args, **kwargs)

        end_time = time.time()
        end_cpu = process.cpu_percent()
        end_mem = process.memory_info().rss

        latency = (end_time - start_time) * 1000
        cpu_usage = end_cpu - start_cpu
        mem_used = (end_mem - start_mem) / (1024 * 1024)  # Convert to MB
        throughput = len(args[0])/(latency/1000) if args and len(args[0]) > 0 else 0

        print(f"🔹 {func.__name__.replace('_', ' ').title()}")
        print(f"    🕒 Latency: {latency:.2f} ms")
        print(f"    🔥 CPU Usage: {cpu_usage:.2f}%")
        print(f"    🧠 Memory Used: {mem_used:.2f} MB")
        print(f"    ⚡ Throughput: {throughput:.2f} ops/sec\n")
        return result
    return wrapper

@measure_resources
def insert_vectors():
    points = [
        PointStruct(
            id=i,
            vector=generate_embedding(f"Example text {i}"),
            payload={"text": f"Example text {i}"}
        ) for i in range(1000)
    ]
    client.upsert(collection_name="my_collection", points=points)

@measure_resources
def search_vectors():
    query_vector = generate_embedding("Example query")
    return client.search(
        collection_name="my_collection",
        query_vector=query_vector,
        limit=5
    )

@measure_resources
def update_vectors():
    new_vector = generate_embedding("Updated text")
    client.upsert(
        collection_name="my_collection",
        points=[PointStruct(id=0, vector=new_vector, payload={"text": "Updated text"})]
    )

@measure_resources
def delete_vectors():
    client.delete(
        collection_name="my_collection",
        points_selector=Filter(
            must=[FieldCondition(key="text", match=MatchValue(value="Updated text"))]
        )
    )

if __name__ == "__main__":
    create_collection()  # Create collection first
    insert_vectors()
    search_vectors()
    update_vectors()
    delete_vectors()