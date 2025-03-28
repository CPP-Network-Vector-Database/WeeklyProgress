import uuid
import time
import psutil
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Configuration for Trial Keys
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "cohere_trial_benchmark"
NUM_RECORDS = 100  # Reduced for trial limits
BATCH_SIZE = 8  # Conservative for rate limits
COHERE_API_KEY = "qlvdXdFj0TsRvtRczx1kJPBqPJRyYL0YhTfAhfpS"  # From https://dashboard.cohere.com/
MODEL_NAME = "embed-english-light-v3.0"  # Trial-compatible model
VECTOR_SIZE = 384  # Dimension for light model

# Initialize Cohere with validation
try:
    co = cohere.Client(COHERE_API_KEY)
    # Test API connectivity
    test_response = co.embed(
        texts=["API test"],
        model=MODEL_NAME,
        input_type="search_document",
        truncate="END"
    )
    if not test_response.embeddings:
        raise ValueError("API test failed - no embeddings received")
    print("✅ Cohere API connection successful")
except Exception as e:
    print(f"❌ Cohere API Error: {str(e)}")
    print("1. Verify key at https://dashboard.cohere.com/")
    print("2. Ensure model is accessible with your plan")
    exit(1)

# Initialize Qdrant
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create collection with trial dimensions
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# Generate sample network data
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, protocols)]

# Embedding generation with rate limit handling
vectors = []
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    if not batch:
        continue
        
    try:
        response = co.embed(
            texts=batch,
            model=MODEL_NAME,
            input_type="search_document",
            truncate="END"
        )
        if len(response.embeddings) != len(batch):
            raise ValueError(f"Received {len(response.embeddings)} embeddings for {len(batch)} texts")
            
        vectors.extend(response.embeddings)
        print(f"✅ Processed batch {i//BATCH_SIZE + 1}/{(NUM_RECORDS//BATCH_SIZE)+1}")
        time.sleep(1)  # Rate limit buffer
        
    except Exception as e:
        print(f"🚨 Batch {i//BATCH_SIZE + 1} failed: {str(e)}")
        print("Possible solutions:")
        print("- Check remaining API credits")
        print("- Reduce BATCH_SIZE further")
        print("- Upgrade plan at https://dashboard.cohere.com/")
        exit(1)

def benchmark(operation_name, function):
    """Benchmarking utility"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = function()
    
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)
    latency = (time.time() - start_time) * 1000
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    print(f"\n🔹 {operation_name}")
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
            vector=vectors[i],
            payload={"source": s, "destination": d, "protocol": p}
        ) for i, (s, d, p) in enumerate(zip(sources, destinations, protocols))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return points

def search_vectors():
    results = []
    for vec in vectors[:5]:  # Search first 5 to limit calls
        results.extend(client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=3
        ))
        time.sleep(0.5)  # Rate limit buffer
    return results

def update_vectors():
    update_ids = ids[:10]  # Limited for trial
    new_docs = [f"Updated {doc}" for doc in documents[:10]]
    
    try:
        response = co.embed(
            texts=new_docs,
            model=MODEL_NAME,
            input_type="search_document",
            truncate="END"
        )
        new_vectors = response.embeddings
    except Exception as e:
        print(f"Update failed: {str(e)}")
        return []

    points = [
        PointStruct(
            id=uid,
            vector=vec,
            payload={"source": "updated", "destination": "updated", "protocol": "UPDATED"}
        ) for uid, vec in zip(update_ids, new_vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return points

def delete_vectors():
    delete_ids = ids[:10]  # Limited scope
    client.delete(collection_name=COLLECTION_NAME, points_selector=delete_ids)
    return delete_ids

# Run benchmarks
print("\nStarting Trial Key Benchmarks...")
benchmark("INSERT (CREATE)", insert_vectors)
benchmark("SEARCH (READ)", search_vectors)
benchmark("UPDATE", update_vectors)
benchmark("DELETE", delete_vectors)