from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import time
import psutil

# Define the models
models_list = [
    {"name": "all-MiniLM-L6-v2", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "gte-small", "model": "thenlper/gte-small"},
    {"name": "bge-small-en", "model": "BAAI/bge-small-en"},
    {"name": "multilingual-e5-small", "model": "intfloat/multilingual-e5-small"},
    {"name": "nomic-embed-text-v1", "model": "nomic-ai/nomic-embed-text-v1"},
]

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Sample data
documents = [
    {"id": 1, "text": "This is the first document."},
    {"id": 2, "text": "This document is the second document."},
    {"id": 3, "text": "And this is the third one."},
    {"id": 4, "text": "Is this the first document?"},
]

# Function to measure CPU, memory, and latency
def measure_performance():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().used / (1024 ** 2)  # in MB
    return cpu_usage, memory_usage

# Iterate through each model
for model_info in models_list:
    print(f"\nUsing model: {model_info['name']}")
    
    # Load the model
    model = SentenceTransformer(model_info["model"])
    
    # Create a collection in Qdrant
    collection_name = f"collection_{model_info['name']}"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    
    # Generate embeddings and upload to Qdrant
    start_time = time.time()
    embeddings = model.encode([doc["text"] for doc in documents])
    
    # Measure CPU and memory usage
    cpu_usage, memory_usage = measure_performance()
    
    # Insert data into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=[doc["id"] for doc in documents],
            vectors=embeddings.tolist(),
            payloads=[{"text": doc["text"]} for doc in documents]
        )
    )
    
    # Measure latency
    latency = time.time() - start_time
    
    print(f"Model: {model_info['name']}")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"Latency: {latency:.2f} seconds")

    # Search for similar vectors
query = "This is a query document."
query_embedding = model.encode(query).tolist()

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=2  # Top 2 results
)
print("Search Results:", search_result)

# Update a specific point
client.set_payload(
    collection_name=collection_name,
    payload={"text": "Updated document text"},
    points=[1]  # Update document with ID 1
)

# Delete a specific point
client.delete(
    collection_name=collection_name,
    points=[1]  # Delete document with ID 1
)
