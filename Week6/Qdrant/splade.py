import uuid
import time
import psutil
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "splade_benchmark"
NUM_RECORDS = 500  # Reduced for memory constraints
BATCH_SIZE = 16     # Optimized for SPLADE's requirements
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

# Initialize SPLADE components
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Qdrant client
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create collection with SPLADE's vocabulary size
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=30522, distance=Distance.DOT)
)

# Generate sample network data
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, protocols)]

# Enhanced SPLADE embedding generation with validation
def generate_splade_embeddings(texts):
    """Convert text to SPLADE sparse-dense embeddings"""
    if not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("Invalid input texts for embedding generation")
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # SPLADE activation calculation
    logits = outputs.logits
    activations = torch.max(
        torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1),
        dim=1
    ).values
    
    return activations.cpu().numpy()

# Batch processing with validation
vectors = []
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    if not batch:
        print(f"⚠️ Empty batch detected at index {i}")
        continue
        
    try:
        batch_embeddings = generate_splade_embeddings(batch)
        if len(batch_embeddings) != len(batch):
            raise ValueError(f"Generated {len(batch_embeddings)} embeddings for {len(batch)} texts")
            
        vectors.extend([embedding.tolist() for embedding in batch_embeddings])
        print(f"✅ Processed batch {i//BATCH_SIZE + 1}/{(NUM_RECORDS//BATCH_SIZE)+1}")
        
    except Exception as e:
        print(f"🚨 Batch processing failed: {str(e)}")
        print(f"Problematic batch: {batch}")
        exit(1)

def benchmark(operation_name, function):
    """Benchmarking utility with enhanced diagnostics"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = function()
    
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)
    latency = (time.time() - start_time) * 1000
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    print(f"\n {operation_name}")
    print(f"   Latency: {latency:.2f} ms")
    print(f"   CPU Usage: {cpu_after - cpu_before:.2f}%")
    print(f"   Memory Used: {mem_after - mem_before:.2f} MB")
    print(f"   Throughput: {throughput:.2f} ops/sec")
    print("-" * 50)
    return result

# CRUD Operations with validation
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

# 2️⃣ Search Operation
def search_vectors():
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vectors[0],  # Search using the first vector
        limit=1
    )

def update_vectors():
    update_ids = ids[:50]  # Reduced update batch
    new_docs = [f"Updated {doc}" for doc in documents[:50]]
    
    try:
        new_vectors = generate_splade_embeddings(new_docs)
    except Exception as e:
        print(f"🔄 Update failed: {str(e)}")
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
    delete_ids = ids[:50]  # Reduced deletion scope
    client.delete(collection_name=COLLECTION_NAME, points_selector=delete_ids)
    return delete_ids

# Run benchmarks
print("\nStarting SPLADE Benchmark Suite...")
benchmark("INSERT (CREATE)", insert_vectors)
benchmark("SEARCH (READ)", search_vectors)
benchmark("UPDATE", update_vectors)
benchmark("DELETE", delete_vectors)
