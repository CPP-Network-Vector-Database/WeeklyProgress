import uuid
import time
import psutil
import torch
import numpy as np
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
NUM_RECORDS = 500  # Reduced for memory constraints
BATCH_SIZE = 16     # Optimized for SPLADE's requirements
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=30522, distance=Distance.DOT)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=30522, distance=Distance.DOT, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=30522, distance=Distance.DOT)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=30522, distance=Distance.DOT, on_disk=True)}
]

# Initialize SPLADE components
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Qdrant client
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Generate sample network data with validation
def generate_network_data():
    sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
    destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
    protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
    
    documents = []
    for s, d in zip(sources, destinations):
        protocol = np.random.choice(protocols)
        doc = f"{s}, {d}, {protocol}"
        if not doc.strip():
            doc = "Default network entry"  # Fallback for empty docs
        documents.append(doc)
    
    return sources, destinations, protocols, documents

sources, destinations, protocols, documents = generate_network_data()
ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]

# Enhanced SPLADE embedding generation with robust validation
def generate_splade_embeddings(texts):
    """Convert text to SPLADE sparse-dense embeddings with validation"""
    if not isinstance(texts, list) or not texts:
        raise ValueError("Input must be a non-empty list of strings")
    
    # Clean and validate input texts
    valid_texts = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Invalid text at index {idx}: {text}")
            valid_texts.append("Default network log entry")
        else:
            valid_texts.append(text.strip())
    
    try:
        inputs = tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        raise ValueError(f"Tokenizer failed: {str(e)}") from e
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except RuntimeError as e:
        raise RuntimeError(f"Model inference failed: {str(e)}") from e

    # SPLADE activation calculation with dimension validation
    logits = outputs.logits
    if logits.dim() != 3:
        raise ValueError(f"Unexpected logits dimension: {logits.dim()}")
    
    try:
        activations = torch.max(
            torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1),
            dim=1
        ).values
    except RuntimeError as e:
        raise RuntimeError(f"Activation calculation failed: {str(e)}") from e

    return activations.cpu().numpy()

# Batch processing with enhanced error handling
vectors = []
success_count = 0
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    if not batch:
        print(f"⚠️ Empty batch detected at index {i}, skipping")
        continue
        
    try:
        # Check for duplicate texts that might cause issues
        if len(batch) != len(set(batch)):
            print(f"⚠️ Duplicate texts detected in batch {i//BATCH_SIZE + 1}")
            
        batch_embeddings = generate_splade_embeddings(batch)
        
        # Validate output dimensions
        if batch_embeddings.shape[1] != 30522:
            raise ValueError(f"Invalid embedding dimension: {batch_embeddings.shape}")
            
        vectors.extend([embedding.tolist() for embedding in batch_embeddings])
        success_count += len(batch)
        print(f"✅ Successfully processed batch {i//BATCH_SIZE + 1}/{(NUM_RECORDS//BATCH_SIZE)+1}")
        
    except Exception as e:
        print(f"🚨 Critical error processing batch {i//BATCH_SIZE + 1}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"Batch contents: {batch}")
        print("Skipping problematic batch...")
        continue

if success_count < NUM_RECORDS:
    print(f"\n⚠️ Warning: Generated {success_count}/{NUM_RECORDS} embeddings")
    print("Possible solutions:")
    print("1. Verify input documents contain valid text")
    print("2. Reduce BATCH_SIZE further")
    print("3. Check model compatibility")

def benchmark(operation):
    """Benchmarking utility with resource monitoring"""
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
        collection_name = f"splade_benchmark_{config['name'].lower().replace(' ', '_')}"
        
        # Collection management with validation
        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=config["vector_config"]
            )
        except Exception as e:
            print(f"🚨 Collection setup failed for {config['name']}: {str(e)}")
            continue

        # Define CRUD operations with error handling
        def insert():
            try:
                points = [
                    PointStruct(
                        id=ids[i],
                        vector=vectors[i],
                        payload={"source": s, "destination": d, "protocol": p}
                    ) for i, (s, d, p) in enumerate(zip(sources, destinations, protocols))
                ]
                client.upsert(collection_name=collection_name, points=points)
                return points
            except Exception as e:
                print(f"⚠️ Insert failed: {str(e)}")
                return []

        def search():
            try:
                return client.search(
                    collection_name=collection_name,
                    query_vector=vectors[0],
                    limit=1
                )
            except Exception as e:
                print(f"⚠️ Search failed: {str(e)}")
                return []

        def update():
            try:
                update_ids = ids[:50]
                new_docs = [f"Updated {doc}" for doc in documents[:50]]
                new_vectors = generate_splade_embeddings(new_docs)
                points = [
                    PointStruct(
                        id=uid,
                        vector=vec.tolist(),
                        payload={"source": "updated", "destination": "updated", "protocol": "UPDATED"}
                    ) for uid, vec in zip(update_ids, new_vectors)
                ]
                client.upsert(collection_name=collection_name, points=points)
                return points
            except Exception as e:
                print(f"⚠️ Update failed: {str(e)}")
                return []

        def delete():
            try:
                delete_ids = ids[:50]
                client.delete(collection_name=collection_name, points_selector=delete_ids)
                return delete_ids
            except Exception as e:
                print(f"⚠️ Delete failed: {str(e)}")
                return []

        # Run benchmarks and handle failures
        try:
            results["INSERT"].append((config["name"], *benchmark(insert)))
            results["SEARCH"].append((config["name"], *benchmark(search)))
            results["UPDATE"].append((config["name"], *benchmark(update)))
            results["DELETE"].append((config["name"], *benchmark(delete)))
        except Exception as e:
            print(f"🚨 Benchmark failed for {config['name']}: {str(e)}")

        # Cleanup collection
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            print(f"⚠️ Collection cleanup failed: {str(e)}")

    # Format and print results
    headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
    table_data = []
    for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
        row = [op]
        for metric in results[op]:
            metrics_str = (
                f"Latency: {metric[1]:.2f}ms\n"
                f"CPU: {metric[2]:.2f}%\n"
                f"Memory: {metric[3]:.2f}MB\n"
                f"Throughput: {metric[4]:.2f} ops/s"
            )
            row.append(metrics_str)
        table_data.append(row)
    
    print("\nSPLADE Benchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    run_benchmarks()