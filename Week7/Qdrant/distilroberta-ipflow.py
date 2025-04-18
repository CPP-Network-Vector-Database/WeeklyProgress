import time
import psutil
import numpy as np
import pandas as pd
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_HOST = "localhost"  # Host without protocol to avoid errors
QDRANT_PORT = 6333         # Default Qdrant port
NUM_RECORDS = 10000         # Number of records to process
BATCH_SIZE = 100           # Batch size for embedding generation
DATASET_PATH = "ip_flow_dataset.csv"  # Update this to your CSV file path

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=768, distance=Distance.COSINE)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=768, distance=Distance.COSINE)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)}
]

# Initialize components
embedder = SentenceTransformer("all-distilroberta-v1", device="cpu")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)  # Initialize Qdrant client correctly

# Load dataset and prepare data
df = pd.read_csv(DATASET_PATH).head(NUM_RECORDS)
ids = df["frame.number"].astype(int).tolist()  # Extract integer IDs from frame.number
documents = (
    "Source IP: " + df["ip.src"].astype(str) + ", Source Port: " + df["tcp.srcport"].astype(str) +
    ", Destination IP: " + df["ip.dst"].astype(str) + ", Destination Port: " + df["tcp.dstport"].astype(str) +
    ", Protocol: " + df["_ws.col.protocol"].astype(str)
).tolist()

# Generate embeddings in batches
vectors = []
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    vectors.extend(embedder.encode(batch).tolist())

def benchmark(function):
    """Utility function to measure latency, CPU, memory, and throughput"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = function()
    
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    cpu_usage = process.cpu_percent(interval=None) - cpu_before
    mem_usage = (process.memory_info().rss / (1024 ** 2)) - mem_before
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    return (latency, cpu_usage, mem_usage, throughput)

def run_benchmarks():
    """Run benchmarks for insert, search, update, and delete operations"""
    results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
    
    for config in CONFIGURATIONS:
        collection_name = f"network_benchmark_{config['name'].lower().replace(' ', '_')}"
        
        # Recreate collection for each configuration
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=config["vector_config"]
        )
        
        # Insert Operation
        def insert_op():
            payloads = df.to_dict(orient="records")
            points = [
                PointStruct(
                    id=ids[i],  # Use integer IDs
                    vector=vectors[i],
                    payload=payloads[i]
                ) for i in range(NUM_RECORDS)
            ]
            client.upsert(collection_name=collection_name, points=points)
            return points
        
        # Search Operation
        def search_op():
            return client.search(
                collection_name=collection_name,
                query_vector=vectors[0],
                limit=1
            )
        
        # Update Operation
        def update_op():
            update_indices = list(range(100))  # Update first 100 records
            updated_documents = [f"Updated: {documents[i]}" for i in update_indices]
            updated_vectors = embedder.encode(updated_documents).tolist()
            updated_payloads = [dict(df.iloc[i].to_dict(), updated=True) for i in update_indices]
            points = [
                PointStruct(
                    id=ids[i],  # Use integer IDs
                    vector=updated_vectors[idx],
                    payload=updated_payloads[idx]
                ) for idx, i in enumerate(update_indices)
            ]
            client.upsert(collection_name=collection_name, points=points)
            return points
        
        # Delete Operation
        def delete_op():
            delete_ids = ids[:100]  # Delete first 100 records
            client.delete(collection_name=collection_name, points_selector=delete_ids)
            return delete_ids
        
        # Run benchmarks for current configuration
        insert_metrics = benchmark(insert_op)
        search_metrics = benchmark(search_op)
        update_metrics = benchmark(update_op)
        delete_metrics = benchmark(delete_op)
        
        # Store results
        results["INSERT"].append((config["name"], *insert_metrics))
        results["SEARCH"].append((config["name"], *search_metrics))
        results["UPDATE"].append((config["name"], *update_metrics))
        results["DELETE"].append((config["name"], *delete_metrics))
        
        # Clean up collection
        client.delete_collection(collection_name)
    
    # Format and print results
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

    # Save results to CSV
    result_rows = []
    for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
        for metric in results[op]:
            result_rows.append({
                "Operation": op,
                "Configuration": metric[0],
                "Latency (ms)": metric[1],
                "CPU Usage (%)": metric[2],
                "Memory Usage (MB)": metric[3],
                "Throughput (ops/s)": metric[4]
            })

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv("distilroberta_results.csv", index=False)
    print("Results saved to distilroberta_results.csv")

if __name__ == "__main__":
    run_benchmarks()