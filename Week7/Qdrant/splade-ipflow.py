import uuid
import time
import psutil
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 16     # Optimized for SPLADE's requirements
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
CSV_PATH = "C:/Users/thest\ip_flow/ip_flow_dataset.csv"  # Path to your dataset

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
client = QdrantClient(
    QDRANT_HOST, 
    port=QDRANT_PORT
)

# Load 5-Tuple IP Flow Dataset
def load_ip_flow_data(csv_path, limit=1000):
    """Load IP flow data from CSV and prepare for embedding"""
    try:
        # Load CSV data with proper handling of quote characters
        df = pd.read_csv(csv_path, quotechar='"', escapechar='\\', nrows=limit)
        print(f"✅ Successfully loaded {len(df)} records from CSV")
        
        # Create text representation of each flow for embedding
        documents = []
        for _, row in df.iterrows():
            # Create a string representation of the flow data
            doc = (f"Time: {row['frame.time']}, Source: {row['ip.src']}:{row.get('tcp.srcport', 'N/A')}, "
                  f"Destination: {row['ip.dst']}:{row.get('tcp.dstport', 'N/A')}, "
                  f"Protocol: {row['_ws.col.protocol']}, Length: {row['frame.len']}")
            documents.append(doc)
        
        return df, documents
    except Exception as e:
        print(f"🚨 Error loading CSV data: {str(e)}")
        raise

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

# Batch processing function for embeddings
def process_embeddings(documents):
    vectors = []
    success_count = 0
    
    for i in range(0, len(documents), BATCH_SIZE):
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
            print(f"✅ Successfully processed batch {i//BATCH_SIZE + 1}/{(len(documents)//BATCH_SIZE)+1}")
            
        except Exception as e:
            print(f"🚨 Critical error processing batch {i//BATCH_SIZE + 1}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Message: {str(e)}")
            print("Skipping problematic batch...")
            continue

    if success_count < len(documents):
        print(f"\n⚠️ Warning: Generated {success_count}/{len(documents)} embeddings")
        print("Possible solutions:")
        print("1. Verify input documents contain valid text")
        print("2. Reduce BATCH_SIZE further")
        print("3. Check model compatibility")
        
    return vectors, success_count

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

def run_benchmarks(df, vectors):
    # Generate unique IDs for the records
    ids = [str(uuid.uuid4()) for _ in range(len(df))]
    results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
    
    for config in CONFIGURATIONS:
        collection_name = f"splade_benchmark_{config['name'].lower().replace(' ', '_')}"
        
        # Collection management with validation - with patience
        print(f"\nSetting up collection for {config['name']}...")
        try:
            # Check if collection exists
            if client.collection_exists(collection_name):
                print(f"Collection {collection_name} exists, deleting it...")
                client.delete_collection(collection_name)
                print(f"Collection {collection_name} deleted.")
            
            # Create collection
            print(f"Creating collection {collection_name}...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=config["vector_config"]
            )
            print(f"Collection {collection_name} created successfully.")
        except Exception as e:
            print(f"🚨 Collection setup failed for {config['name']}: {str(e)}")
            continue

        # Define CRUD operations with error handling and patience
        def insert():
            try:
                print(f"Starting insertion of {len(df)} records...")
                
                # Use smaller batches for insertion to prevent timeouts
                insert_batch_size = 50
                all_points = []
                
                for i in range(0, len(df), insert_batch_size):
                    batch_end = min(i + insert_batch_size, len(df))
                    print(f"Preparing batch {i//insert_batch_size + 1}/{(len(df)//insert_batch_size) + 1}...")
                    
                    points = []
                    for j in range(i, batch_end):
                        if j >= len(vectors):  # Safety check
                            continue
                            
                        row = df.iloc[j]
                        payload = {
                            "frame.number": str(row.get("frame.number", "")),
                            "frame.time": str(row.get("frame.time", "")),
                            "ip.src": str(row.get("ip.src", "")),
                            "ip.dst": str(row.get("ip.dst", "")),
                            "tcp.srcport": str(row.get("tcp.srcport", "")),
                            "tcp.dstport": str(row.get("tcp.dstport", "")),
                            "protocol": str(row.get("_ws.col.protocol", "")),
                            "frame.len": str(row.get("frame.len", ""))
                        }
                        
                        point = PointStruct(
                            id=ids[j],
                            vector=vectors[j],
                            payload=payload
                        )
                        points.append(point)
                    
                    print(f"Inserting batch {i//insert_batch_size + 1}...")
                    client.upsert(
                        collection_name=collection_name, 
                        points=points
                    )
                    all_points.extend(points)
                    print(f"Batch {i//insert_batch_size + 1} inserted successfully.")
                
                print(f"All {len(all_points)} records inserted successfully.")
                return all_points
            except Exception as e:
                print(f"⚠️ Insert failed: {str(e)}")
                return []

        def search():
            try:
                print("Performing search operation...")
                # Use query_points instead of deprecated search method
                result = client.query_points(
                    collection_name=collection_name,
                    query_vector=vectors[0],
                    limit=10
                )
                print(f"Search completed, found {len(result)} results.")
                return result
            except Exception as e:
                # Fallback to search if query_points not available in this version
                try:
                    result = client.search(
                        collection_name=collection_name,
                        query_vector=vectors[0],
                        limit=10
                    )
                    print(f"Search completed using deprecated method, found {len(result)} results.")
                    return result
                except Exception as e2:
                    print(f"⚠️ Search failed with both methods: {str(e)} and {str(e2)}")
                    return []

        def update():
            try:
                # Update subset of records (first 50 or less)
                update_count = min(50, len(ids))
                update_ids = ids[:update_count]
                print(f"Updating {update_count} records...")
                
                # Create modified documents for updating
                update_docs = []
                for i in range(update_count):
                    row = df.iloc[i]
                    doc = (f"UPDATED - Time: {row['frame.time']}, Source: {row['ip.src']}:{row.get('tcp.srcport', 'N/A')}, "
                          f"Destination: {row['ip.dst']}:{row.get('tcp.dstport', 'N/A')}, "
                          f"Protocol: {row['_ws.col.protocol']}, Length: {row['frame.len']}")
                    update_docs.append(doc)
                
                # Generate new embeddings for updated documents
                print("Generating embeddings for updated records...")
                new_vectors, _ = process_embeddings(update_docs)
                
                # Create points for update
                points = []
                for i in range(len(update_ids)):
                    if i >= len(new_vectors):  # Safety check
                        continue
                        
                    row = df.iloc[i]
                    payload = {
                        "frame.number": str(row.get("frame.number", "")),
                        "frame.time": str(row.get("frame.time", "")),
                        "ip.src": str(row.get("ip.src", "")),
                        "ip.dst": str(row.get("ip.dst", "")),
                        "tcp.srcport": str(row.get("tcp.srcport", "")),
                        "tcp.dstport": str(row.get("tcp.dstport", "")),
                        "protocol": str(row.get("_ws.col.protocol", "")),
                        "frame.len": str(row.get("frame.len", "")),
                        "updated": True
                    }
                    
                    point = PointStruct(
                        id=update_ids[i],
                        vector=new_vectors[i],
                        payload=payload
                    )
                    points.append(point)
                
                print(f"Updating {len(points)} records...")
                client.upsert(
                    collection_name=collection_name, 
                    points=points
                )
                print("Update completed successfully.")
                return points
            except Exception as e:
                print(f"⚠️ Update failed: {str(e)}")
                return []

        def delete():
            try:
                # Delete subset of records (first 50 or less)
                delete_count = min(50, len(ids))
                delete_ids = ids[:delete_count]
                print(f"Deleting {delete_count} records...")
                client.delete(
                    collection_name=collection_name, 
                    points_selector=delete_ids
                )
                print("Delete completed successfully.")
                return delete_ids
            except Exception as e:
                print(f"⚠️ Delete failed: {str(e)}")
                return []

        # Run benchmarks and handle failures
        try:
            print("\nRunning INSERT benchmark...")
            results["INSERT"].append((config["name"], *benchmark(insert)))
            
            print("\nRunning SEARCH benchmark...")
            results["SEARCH"].append((config["name"], *benchmark(search)))
            
            print("\nRunning UPDATE benchmark...")
            results["UPDATE"].append((config["name"], *benchmark(update)))
            
            print("\nRunning DELETE benchmark...")
            results["DELETE"].append((config["name"], *benchmark(delete)))
        except Exception as e:
            print(f"🚨 Benchmark failed for {config['name']}: {str(e)}")

        # Cleanup collection
        try:
            print(f"\nCleaning up collection {collection_name}...")
            client.delete_collection(collection_name)
            print(f"Collection {collection_name} deleted successfully.")
        except Exception as e:
            print(f"⚠️ Collection cleanup failed: {str(e)}")

    # Format and print results
    headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
    table_data = []
    for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
        row = [op]
        for config in CONFIGURATIONS:
            # Find the result for this config
            result_found = False
            for result in results[op]:
                if result and len(result) >= 1 and result[0] == config["name"]:
                    if len(result) >= 5:  # Make sure we have all metrics
                        metrics_str = (
                            f"Latency: {result[1]:.2f}ms\n"
                            f"CPU: {result[2]:.2f}%\n"
                            f"Memory: {result[3]:.2f}MB\n"
                            f"Throughput: {result[4]:.2f} ops/s"
                        )
                        row.append(metrics_str)
                    else:
                        row.append("Incomplete metrics")
                    result_found = True
                    break
            
            if not result_found:
                row.append("Not completed")
        
        table_data.append(row)
    
    print("\nSPLADE Benchmark Results with 5-Tuple IP Flow Dataset:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

    # Save results to CSV
    result_rows = []
    for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
        for config in CONFIGURATIONS:
            config_name = config["name"]
            result_metrics = next((r for r in results[op] if r[0] == config_name), None)
            if result_metrics and len(result_metrics) == 5:
                _, latency, cpu_usage, mem_usage, throughput = result_metrics
                result_rows.append({
                    "Operation": op,
                    "Configuration": config_name,
                    "Latency (ms)": latency,
                    "CPU Usage (%)": cpu_usage,
                    "Memory Usage (MB)": mem_usage,
                    "Throughput (ops/s)": throughput
                })
            else:
                result_rows.append({
                    "Operation": op,
                    "Configuration": config_name,
                    "Latency (ms)": None,
                    "CPU Usage (%)": None,
                    "Memory Usage (MB)": None,
                    "Throughput (ops/s)": None
                })

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv("splade_results.csv", index=False)
    print("Results saved to splade_results.csv")

if __name__ == "__main__":
    # Load dataset
    try:
        print(f"Loading dataset from {CSV_PATH}...")
        df, documents = load_ip_flow_data(CSV_PATH)
        
        # Process embeddings
        print(f"Generating SPLADE embeddings for {len(documents)} documents...")
        vectors, success_count = process_embeddings(documents)
        
        if success_count > 0:
            # Run benchmarks if we have embeddings
            print(f"Running benchmarks with {success_count} vectors...")
            run_benchmarks(df.iloc[:success_count], vectors)
        else:
            print("🚨 No embeddings generated. Cannot run benchmarks.")
    except Exception as e:
        print(f"🚨 Critical error: {str(e)}")