import uuid
import time
import psutil
import numpy as np
import torch
import pandas as pd
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoModel, AutoTokenizer

# Configuration
QDRANT_HOST = "localhost"  # Changed from http://localhost to just localhost
QDRANT_PORT = 6333
BATCH_SIZE = 32  # Reduced for CPU memory constraints
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DATA_PATH = "ip_flow_dataset.csv"  # Update this with your actual file path
TIMEOUT = 120.0  # Set a longer timeout (in seconds)

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=768, distance=Distance.COSINE)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=768, distance=Distance.COSINE)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)}
]

# Load data from CSV file
def load_data(file_path, max_records=10000):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # Limit the number of records if needed
        if len(df) > max_records:
            df = df.head(max_records)
        
        # Extract required columns
        ids = [str(uuid.uuid4()) for _ in range(len(df))]
        sources = df['ip.src'].tolist()
        destinations = df['ip.dst'].tolist()
        protocols = df['_ws.col.protocol'].tolist()
        ports = [f"{src_port}:{dst_port}" for src_port, dst_port in zip(df['tcp.srcport'], df['tcp.dstport'])]
        frame_lens = df['frame.len'].astype(str).tolist()  # Convert to string to avoid issues
        
        # Create text documents for embedding
        documents = [
            f"Source: {src}, Destination: {dst}, Protocol: {proto}, Ports: {port}, Size: {size}" 
            for src, dst, proto, port, size in zip(sources, destinations, protocols, ports, frame_lens)
        ]
        
        return ids, sources, destinations, protocols, ports, frame_lens, documents
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def generate_embeddings(texts, batch_size=32):
    """Generate embeddings in batches to avoid memory issues"""
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        print(f"Processing embedding batch {i//batch_size + 1}/{total_batches}")
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Calculate sentence embeddings (mean pooling)
            embeddings = (outputs.last_hidden_state * inputs.attention_mask.unsqueeze(-1)).sum(dim=1)
            embeddings = embeddings / inputs.attention_mask.sum(dim=1, keepdim=True)
            all_embeddings.extend(embeddings.numpy())
        except Exception as e:
            print(f"Error generating embeddings for batch: {e}")
            # Add zeros embeddings as fallback to maintain data structure
            empty_embeds = np.zeros((len(batch), 768))
            all_embeddings.extend(empty_embeds)
    
    return all_embeddings

def benchmark(operation):
    """Benchmarking utility with error handling"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    try:
        result = operation()
        if result is None:
            result = []
    except Exception as e:
        print(f"Operation failed: {e}")
        result = []
    
    latency = (time.time() - start_time) * 1000
    cpu_usage = process.cpu_percent(interval=None) - cpu_before
    mem_usage = (process.memory_info().rss / (1024 ** 2)) - mem_before
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    return (latency, cpu_usage, mem_usage, throughput)

def init_qdrant_client():
    """Initialize QdrantClient with proper error handling"""
    try:
        # Try to connect without http prefix first, with increased timeout
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} with timeout {TIMEOUT}s...")
        client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, timeout=TIMEOUT)
        
        # Test connection
        try:
            client.get_collections()
            print("Successfully connected to Qdrant server")
            return client
        except Exception as e:
            print(f"Error testing connection: {e}")
            print("Trying alternative connection method...")
            
            # Try with http prefix
            client = QdrantClient(f"http://{QDRANT_HOST}", port=QDRANT_PORT, timeout=TIMEOUT)
            client.get_collections()
            print("Successfully connected to Qdrant server with http prefix")
            return client
    except Exception as e:
        print(f"Failed to connect to Qdrant server: {e}")
        print("Is your Qdrant server running? Check if it's running with the correct port.")
        print("Make sure you've started the Qdrant server with: docker run -p 6333:6333 qdrant/qdrant")
        raise

def run_benchmarks():
    try:
        # Initialize client with proper error handling
        client = init_qdrant_client()
        
        # Load data from CSV file
        print("Loading data from CSV file...")
        ids, sources, destinations, protocols, ports, frame_lens, documents = load_data(DATA_PATH)
        NUM_RECORDS = len(documents)
        print(f"Loaded {NUM_RECORDS} records")
        
        # Generate embeddings
        print("Generating embeddings with Nomic Embed model...")
        vectors = generate_embeddings(documents, BATCH_SIZE)
        print(f"Generated {len(vectors)} embeddings")
        
        results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
        
        for config in CONFIGURATIONS:
            print(f"\n{'='*40}\nRunning {config['name']} configuration\n{'='*40}")
            collection_name = f"network_benchmark_{config['name'].lower().replace(' ', '_').replace('+', 'plus')}"
            
            # Collection management with error handling
            try:
                if client.collection_exists(collection_name):
                    print(f"Deleting existing collection: {collection_name}")
                    client.delete_collection(collection_name)
                
                print(f"Creating collection: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=config["vector_config"]
                )
                print(f"Collection {collection_name} created successfully")
            except Exception as e:
                print(f"Error managing collection {collection_name}: {e}")
                # Skip this configuration if we can't create the collection
                for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
                    results[op].append((config["name"], 0, 0, 0, 0))
                continue

            # Define operations
            def insert():
                try:
                    # Split into smaller batches to avoid timeouts
                    all_points = [
                        PointStruct(
                            id=ids[i],
                            vector=vec.tolist(),
                            payload={
                                "source": sources[i], 
                                "destination": destinations[i], 
                                "protocol": protocols[i],
                                "port": ports[i],
                                "frame_len": frame_lens[i]
                            }
                        ) for i, vec in enumerate(vectors)
                    ]
                    
                    # Insert in smaller batches to avoid timeouts
                    small_batch_size = 25  # Even smaller for the larger Nomic embed vectors
                    for i in range(0, len(all_points), small_batch_size):
                        batch_points = all_points[i:i+small_batch_size]
                        client.upsert(collection_name=collection_name, points=batch_points)
                        print(f"Inserted batch {i//small_batch_size + 1}/{(len(all_points) + small_batch_size - 1) // small_batch_size}")
                    
                    return all_points
                except Exception as e:
                    print(f"Error during insert: {e}")
                    return []

            def search():
                try:
                    return client.search(
                        collection_name=collection_name,
                        query_vector=vectors[0].tolist(),
                        limit=10
                    )
                except Exception as e:
                    print(f"Error during search: {e}")
                    return []

            def update():
                try:
                    # Generate new embeddings for updated text
                    update_limit = min(50, NUM_RECORDS)  # Limit to 50 for better performance
                    update_texts = [f"Updated {documents[i]}" for i in range(update_limit)]
                    update_vectors = generate_embeddings(update_texts, batch_size=16)  # Smaller batch size for updates
                    
                    points = [
                        PointStruct(
                            id=ids[i],
                            vector=vec.tolist(),
                            payload={
                                "source": "updated", 
                                "destination": "updated", 
                                "protocol": "UPDATED",
                                "port": "0:0",
                                "frame_len": "0"
                            }
                        ) for i, vec in enumerate(update_vectors)
                    ]
                    
                    # Update in smaller batches to avoid timeouts
                    small_batch_size = 25
                    for i in range(0, len(points), small_batch_size):
                        batch_points = points[i:i+small_batch_size]
                        client.upsert(collection_name=collection_name, points=batch_points)
                        print(f"Updated batch {i//small_batch_size + 1}/{(len(points) + small_batch_size - 1) // small_batch_size}")
                    
                    return points
                except Exception as e:
                    print(f"Error during update: {e}")
                    return []

            def delete():
                try:
                    delete_limit = min(50, NUM_RECORDS)
                    delete_ids = ids[:delete_limit]
                    client.delete(collection_name=collection_name, points_selector=delete_ids)
                    return delete_ids
                except Exception as e:
                    print(f"Error during delete: {e}")
                    return []

            # Run benchmarks with error handling
            print("Running INSERT benchmark...")
            insert_result = benchmark(insert)
            results["INSERT"].append((config["name"], *insert_result))
            
            print("Running SEARCH benchmark...")
            search_result = benchmark(search)
            results["SEARCH"].append((config["name"], *search_result))
            
            print("Running UPDATE benchmark...")
            update_result = benchmark(update)
            results["UPDATE"].append((config["name"], *update_result))
            
            print("Running DELETE benchmark...")
            delete_result = benchmark(delete)
            results["DELETE"].append((config["name"], *delete_result))

            try:
                print(f"Cleaning up collection {collection_name}")
                client.delete_collection(collection_name)
            except Exception as e:
                print(f"Error deleting collection {collection_name}: {e}")

        # Format results
        headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
        table_data = []
        for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
            row = [op]
            for config_idx, config in enumerate(CONFIGURATIONS):
                metric = results[op][config_idx]
                row.append(
                    f"Latency: {metric[1]:.2f}ms\n"
                    f"CPU: {metric[2]:.2f}%\n"
                    f"Memory: {metric[3]:.2f}MB\n"
                    f"Throughput: {metric[4]:.2f} ops/s"
                )
            table_data.append(row)
        
        print("\n\nBenchmark Results Summary:")
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

        # Save results to CSV
        result_rows = []
        for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
            for config_idx, config in enumerate(CONFIGURATIONS):
                metric = results[op][config_idx]
                result_rows.append({
                    "Operation": op,
                    "Configuration": config["name"],
                    "Latency (ms)": metric[1],
                    "CPU Usage (%)": metric[2],
                    "Memory Usage (MB)": metric[3],
                    "Throughput (ops/s)": metric[4],
                    "Model": MODEL_NAME
                })
        
        results_df = pd.DataFrame(result_rows)
        results_df.to_csv("nomic_embed_benchmark_results.csv", index=False)
        print("Results saved to nomic_embed_benchmark_results.csv")
    
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")

if __name__ == "__main__":
    try:
        # Initialize tokenizer and model
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("Model loaded successfully")
        
        run_benchmarks()
    except Exception as e:
        print(f"Fatal error: {e}")