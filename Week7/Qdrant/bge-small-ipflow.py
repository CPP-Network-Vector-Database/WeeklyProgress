import uuid
import time
import psutil
import numpy as np
import pandas as pd
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding

# Configuration
QDRANT_HOST = "localhost"  # Changed from http://localhost to just localhost
QDRANT_PORT = 6333
BATCH_SIZE = 100
MODEL_NAME = "BAAI/bge-small-en"
DATA_PATH = "ip_flow_dataset.csv"
TIMEOUT = 120.0  # Set a longer timeout (in seconds)

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=384, distance=Distance.COSINE)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=384, distance=Distance.COSINE, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=384, distance=Distance.COSINE)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=384, distance=Distance.COSINE, on_disk=True)}
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

def benchmark(operation):
    """Benchmarking utility without prints"""
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
        print("Generating embeddings...")
        vectors = []
        for i, batch in enumerate([documents[i:i+BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]):
            print(f"Processing batch {i+1}/{(len(documents) + BATCH_SIZE - 1) // BATCH_SIZE}")
            vectors.extend(list(embedder.embed(batch)))
        print("Embeddings generated")
        
        results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
        
        for config in CONFIGURATIONS:
            print(f"Running benchmark for {config['name']}...")
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
                    small_batch_size = 50
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
                    update_ids = ids[:min(100, NUM_RECORDS)]
                    update_idx = range(min(100, NUM_RECORDS))
                    points = [
                        PointStruct(
                            id=ids[i],
                            vector=vectors[i].tolist(),
                            payload={
                                "source": "updated", 
                                "destination": "updated", 
                                "protocol": "UPDATED",
                                "port": "0:0",
                                "frame_len": "0"
                            }
                        ) for i in update_idx
                    ]
                    
                    # Update in smaller batches to avoid timeouts
                    small_batch_size = 50
                    for i in range(0, len(points), small_batch_size):
                        batch_points = points[i:i+small_batch_size]
                        client.upsert(collection_name=collection_name, points=batch_points)
                    
                    return points
                except Exception as e:
                    print(f"Error during update: {e}")
                    return []

            def delete():
                try:
                    delete_ids = ids[:min(100, NUM_RECORDS)]
                    client.delete(collection_name=collection_name, points_selector=delete_ids)
                    return delete_ids
                except Exception as e:
                    print(f"Error during delete: {e}")
                    return []

            # Run benchmarks with error handling
            print("Running INSERT benchmark...")
            results["INSERT"].append((config["name"], *benchmark(insert)))
            
            print("Running SEARCH benchmark...")
            results["SEARCH"].append((config["name"], *benchmark(search)))
            
            print("Running UPDATE benchmark...")
            results["UPDATE"].append((config["name"], *benchmark(update)))
            
            print("Running DELETE benchmark...")
            results["DELETE"].append((config["name"], *benchmark(delete)))

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
        
        print("\nBenchmark Results:")
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
                    "Throughput (ops/s)": metric[4]
                })
        
        results_df = pd.DataFrame(result_rows)
        results_df.to_csv("bge-small_results.csv", index=False)
        print("Results saved to bge-small_results.csv")
    
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")

if __name__ == "__main__":
    try:
        # Initialize text embedding model
        print(f"Initializing text embedding model: {MODEL_NAME}")
        embedder = TextEmbedding(model_name=MODEL_NAME)
        
        run_benchmarks()
    except Exception as e:
        print(f"Fatal error: {e}")
