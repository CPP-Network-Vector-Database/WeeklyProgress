import time
import numpy as np
import pandas as pd
import faiss
import random
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Number of flows
n = 10000  
num_queries = 10  

# Function to generate random IP addresses
def random_ip():
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

# Function to generate five-tuple flows (IP-based)
def generate_flows(num):
    flows = []
    for _ in range(num):
        # Convert IP addresses into integers by removing dots (naïve approach)
        flow = [
            int(random_ip().replace(".", "")),  # Source IP transformed into an integer
            int(random_ip().replace(".", "")),  # Destination IP transformed into an integer
            random.randint(1024, 65535),  # Source port (1024-65535)
            random.randint(1024, 65535),  # Destination port (1024-65535)
            random.randint(1, 255)  # Protocol number (e.g., TCP = 6, UDP = 17)
        ]
        flows.append(flow)
    return np.array(flows, dtype="float32")

# Generate dataset: each flow is now a 5-dimensional vector
data = generate_flows(n)
queries = generate_flows(num_queries)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define collection name
collection_name = "ip_flow_collection"

# Drop collection if it already exists to start fresh
if collection_name in utility.list_collections():
    collection = Collection(collection_name)
    collection.drop()

# Define schema: each five-tuple flow is stored as a 5D vector
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-generating unique IDs
    FieldSchema(name="flow_vector", dtype=DataType.FLOAT_VECTOR, dim=5)  # Store the 5-tuple flow as a vector
]
schema = CollectionSchema(fields, description="FAISS vs. Milvus for IP Flow Matching")
collection = Collection(collection_name, schema)

# Function to benchmark FAISS performance
def benchmark_faiss(index, name):
    index_time_start = time.time()
    index.add(data)  # Adding our generated flows to FAISS
    index_time = time.time() - index_time_start

    search_time_start = time.time()
    _, _ = index.search(queries, 10)  # Searching for nearest neighbors
    search_time = time.time() - search_time_start

    return {"Index": name, "Index Time (s)": index_time, "Search Time (s)": search_time}

# FAISS Indexing Methods
index_flat = faiss.IndexFlatL2(5)  # Exhaustive search (brute force)
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(5), 5, 100)  # Inverted File Index for speed
index_hnsw = faiss.IndexHNSWFlat(5, 32)  # Hierarchical Navigable Small World Graph for fast search

# Train IVF index if needed
if not index_ivf.is_trained:
    index_ivf.train(data)  # Training IVF index with our dataset

# Benchmark FAISS
results = []
results.append(benchmark_faiss(index_flat, "Flat"))  # Brute-force
results.append(benchmark_faiss(index_ivf, "IVF"))  # Clustering-based
results.append(benchmark_faiss(index_hnsw, "HNSW"))  # Graph-based

# Insert data into Milvus for large-scale search
insert_time_start = time.time()
collection.insert([data.tolist()])  # Store the IP flow vectors in Milvus
insert_time = time.time() - insert_time_start

# Load collection before searching
collection.load()

# Search in Milvus
search_time_start = time.time()
search_param = {"metric_type": "L2", "params": {"nprobe": 10}}  # L2 distance metric
results_milvus = collection.search(queries.tolist(), "flow_vector", search_param, limit=10)
search_time = time.time() - search_time_start

# Add Milvus results to results list
results.append({"Index": "Milvus", "Index Time (s)": insert_time, "Search Time (s)": search_time})

results_df = pd.DataFrame(results)
print(results_df)

'''
- FAISS Flat: brute force, slow but accurate
- FAISS IVF: speeds up with clustering
- FAISS HNSW: faster graph-based search
- Milvus: optimized for large-scale data, some overhead

    Index  Index Time (s)  Search Time (s)                                                                                                                                                  .----.   @   @
0    Flat        0.000126         0.006166                                                                                                                                                 / .-"-.`.  \v/
1     IVF        0.014755         0.007917                                                                                                                                                 | | '\ \ \_/ )
2    HNSW        0.048008         0.008994                                                                                                                                               ,-\ `-.' /.'  /
3  Milvus        0.075657         0.009289                                                                                                                                              '---`----'----'
'''
