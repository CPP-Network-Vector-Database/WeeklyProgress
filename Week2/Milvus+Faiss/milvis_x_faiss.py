import time
import numpy as np
import pandas as pd
import faiss
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Setting up some basics
d = 128   # Vector dimensionality
n = 10000  # Number of vectors
np.random.seed(42)  # Keeping things reproducible

# Generating random vectors for indexing and querying
data = np.random.random((n, d)).astype('float32')
queries = np.random.random((10, d)).astype('float32')

# Connecting to Milvus
connections.connect("default", host="localhost", port="19530")

# Dropping existing collection if it exists
collection_name = "vector_collection"
if collection_name in utility.list_collections():
    collection = Collection(collection_name)
    collection.drop()

# Defining the schema: 
# - `id`: auto-incrementing primary key
# - `vector`: the actual embedding
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=d)
]
schema = CollectionSchema(fields, description="FAISS vs. Milvus Comparison")
collection = Collection(collection_name, schema)

# Function to benchmark FAISS index performance
def benchmark_faiss(index, name):
    # Indexing
    index_time_start = time.time()
    index.add(data)
    index_time = time.time() - index_time_start

    # Searching
    search_time_start = time.time()
    _, _ = index.search(queries, 10)
    search_time = time.time() - search_time_start

    return {"Index": name, "Index Time (s)": index_time, "Search Time (s)": search_time}

# Different FAISS indexing strategies
index_flat = faiss.IndexFlatL2(d)  # Exhaustive search
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)  # Clustering-based search
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # Graph-based search

# IVF needs training before use
if not index_ivf.is_trained:
    index_ivf.train(data)

# Running FAISS benchmarks
results = []
results.append(benchmark_faiss(index_flat, "Flat"))  # Simple but slow
results.append(benchmark_faiss(index_ivf, "IVF"))  # Faster, needs training
results.append(benchmark_faiss(index_hnsw, "HNSW"))  # Uses graphs, usually fast

# Inserting vectors into Milvus
insert_time_start = time.time()
collection.insert([data.tolist()])  # Only inserting vectors, ID is auto-generated
insert_time = time.time() - insert_time_start

# Milvus needs to load before searching
collection.load()

# Running search in Milvus
search_time_start = time.time()
search_param = {"metric_type": "L2", "params": {"nprobe": 10}}
results_milvus = collection.search(queries.tolist(), "vector", search_param, limit=10)
search_time = time.time() - search_time_start

# Adding Milvus results
results.append({"Index": "Milvus", "Index Time (s)": insert_time, "Search Time (s)": search_time})

# Convert results to DataFrame and print
results_df = pd.DataFrame(results)
print(results_df)

'''
Output:

    Index  Index Time (s)  Search Time (s)                                                                                                                                                  .----.   @   @
0    Flat        0.002349         0.000666                                                                                                                                                 / .-"-.`.  \v/
1     IVF        0.013413         0.003772                                                                                                                                                 | | '\ \ \_/ )
2    HNSW        0.480003         0.000618                                                                                                                                               ,-\ `-.' /.'  /
3  Milvus        0.273377         0.058644   
'''