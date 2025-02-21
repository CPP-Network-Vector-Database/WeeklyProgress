import time
import psutil
import numpy as np
import faiss
import requests
import matplotlib.pyplot as plt
import streamlit as st
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# Load Pride and Prejudice from Project Gutenberg
@st.cache_data
def load_gutenberg_text():
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    response = requests.get(url)
    return response.text

document_text = load_gutenberg_text()

# Split text into chunks for processing
def split_document(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_document(document_text)

# Convert text chunks to vectors using SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = np.array(embedding_model.encode(chunks), dtype="float32")
dim = vectors.shape[1]  # Dimensionality of embeddings

# Create FAISS indices
@st.cache_resource
def create_faiss_indices():
    indices = {
        "FlatL2": faiss.IndexFlatL2(dim),
        "IVFFlat": faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100),
        "HNSW": faiss.IndexHNSWFlat(dim, 32),
        "IVFPQ": faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, 100, 8, 8)
    }

    if not indices["IVFFlat"].is_trained:
        indices["IVFFlat"].train(vectors)
    if not indices["IVFPQ"].is_trained:
        indices["IVFPQ"].train(vectors)

    for index in indices.values():
        index.add(vectors)

    return indices

faiss_indices = create_faiss_indices()

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
collection_name = "document_collection"

if collection_name in utility.list_collections():
    Collection(collection_name).drop()

schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
], description="FAISS vs. Milvus Benchmark")

collection = Collection(collection_name, schema)
collection.insert([vectors.tolist()])
collection.load()

# Function to benchmark FAISS and Milvus searches
def benchmark_search(index, queries, search_fn):
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used / (1024 ** 2)

    search_fn(index, queries)

    query_time = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used / (1024 ** 2)

    return {
        "Query Time (s)": query_time,
        "CPU Usage (%)": cpu_after - cpu_before,
        "Memory Usage (MB)": mem_after - mem_before
    }

def faiss_search(index, queries):
    index.search(queries, k=5)

def milvus_search(collection, queries):
    search_param = {"metric_type": "L2", "params": {"nprobe": 10}}
    collection.search(queries.tolist(), "vector", search_param, limit=5)

# Streamlit UI
st.title("faiss+milvus vs milvus+nobody")

# Step 1: Select number of queries
num_queries = st.number_input("Number of queries", min_value=1, max_value=10, value=3, step=1)

# Step 2: Enter queries dynamically
queries = []
for i in range(num_queries):
    query = st.text_input(f"Query {i+1}")
    queries.append(query)

# Run search when the user clicks the button
if st.button("Run Benchmark"):
    query_vectors = np.array(embedding_model.encode(queries), dtype="float32")

    faiss_results = {name: {"time": [], "cpu": [], "memory": []} for name in faiss_indices.keys()}
    milvus_results = {"time": [], "cpu": [], "memory": []}

    for _ in range(num_queries):
        for name, index in faiss_indices.items():
            result = benchmark_search(index, query_vectors, faiss_search)
            faiss_results[name]["time"].append(result["Query Time (s)"])
            faiss_results[name]["cpu"].append(result["CPU Usage (%)"])
            faiss_results[name]["memory"].append(result["Memory Usage (MB)"])

        milvus_result = benchmark_search(collection, query_vectors, milvus_search)
        milvus_results["time"].append(milvus_result["Query Time (s)"])
        milvus_results["cpu"].append(milvus_result["CPU Usage (%)"])
        milvus_results["memory"].append(milvus_result["Memory Usage (MB)"])

    # Plot results
    def plot_save(results, milvus_results, metric, ylabel):
        plt.figure(figsize=(10, 5))
        for name, values in results.items():
            plt.plot(range(num_queries), values[metric], label=f"FAISS-{name}", marker='o')

        plt.plot(range(num_queries), milvus_results[metric], label="Milvus", marker='o', linestyle='dashed')

        plt.xlabel("Query Number")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Across Queries")
        plt.legend()
        plt.grid()

        st.pyplot(plt)

    # Display plots
    st.subheader("Query Time Comparison")
    plot_save(faiss_results, milvus_results, "time", "Query Time (s)")

    st.subheader("CPU Usage Comparison")
    plot_save(faiss_results, milvus_results, "cpu", "CPU Usage (%)")

    st.subheader("Memory Usage Comparison")
    plot_save(faiss_results, milvus_results, "memory", "Memory Usage (MB)")


'''
Run this if no work: 
docker-compose down
docker-compose up -d

'''
