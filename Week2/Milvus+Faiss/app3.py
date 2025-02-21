# FAISS vs. Milvus: Benchmarking Search Performance  
# This script loads a classic book, converts it into vector embeddings, and compares retrieval speeds using FAISS and Milvus.

import time
import psutil  # For monitoring CPU and memory usage
import numpy as np
import faiss  # Facebook AI Similarity Search (FAISS) for fast retrieval
import requests  # To fetch the text of Pride and Prejudice
import matplotlib.pyplot as plt
import streamlit as st  # For interactive UI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer  # For generating vector embeddings

# Load "Pride and Prejudice" from Project Gutenberg  
# Caching the text to avoid unnecessary re-downloads
@st.cache_data
def load_gutenberg_text():
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    response = requests.get(url)
    return response.text

document_text = load_gutenberg_text()

# Split the text into chunks for vector embedding  
# FAISS and Milvus both work best with smaller, manageable pieces
def split_document(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_document(document_text)

# Convert text chunks into vector embeddings  
# SentenceTransformer generates numerical representations of text that can be used for similarity searches
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = np.array(embedding_model.encode(chunks), dtype="float32")
dim = vectors.shape[1]  # The dimensionality of the embeddings

# Creating FAISS indices for different retrieval strategies  
@st.cache_resource
def create_faiss_indices():
    indices = {
        "FlatL2": faiss.IndexFlatL2(dim),  # Brute-force search
        "IVFFlat": faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100),  # Inverted file index for efficient searching
        "HNSW": faiss.IndexHNSWFlat(dim, 32),  # Graph-based nearest neighbor search
        "IVFPQ": faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, 100, 8, 8)  # Product quantization for compression
    }

    # Train indices that require it before adding vectors
    if not indices["IVFFlat"].is_trained:
        indices["IVFFlat"].train(vectors)
    if not indices["IVFPQ"].is_trained:
        indices["IVFPQ"].train(vectors)

    # Add vectors to each index
    for index in indices.values():
        index.add(vectors)

    return indices

faiss_indices = create_faiss_indices()

# Connect to Milvus  
# This assumes Milvus is running locally
connections.connect("default", host="localhost", port="19530")
collection_name = "document_collection"

# Check if the collection exists to avoid unnecessary re-creation
if utility.has_collection(collection_name):
    collection = Collection(collection_name)  # Load existing collection
else:
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ], description="FAISS vs. Milvus Benchmark")

    collection = Collection(collection_name, schema)
    collection.insert([vectors.tolist()])
    collection.load()

# Function to benchmark FAISS and Milvus searches  
# Measures time, CPU usage, and memory usage for a given search function
def benchmark_search(index, queries, search_fn):
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used / (1024 ** 2)

    results = search_fn(index, queries)

    query_time = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used / (1024 ** 2)

    return {
        "Query Time (s)": query_time,
        "CPU Usage (%)": cpu_after - cpu_before,
        "Memory Usage (MB)": mem_after - mem_before,
        "Results": results
    }

# FAISS search function  
# Returns the most relevant text chunks for each query
def faiss_search(index, queries):
    distances, indices = index.search(queries, k=5)
    return [[chunks[idx] for idx in query_result] for query_result in indices]

# Milvus search function  
# Retrieves relevant text chunks from the Milvus collection
def milvus_search(collection, queries):
    search_param = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(queries.tolist(), "vector", search_param, limit=5)

    retrieved_texts = []
    for hits in results:
        if not hits:  # If no results are found, return a placeholder
            retrieved_texts.append(["No relevant results found."])
        else:
            retrieved_texts.append([
                chunks[result.id] if 0 <= result.id < len(chunks) else "Invalid index returned."
                for result in hits
            ])
    return retrieved_texts

# Streamlit UI  
# This section creates an interactive dashboard to compare FAISS and Milvus
st.title("FAISS vs. Milvus: Benchmarking Search Performance")

# Select number of queries to test
num_queries = st.number_input("Number of queries", min_value=1, max_value=10, value=3, step=1)

# Allow the user to enter multiple queries dynamically
queries = []
for i in range(num_queries):
    query = st.text_input(f"Query {i+1}")
    queries.append(query)

# Run search when the user clicks the button
if st.button("Run Benchmark"):
    query_vectors = np.array(embedding_model.encode(queries), dtype="float32")

    # Store FAISS and Milvus results separately
    faiss_results = {name: {"time": [], "cpu": [], "memory": [], "results": []} for name in faiss_indices.keys()}
    milvus_results = {"time": [], "cpu": [], "memory": [], "results": []}

    for _ in range(num_queries):
        for name, index in faiss_indices.items():
            result = benchmark_search(index, query_vectors, faiss_search)
            faiss_results[name]["time"].append(result["Query Time (s)"])
            faiss_results[name]["cpu"].append(result["CPU Usage (%)"])
            faiss_results[name]["memory"].append(result["Memory Usage (MB)"])
            faiss_results[name]["results"].append(result["Results"])

        milvus_result = benchmark_search(collection, query_vectors, milvus_search)
        milvus_results["time"].append(milvus_result["Query Time (s)"])
        milvus_results["cpu"].append(milvus_result["CPU Usage (%)"])
        milvus_results["memory"].append(milvus_result["Memory Usage (MB)"])
        milvus_results["results"].append(milvus_result["Results"])

    # Function to plot and display results
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

    # Display query time, CPU usage, and memory usage comparisons
    st.subheader("Query Time Comparison")
    plot_save(faiss_results, milvus_results, "time", "Query Time (s)")

    st.subheader("CPU Usage Comparison")
    plot_save(faiss_results, milvus_results, "cpu", "CPU Usage (%)")

    st.subheader("Memory Usage Comparison")
    plot_save(faiss_results, milvus_results, "memory", "Memory Usage (MB)")

    # Display retrieval results
    for i in range(num_queries):
        st.subheader(f"Query {i+1}: {queries[i]}")

        st.write("### FAISS Results:")
        for name, results in faiss_results.items():
            st.markdown(f"**{name}**:")
            for retrieved_text in results["results"][i]:
                st.write(f"- {retrieved_text}")

        st.write("### Milvus Results:")
        for retrieved_text in milvus_results["results"][i]:
            st.write(f"- {retrieved_text}")
