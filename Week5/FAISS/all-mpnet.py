import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
from sentence_transformers import SentenceTransformer

def measure(pid, func, *args, **kwargs):
    process = psutil.Process(pid)
    # Initial resource usage
    start_cpu = process.cpu_percent(interval=None)
    start_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    # Final resource usage
    end_cpu = process.cpu_percent(interval=None)
    end_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    end_time = time.time()
    
    # Compute diff
    cpu_usage = end_cpu - start_cpu
    mem_usage = end_mem - start_mem
    execution_time = end_time - start_time
    
    print(f"Function: {func.__name__} | Time: {execution_time:.4f}s | CPU: {cpu_usage:.2f}% | Mem: {mem_usage:.2f}MB")
    return result

model = SentenceTransformer("all-mpnet-base-v2")

csv_path = "/kaggle/input/pcap-2019-dira-125910/dirA.125910-packets.csv"
df = pd.read_csv(csv_path, header=None, names=["timestamp", "src_ip", "dst_ip", "protocol", "size"])

# merge all the elements in a row to one "sentence" so that bert can take it
df["packet_text"] = df["src_ip"] + " " + df["dst_ip"] + " " + df["protocol"] + " " + df["size"].astype(str)

embeddings = model.encode(df["packet_text"].tolist(), convert_to_numpy=True)

# Normalize embeddings 
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Creating FAISS IVFFlat index
dimension = embeddings.shape[1]
nlist = min(100, int(np.sqrt(len(embeddings))))  # Number of clusters- 100 is just an arbitrary number I chose

# IVFFlat index with L2 distance (basically euclidean-> sqrt (x^2 + y^2)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
# Train the index (ivfflatl2 needs training)
index.train(embeddings)
index.add(embeddings) # add embeddings to the index

print(f"FAISS index contains {index.ntotal} embeddings.")
faiss.write_index(index, "packet_embeddings.index")
df.to_csv("packet_metadata.csv", index=False)
print("FAISS index and metadata saved.")

# performance measurements
def query_faiss(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices

def insert_data(index, new_embeddings):
    index.add(new_embeddings)
    print(f"Inserted {len(new_embeddings)} new vectors into FAISS index.")

def delete_data(index, delete_indices, total_embeddings):
    # Ensure delete_indices are valid first
    delete_indices = np.array(delete_indices)
    delete_indices = delete_indices[delete_indices < total_embeddings]
    
    # Remove ids from index
    index.remove_ids(delete_indices)
    print(f"Deleted {len(delete_indices)} embeddings from FAISS index.")
    return index

def update_faiss(index, update_indices, new_packet_texts, total_embeddings):
    index = delete_data(index, update_indices, total_embeddings)
    # compute and normalize new embeddings
    new_embeddings = model.encode(new_packet_texts, convert_to_numpy=True)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    # Add new embeddings
    index.add(new_embeddings)
    
    print(f"Updated {len(update_indices)} embeddings in FAISS index.")
    return index

# try it out with 250 flows now: 

# Query FAISS index with a random packet first
query_text = ["192.168.1.1 192.168.1.2 TCP 100"]
query_embedding = model.encode(query_text, convert_to_numpy=True)
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

pid = os.getpid()
query_result = measure(pid, query_faiss, index, query_embedding, 5)
print(f"Top 5 similar packet indices: {query_result}")

# Simulate inserting 250 new packets (same way as before)
new_packet_texts = [f"192.168.1.{i} 192.168.1.{i+1} TCP {i*10}" for i in range(250)]
new_embeddings = model.encode(new_packet_texts, convert_to_numpy=True)
new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

pid = os.getpid()
measure(pid, insert_data, index, new_embeddings)

#deleting 250 random packets
delete_indices = np.random.choice(index.ntotal, 250, replace=False)
pid = os.getpid()
index = measure(pid, delete_data, index, delete_indices, index.ntotal)

# updating 250 packets
update_indices = np.random.choice(index.ntotal, 250, replace=False)
new_packet_texts = [f"10.0.0.{i} 10.0.0.{i+1} UDP {i*5}" for i in range(250)]
pid = os.getpid()
index = measure(pid, update_faiss, index, update_indices, new_packet_texts, index.ntotal)
