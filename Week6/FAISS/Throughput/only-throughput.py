import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
from sentence_transformers import SentenceTransformer

def measure(pid, func, *args, **kwargs):
    process = psutil.Process(pid)
    start_cpu = process.cpu_percent(interval=None)
    start_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_cpu = process.cpu_percent(interval=None)
    end_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    end_time = time.time()
    
    cpu_usage = end_cpu - start_cpu
    mem_usage = end_mem - start_mem
    execution_time = end_time - start_time
    
    return execution_time, cpu_usage, mem_usage # same fnction as before

def measure_throughput(pid, func, num_operations, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time  # Total execution time
    
    throughput = num_operations / execution_time if execution_time > 0 else 0
    return execution_time, throughput

performance_log = []

def log_performance(model_name, operation, execution_time, throughput, cpu_usage, mem_usage):
    performance_log.append([model_name, operation, execution_time, throughput, cpu_usage, mem_usage])

# usual FAISS Operations
def query_faiss(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices

def insert_data(index, new_embeddings):
    index.add(new_embeddings)

def delete_data(index, delete_indices, total_embeddings):
    delete_indices = np.array(delete_indices)
    delete_indices = delete_indices[delete_indices < total_embeddings]
    index.remove_ids(delete_indices)
    return index

def update_faiss(index, update_indices, new_packet_texts, total_embeddings, model):
    index = delete_data(index, update_indices, total_embeddings)
    new_embeddings = model.encode(new_packet_texts, convert_to_numpy=True)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    index.add(new_embeddings)
    return index

# Load data
csv_path = "/home/pes1ug22am100/Documents/WeeklyProgress/Week6/FAISS/Pipeline/dirA.125910-packets.csv"
df = pd.read_csv(csv_path, header=None, names=["timestamp", "src_ip", "dst_ip", "protocol", "size"])
df["packet_text"] = df["src_ip"] + " " + df["dst_ip"] + " " + df["protocol"] + " " + df["size"].astype(str)

modelList = [
    'paraphrase-MiniLM-L12-v2', 
    "all-MiniLM-L6-v2", 
    'distilbert-base-nli-stsb-mean-tokens', 
    'microsoft/codebert-base', 
    'bert-base-nli-mean-tokens', 
    'sentence-transformers/average_word_embeddings_komninos', 
    'all-mpnet-base-v2'
]

for mod in modelList: 
    model = SentenceTransformer(mod)
    embeddings = model.encode(df["packet_text"].tolist(), convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    nlist = min(100, int(np.sqrt(len(embeddings))))  
    
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)

    faiss.write_index(index, "packet_embeddings.index")
    df.to_csv("packet_metadata.csv", index=False)
    
    query_text = ["192.168.1.1 192.168.1.2 TCP 100"]
    query_embedding = model.encode(query_text, convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    pid = os.getpid()
    
    # Measure Query Throughput
    exec_time, throughput = measure_throughput(pid, query_faiss, 5, index, query_embedding, 5)
    log_performance(mod, "Query Throughput", exec_time, throughput, 0, 0)
    
    # Insert 250 new packets
    new_packet_texts = [f"192.168.1.{i} 192.168.1.{i+1} TCP {i*10}" for i in range(250)]
    new_embeddings = model.encode(new_packet_texts, convert_to_numpy=True)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    
    exec_time, throughput = measure_throughput(pid, insert_data, new_embeddings.shape[0], index, new_embeddings)
    log_performance(mod, "Insertion Throughput", exec_time, throughput, 0, 0)
    
    # Delete 250 random packets
    delete_indices = np.random.choice(index.ntotal, 250, replace=False)
    exec_time, throughput = measure_throughput(pid, delete_data, len(delete_indices), index, delete_indices, index.ntotal)
    log_performance(mod, "Deletion Throughput", exec_time, throughput, 0, 0)
    
    # Update 250 packets
    update_indices = np.random.choice(index.ntotal, 250, replace=False)
    new_packet_texts = [f"10.0.0.{i} 10.0.0.{i+1} UDP {i*5}" for i in range(250)]
    exec_time, throughput = measure_throughput(pid, update_faiss, len(update_indices), index, update_indices, new_packet_texts, index.ntotal, model)
    log_performance(mod, "Update Throughput", exec_time, throughput, 0, 0)

# Save performance logs to CSV
performance_df = pd.DataFrame(performance_log, columns=["Model", "Operation", "Time (s)", "Throughput (ops/sec)", "CPU (%)", "Memory (MB)"])
performance_df.to_csv("/home/pes1ug22am100/Documents/WeeklyProgress/Week6/FAISS/Throughput/faiss_performance.csv", index=False)

print("Performance metrics saved to faiss_performance.csv")
