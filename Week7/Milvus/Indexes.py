import pandas as pd
import numpy as np
import torch
import time
import psutil
import threading
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import utility
from transformers import DistilBertTokenizer, DistilBertModel

# ----------------- CPU and Memory Monitoring Setup -----------------
cpu_usage_data = []
memory_usage_data = []
recording = True

def record_system_resources():
    """Continuously record CPU and memory usage while the flag is set."""
    global recording, cpu_usage_data, memory_usage_data
    process = psutil.Process()
    while recording:
        cpu_usage_data.append(psutil.cpu_percent(interval=0.01) / psutil.cpu_count())
        memory_usage_data.append(process.memory_info().rss / (1024 * 1024))  # MB
        time.sleep(0.01)

def start_resource_monitor():
    """Start the CPU and memory usage monitoring thread."""
    global recording, cpu_usage_data, memory_usage_data
    recording = True
    cpu_usage_data = []
    memory_usage_data = []
    monitor_thread = threading.Thread(target=record_system_resources, daemon=True)
    monitor_thread.start()

def stop_resource_monitor():
    """Stop CPU and memory monitoring and save logs to files."""
    global recording
    recording = False
   
    
    # Save memory usage
    with open("memory_usage_log.txt", "w") as f:
        for value in memory_usage_data:
            f.write(f"{value}\n")
   
    return {
        "cpu_usage": cpu_usage_data,
        "memory_usage": memory_usage_data
    }

def get_system_resource_snapshot():
    """Get a snapshot of current system resource usage."""
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=0.01) / psutil.cpu_count()
    mem_usage = process.memory_info().rss / (1024 * 1024)  
    return {
        "cpu_usage": cpu_usage,
        "memory_usage_MB": mem_usage
    }


def analyze_port_distances(df):
    """
    Analyze the distances between occurrences of unique ports in the dataset.
    Returns a DataFrame with port with various distance metrics.
    """
    print("\n======= Analyzing Port Distance Distribution =======")
   
    # Create a dictionary to store information for each port
    port_info = {}
   
    # Iterate through the dataframe
    for idx, port in enumerate(df['tcp.srcport']):
        if port not in port_info:
            port_info[port] = {
                'positions': [idx],
                'count': 1
            }
        else:
            port_info[port]['positions'].append(idx)
            port_info[port]['count'] += 1
   
    # Calculate distances between consecutive occurrences
    results = []
    for port, info in port_info.items():
        if len(info['positions']) > 1:
            distances = [info['positions'][i] - info['positions'][i-1] for i in range(1, len(info['positions']))]
           
            result = {
                'port': port,
                'frequency': info['count'],
                'avg_distance': np.mean(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'std_distance': np.std(distances),
                'first_position': info['positions'][0],
                'last_position': info['positions'][-1]
            }
            results.append(result)
   
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('frequency', ascending=False).reset_index(drop=True)
   
    
    print(f"Found {len(results_df)} unique ports with multiple occurrences")
    print("\nTop 10 most frequent ports:")
    print(results_df.head(10)[['port', 'frequency', 'avg_distance', 'min_distance', 'max_distance']])
   
    return results_df

def benchmark_query_with_port_distances(collection, query_vector, metric, df, port_distances_df, top_k=10):
    
    print(f"\nTesting with Distance Metric: {metric}")

    # Get initial resource usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  
    initial_cpu = process.cpu_percent(interval=0.01) / psutil.cpu_count()

    start_resource_monitor()
   
    
    try:
        
        collection.release()
        collection.drop_index()
       
        # Create index with the specified metric
        index_params = {}
        if metric == "COSINE":
            index_params = {
                "index_type": "HNSW",
                "metric_type": metric,
                "params": {"M": 16, "efConstruction": 200}
            }
        else:
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": metric,
                "params": {"nlist": 128}
            }
       
        print(f"Creating {metric} index...")
        collection.create_index("embedding", index_params)
       
        
        collection.load()
       
        
        query_start_time = time.time()
        results = collection.search(
            [query_vector], "embedding",
            {"metric_type": metric, "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["tcp_srcport"]
        )
        query_end_time = time.time()
    except Exception as e:
        print(f"Error creating index or searching: {e}")
        resource_data = stop_resource_monitor()
        return {
            "results": None,
            "query_latency": 0,
            "avg_cpu_usage": 0,
            "peak_memory_usage_MB": 0,
            "final_memory_usage_MB": 0,
            "memory_delta_MB": 0
        }

    # Get final resource usage
    resource_data = stop_resource_monitor()
    final_snapshot = get_system_resource_snapshot()
   
    duration = query_end_time - query_start_time
    avg_cpu = np.mean(resource_data["cpu_usage"]) if resource_data["cpu_usage"] else final_snapshot["cpu_usage"]
    max_memory = np.max(resource_data["memory_usage"]) if resource_data["memory_usage"] else final_snapshot["memory_usage_MB"]

    print(f"=== {metric} Metric Results ===")
    print(f"Query Latency: {duration:.6f} seconds")
    print(f"Average CPU Usage: {avg_cpu:.2f}%")
    print(f"Peak Memory Usage: {max_memory:.2f} MB")
    print(f"Memory Delta: {final_snapshot['memory_usage_MB'] - initial_memory:.2f} MB")

    
    print(f"\nQuery Results with Distance Metrics:")
   
    found_ports = []
    prev_port = None
    port_vector_distances = []
    row_distances = []
   
    if results:
        for hit in results[0]:
            hit_port = hit.entity.get('tcp_srcport')
            found_ports.append(hit_port)
           
            # Get port distance information if available in our distance dataframe
            port_info = port_distances_df[port_distances_df['port'] == hit_port]
           
            if not port_info.empty:
                freq = port_info.iloc[0]['frequency']
                avg_dist = port_info.iloc[0]['avg_distance']
               
                print(f"Port: {hit_port}, Vector Distance: {hit.distance:.6f}, Frequency: {freq}, Avg Row Distance: {avg_dist:.2f}")
               
                
                if prev_port is not None:
                    
                    curr_pos = df[df['tcp.srcport'] == hit_port].index.tolist()
                    prev_pos = df[df['tcp.srcport'] == prev_port].index.tolist()
                   
                    if curr_pos and prev_pos:
                        # Find the minimum distance between any occurrence of the previous port and the current port
                        min_row_dist = min([abs(c - p) for c in curr_pos for p in prev_pos])
                        row_distances.append(min_row_dist)
               
                port_vector_distances.append(hit.distance)
                prev_port = hit_port
            else:
                print(f"Port: {hit_port}, Vector Distance: {hit.distance:.6f}, Distance Info: Not available")
   
    
    if row_distances:
        print("\nDistance Statistics for Consecutive Query Results:")
        print(f"Average Row Distance Between Consecutive Results: {np.mean(row_distances):.2f}")
        print(f"Min Row Distance: {np.min(row_distances)}, Max Row Distance: {np.max(row_distances)}")
       
    if port_vector_distances:
        print(f"\nVector Distance Statistics:")
        print(f"Average Vector Distance: {np.mean(port_vector_distances):.6f}")
        print(f"Min Vector Distance: {np.min(port_vector_distances):.6f}, Max Vector Distance: {np.max(port_vector_distances):.6f}")

    return {
        "results": results,
        "found_ports": found_ports,
        "query_latency": duration,
        "avg_cpu_usage": avg_cpu,
        "peak_memory_usage_MB": max_memory,
        "final_memory_usage_MB": final_snapshot["memory_usage_MB"],
        "memory_delta_MB": final_snapshot["memory_usage_MB"] - initial_memory,
        "row_distances": row_distances if row_distances else None,
        "vector_distances": port_vector_distances if port_vector_distances else None
    }


connections.connect("default", host="localhost", port="19530")


file_path = "./Netflix.xlsx"
df = pd.read_excel(file_path)


df['tcp.srcport'] = df['tcp.srcport'].astype(str)


port_distances_df = analyze_port_distances(df)

# Generate DistilBERT Embeddings

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()

def get_distilbert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()


df['port_embedding'] = df['tcp.srcport'].apply(get_distilbert_embedding)
vectors = np.vstack(df['port_embedding'].values)
print(f"Shape of ALL Source Port Embeddings: {vectors.shape}")


collection_name = "source_port_embeddings"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped existing collection: {collection_name}")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
    FieldSchema(name="tcp_srcport", dtype=DataType.VARCHAR, max_length=20)
]
schema = CollectionSchema(fields, description="Source Ports Embedded")


collection = Collection(name=collection_name, schema=schema)


insert_data = [
    vectors.tolist(),
    df['tcp.srcport'].tolist()
]
collection.insert(insert_data)
collection.flush()
print(f"Inserted {collection.num_entities} entities into collection")


# Create initial index 

index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

print("Creating initial L2 index...")
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
print("Collection loaded successfully!")


# Select the top 5 most frequent ports as queries
top_ports = port_distances_df.head(5)['port'].tolist()

print(f"Top 5 most frequent ports selected for search: {top_ports}")

# Run benchmarks for each metric separately
metrics = ["L2", "IP", "COSINE"]

for port in top_ports:
    print(f"\n Benchmarking for Port: {port}")
    query_embedding = get_distilbert_embedding(port).tolist()
   
    
    for metric in metrics:
        try:
            benchmark_query_with_port_distances(collection, query_embedding, metric, df, port_distances_df)
        except Exception as e:
            print(f"Error benchmarking {metric} for port {port}: {e}")


print("\n\nBenchmarking ports with varying frequencies:")
# Sample ports with different frequencies
low_freq_port = port_distances_df[port_distances_df['frequency'] < 10].iloc[0]['port'] if not port_distances_df[port_distances_df['frequency'] < 10].empty else None
mid_freq_port = port_distances_df[(port_distances_df['frequency'] >= 10) & (port_distances_df['frequency'] < 50)].iloc[0]['port'] if not port_distances_df[(port_distances_df['frequency'] >= 10) & (port_distances_df['frequency'] < 50)].empty else None
high_freq_port = port_distances_df[port_distances_df['frequency'] >= 50].iloc[0]['port'] if not port_distances_df[port_distances_df['frequency'] >= 50].empty else None

test_ports = [p for p in [low_freq_port, mid_freq_port, high_freq_port] if p is not None]
print(f"Testing ports with varying frequencies: {test_ports}")

for port in test_ports:
    freq = port_distances_df[port_distances_df['port'] == port].iloc[0]['frequency']
    avg_dist = port_distances_df[port_distances_df['port'] == port].iloc[0]['avg_distance']
    print(f"\n Benchmarking for Port: {port} (Frequency: {freq}, Avg Distance: {avg_dist:.2f})")
   
    query_embedding = get_distilbert_embedding(port).tolist()
   
    
    try:
        benchmark_query_with_port_distances(collection, query_embedding, "L2", df, port_distances_df)
    except Exception as e:
        print(f"Error: {e}")


print("\nCleaning up...")
utility.drop_collection(collection_name)
print("Complete!")
	
