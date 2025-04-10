import psutil
import pandas as pd
import numpy as np
import time
import threading
from pymilvus import connections, Collection, FieldSchema,CollectionSchema, DataType
from transformers import DistilBertTokenizer, DistilBertModel
import torch



cpu_usage_data = []
recording = True

def record_cpu_usage():
     
     global recording
     while recording:
         cpu_usage_data.append(psutil.cpu_percent(interval=0.01) /
psutil.cpu_count())

def start_cpu_monitor():
     """Start the CPU usage monitoring thread."""
     global recording
     recording = True
     monitor_thread = threading.Thread(target=record_cpu_usage, daemon=True)
     monitor_thread.start()

def stop_cpu_monitor():
     """Stop CPU monitoring """
     global recording
     recording = False
     



file_path = "./Netflix.xlsx"
df = pd.read_excel(file_path)

# Convert IP addresses and ports to string format
df['tcp.srcport'] = df['tcp.srcport'].astype(str)
df['tcp.dstport'] = df['tcp.dstport'].astype(str)
df['ip.src'] = df['ip.src'].astype(str)
df['ip.dst'] = df['ip.dst'].astype(str)


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_distilbert_embedding(text):
     """Generate DistilBERT embeddings for the input"""
     inputs = tokenizer(text, return_tensors="pt", truncation=True,
padding=True, max_length=128)
     with torch.no_grad():
         outputs = model(**inputs)
     return outputs.last_hidden_state[:, 0, :].numpy().flatten()

df['combined_text'] = df.apply(lambda row: f"Source IP: {row['ip.src']},
Destination IP: {row['ip.dst']}, "
                                            f"Source Port:
{row['tcp.srcport']}, Destination Port: {row['tcp.dstport']}, "
                                            f"Protocol:
{row['_ws.col.Protocol']}" , axis=1)

df['text_embedding'] = df['combined_text'].apply(get_distilbert_embedding)
vectors = np.vstack(df['text_embedding'].values)



connections.connect("default", host="localhost", port="19530")

fields = [
     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True,auto_id=True),
     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,dim=vectors.shape[1])
]

schema = CollectionSchema(fields, description="Network Traffic with DistilBERT Embeddings")
collection = Collection("network_traffic_distilbert11", schema)

data = [vectors.tolist()]
collection.insert(data)
print("Inserted data into Milvus successfully!")

collection.load()



def benchmark_query(query_vector, metric, top_k=5):
     
     print(f"\nTesting with Distance Metric: {metric}")

     start_cpu_monitor()
     start_time = time.time()


     collection.release()


     collection.drop_index()


     collection.create_index("vector", {
         "index_type": "IVF_FLAT",
         "metric_type": metric,
         "params": {"nlist": 128}
     })


     collection.load()

     # Search
     query_start_time = time.time()
     results = collection.search(
         [query_vector], "vector",
         {"metric_type": metric, "params": {"nprobe": 16}},
         limit=top_k
     )
     query_end_time = time.time()

     duration = query_end_time - query_start_time
     stop_cpu_monitor()

    
     process = psutil.Process()
     cpu_usage = process.cpu_percent(interval=0.01) / psutil.cpu_count()
     mem_usage = process.memory_info().rss / (1024 * 1024)

     return {
         "results": results,
         "query_latency": duration,
         "cpu_usage": cpu_usage,
         "memory_usage_MB": mem_usage
     }



metrics = ["L2", "IP", "COSINE"]
query_vector = vectors[0].tolist()
top_k = 5

for metric in metrics:
     results = benchmark_query(query_vector, metric, top_k)


     print(f"\n=== {metric} Metric Results ===")
     print(f"Query Latency: {results['query_latency']:.6f} seconds")
     print(f"CPU Usage: {results['cpu_usage']:.2f}%")
     print(f"Memory Usage: {results['memory_usage_MB']:.2f} MB")

     
     for i, hits in enumerate(results['results']):
         for j, hit in enumerate(hits):
              print(f"Query {i+1}, Result {j+1}: ID = {hit.id}, Distance= {hit.distance:.6f}")



	
