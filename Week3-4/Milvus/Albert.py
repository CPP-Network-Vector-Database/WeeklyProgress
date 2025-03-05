import pandas as pd
import numpy as np
import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from transformers import AlbertTokenizer, AlbertModel
import torch


file_path = "./Netflix.xlsx"
df = pd.read_excel(file_path)


print(df.head())

# Function to convert IP addresses to string format
def ip_to_str(ip):
    return str(ip)

# Normalize port numbers and convert to string
df['tcp.srcport'] = df['tcp.srcport'].astype(str)
df['tcp.dstport'] = df['tcp.dstport'].astype(str)
df['ip.src'] = df['ip.src'].apply(ip_to_str)
df['ip.dst'] = df['ip.dst'].apply(ip_to_str)


tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
model = AlbertModel.from_pretrained("albert-base-v2")

# generating ALBERT embeddings
def get_albert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# Combine all features into a structured text format
df['combined_text'] = df.apply(lambda row: f"Source IP: {row['ip.src']}, Destination IP: {row['ip.dst']}, "
                                           f"Source Port: {row['tcp.srcport']}, Destination Port: {row['tcp.dstport']}, "
                                           f"Protocol: {row['_ws.col.Protocol']}" , axis=1)


df['text_embedding'] = df['combined_text'].apply(get_albert_embedding)
vectors = np.vstack(df['text_embedding'].values)


connections.connect("default", host="localhost", port="19530")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1])
]

schema = CollectionSchema(fields, description="Network Traffic with ALBERT Embeddings")
collection = Collection("network_traffic_albert", schema)

data = [vectors.tolist()]
collection.insert(data)
print("Inserted data into Milvus successfully!")

# Creating index for fast search
collection.create_index("vector", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
collection.load()

# Function to Measure Query Latency
def measure_query_latency(top_k):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    query_vector = vectors[0].tolist()  

    start_time = time.time()
    results = collection.search([query_vector], "vector", search_params, limit=top_k)
    end_time = time.time()

    latency = end_time - start_time
    print(f"Query latency for top {top_k} results: {latency:.6f} seconds")
    return results, latency


while True:
    try:
        top_k = int(input("\nEnter the number of top results to retrieve (or type 0 to exit): "))
        if top_k == 0:
            print("Exiting CLI.")
            break

        results, latency = measure_query_latency(top_k)
        print(f"\nTop {top_k} Results:")
        for i, result in enumerate(results[0]):
            print(f"{i+1}. ID: {result.id}")

        print(f"\nQuery Latency: {latency:.6f} seconds")

    except ValueError:
        print("Invalid input! Please enter a number.")
