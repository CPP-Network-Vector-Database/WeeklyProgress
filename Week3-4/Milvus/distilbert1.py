import pandas as pd
import numpy as np
import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import socket
import struct


file_path = "./Netflix.xlsx"
df = pd.read_excel(file_path)

print(df.head())

# Function to convert IP addresses to numerical values
def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return 0  

# Convert IPs to numerical format
df['ip.src'] = df['ip.src'].apply(ip_to_int)
df['ip.dst'] = df['ip.dst'].apply(ip_to_int)

# Normalize port numbers
df['tcp.srcport'] = df['tcp.srcport'] / 65535
df['tcp.dstport'] = df['tcp.dstport'] / 65535

# Convert Protocol to categorical values
df['_ws.col.Protocol'] = df['_ws.col.Protocol'].astype('category').cat.codes


vectors = np.vstack((df[['ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', '_ws.col.Protocol']].values))


connections.connect("default", host="localhost", port="19530")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1])
]

schema = CollectionSchema(fields, description="Network Traffic with Numerical Embeddings")
collection = Collection("network_traffic_numerical", schema)

data = [vectors.tolist()]
collection.insert(data)
print("Inserted data into Milvus successfully!")

# Create an index for fast search
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
