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
df['_ws.col.Protocol'] = df['_ws.col.Protocol'].astype('category')
protocol_mapping = dict(enumerate(df['_ws.col.Protocol'].cat.categories))
df['_ws.col.Protocol'] = df['_ws.col.Protocol'].cat.codes

print("Protocol Mapping:", protocol_mapping)


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


def retrieve_top_source_ips():
    return df['ip.src'].value_counts().head(5)

def retrieve_top_source_ports():
    return df['tcp.srcport'].value_counts().head(5)

def find_protocols_common_ip():
    return df.groupby(['ip.src', 'ip.dst'])['_ws.col.Protocol'].apply(set).head(5)

def retrieve_frequent_destination_ports():
    return df['tcp.dstport'].value_counts().head(5)

def retrieve_common_protocols():
    return df['_ws.col.Protocol'].value_counts().head(5).rename(index=protocol_mapping)


while True:
    print("\nChoose a Query:")
    print("1. Retrieve top 5 frequently used source IP addresses")
    print("2. Retrieve top 5 source ports")
    print("3. Find protocols that connect common source and destination IP")
    print("4. Retrieve frequently accessed destination ports")
    print("5. Retrieve commonly used protocols")
    print("0. Exit")
    
    try:
        choice = int(input("Enter your choice: "))
        if choice == 0:
            print("Exiting CLI.")
            break
        elif choice == 1:
            print(retrieve_top_source_ips())
        elif choice == 2:
            print(retrieve_top_source_ports())
        elif choice == 3:
            print(find_protocols_common_ip())
        elif choice == 4:
            print(retrieve_frequent_destination_ports())
        elif choice == 5:
            print(retrieve_common_protocols())
        else:
            print("Invalid choice! Please enter a number between 0 and 5.")
    except ValueError:
        print("Invalid input! Please enter a number.")
