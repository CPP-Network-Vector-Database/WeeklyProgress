import uuid
import time
import psutil
import cohere
import csv
import os
from tabulate import tabulate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Configuration
QDRANT_HOST = "http://localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 8
COHERE_API_KEY = "enter-api-key"
MODEL_NAME = "embed-english-light-v3.0"
VECTOR_SIZE = 384
DATASET_PATH = "C:/Users/thest\ip_flow/ip_flow_dataset.csv"  # Update this with your actual file path

# Benchmark configurations
CONFIGURATIONS = [
    {"name": "HNSW Baseline", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
    {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True)},
    {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
    {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True)}
]

# Initialize Cohere
try:
    co = cohere.Client(COHERE_API_KEY)
    test_response = co.embed(texts=["API test"], model=MODEL_NAME, input_type="search_document", truncate="END")
    if not test_response.embeddings:
        raise ValueError("API test failed")
    print("✅ Cohere API connection successful")
except Exception as e:
    print(f"❌ Cohere API Error: {str(e)}")
    exit(1)

# Initialize Qdrant
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Load and process the dataset
def load_data(file_path, max_records=1000):
    """Load data from the CSV file and prepare for embedding"""
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        exit(1)
    
    data = []
    ids = []
    payloads = []
    
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            
            for i, row in enumerate(reader):
                if i >= max_records:
                    break
                
                # Extract fields
                frame_number = row[0].strip('"')
                frame_time = row[1].strip('"')
                ip_src = row[2].strip('"')
                ip_dst = row[3].strip('"')
                tcp_srcport = row[4].strip('"')
                tcp_dstport = row[5].strip('"')
                protocol = row[6].strip('"')
                frame_len = row[7].strip('"')
                
                # Create a unique ID
                record_id = str(uuid.uuid4())
                ids.append(record_id)
                
                # Create a document string for embedding
                document = f"{ip_src}:{tcp_srcport} -> {ip_dst}:{tcp_dstport} {protocol} {frame_len}bytes {frame_time}"
                data.append(document)
                
                # Create payload
                payload = {
                    "frame_number": frame_number,
                    "frame_time": frame_time,
                    "ip_src": ip_src,
                    "ip_dst": ip_dst,
                    "tcp_srcport": tcp_srcport,
                    "tcp_dstport": tcp_dstport,
                    "protocol": protocol,
                    "frame_len": frame_len
                }
                payloads.append(payload)
                
        print(f"✅ Loaded {len(data)} records from dataset")
        return ids, data, payloads
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit(1)

# Load data
ids, documents, payloads = load_data(DATASET_PATH)
NUM_RECORDS = len(documents)

# Generate embeddings
vectors = []
for i in range(0, NUM_RECORDS, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    if not batch:
        continue
        
    try:
        response = co.embed(
            texts=batch,
            model=MODEL_NAME,
            input_type="search_document",
            truncate="END"
        )
        vectors.extend(response.embeddings)
        print(f"✅ Processed batch {i//BATCH_SIZE + 1}/{(NUM_RECORDS//BATCH_SIZE)+1}")
        time.sleep(1)  # Rate limiting
    except Exception as e:
        print(f"🚨 Batch error: {str(e)}")
        exit(1)

def benchmark(operation):
    """Benchmarking utility"""
    process = psutil.Process()
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    result = operation()
    
    latency = (time.time() - start_time) * 1000
    cpu_usage = process.cpu_percent(interval=None) - cpu_before
    mem_usage = (process.memory_info().rss / (1024 ** 2)) - mem_before
    throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
    return (latency, cpu_usage, mem_usage, throughput)

def run_benchmarks():
    results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
    
    for config in CONFIGURATIONS:
        collection_name = f"flow_data_benchmark_{config['name'].lower().replace(' ', '_')}"
        
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=config["vector_config"]
        )

        # Define operations
        def insert():
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=payloads[i]
                ) for i in range(len(ids))
            ]
            client.upsert(collection_name=collection_name, points=points)
            return points

        def search():
            # Search for similar network flows
            return client.search(
                collection_name=collection_name,
                query_vector=vectors[0],
                limit=10
            )

        def update():
            # Update 10 records with modified protocol info
            update_ids = ids[:10]
            updated_docs = []
            
            for i in range(10):
                # Create modified document with "MODIFIED" protocol
                ip_src = payloads[i]["ip_src"]
                ip_dst = payloads[i]["ip_dst"]
                tcp_srcport = payloads[i]["tcp_srcport"]
                tcp_dstport = payloads[i]["tcp_dstport"]
                frame_time = payloads[i]["frame_time"]
                frame_len = payloads[i]["frame_len"]
                
                updated_doc = f"{ip_src}:{tcp_srcport} -> {ip_dst}:{tcp_dstport} MODIFIED {frame_len}bytes {frame_time}"
                updated_docs.append(updated_doc)
            
            # Get embeddings for updated documents
            response = co.embed(texts=updated_docs, model=MODEL_NAME, input_type="search_document", truncate="END")
            updated_vectors = response.embeddings
            
            points = [
                PointStruct(
                    id=uid,
                    vector=vec,
                    payload={**payloads[i], "protocol": "MODIFIED"}
                ) for i, (uid, vec) in enumerate(zip(update_ids, updated_vectors))
            ]
            client.upsert(collection_name=collection_name, points=points)
            return points

        def delete():
            delete_ids = ids[:10]
            client.delete(collection_name=collection_name, points_selector=delete_ids)
            return delete_ids

        # Run benchmarks
        print(f"\nRunning benchmarks for configuration: {config['name']}")
        results["INSERT"].append((config["name"], *benchmark(insert)))
        results["SEARCH"].append((config["name"], *benchmark(search)))
        results["UPDATE"].append((config["name"], *benchmark(update)))
        results["DELETE"].append((config["name"], *benchmark(delete)))

        # Clean up
        client.delete_collection(collection_name)

    # Format results
    headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
    table_data = []
    
    for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
        row = [op]
        for result in results[op]:
            config_name, latency, cpu_usage, mem_usage, throughput = result
            row.append(
                f"Latency: {latency:.2f}ms\n"
                f"CPU: {cpu_usage:.2f}%\n"
                f"Memory: {mem_usage:.2f}MB\n"
                f"Throughput: {throughput:.2f} ops/s"
            )
        table_data.append(row)
    
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    run_benchmarks()

# import uuid
# import time
# import psutil
# import cohere
# from tabulate import tabulate
# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, VectorParams, Distance

# # Configuration
# QDRANT_HOST = "http://localhost"
# QDRANT_PORT = 6333
# NUM_RECORDS = 1000
# BATCH_SIZE = 8
# COHERE_API_KEY = ""
# MODEL_NAME = "embed-english-light-v3.0"
# VECTOR_SIZE = 384

# # Benchmark configurations
# CONFIGURATIONS = [
#     {"name": "HNSW Baseline", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
#     {"name": "Disk-based Indexing Only", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True)},
#     {"name": "Payload-based Indexing Only", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
#     {"name": "Disk-based + Payload Indexing", "vector_config": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True)}
# ]

# # Initialize Cohere
# try:
#     co = cohere.Client(COHERE_API_KEY)
#     test_response = co.embed(texts=["API test"], model=MODEL_NAME, input_type="search_document", truncate="END")
#     if not test_response.embeddings:
#         raise ValueError("API test failed")
#     print("✅ Cohere API connection successful")
# except Exception as e:
#     print(f"❌ Cohere API Error: {str(e)}")
#     exit(1)

# # Initialize Qdrant
# client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# # Generate sample data
# ids = [str(uuid.uuid4()) for _ in range(NUM_RECORDS)]
# sources = [f"192.168.1.{i%254}" for i in range(NUM_RECORDS)]
# destinations = [f"10.0.0.{i%254}" for i in range(NUM_RECORDS)]
# protocols = ["HTTP", "HTTPS", "SSH", "DNS", "FTP"]
# documents = [f"{s}, {d}, {p}" for s, d, p in zip(sources, destinations, protocols)]

# # Generate embeddings
# vectors = []
# for i in range(0, NUM_RECORDS, BATCH_SIZE):
#     batch = documents[i:i+BATCH_SIZE]
#     if not batch:
#         continue
        
#     try:
#         response = co.embed(
#             texts=batch,
#             model=MODEL_NAME,
#             input_type="search_document",
#             truncate="END"
#         )
#         vectors.extend(response.embeddings)
#         print(f"✅ Processed batch {i//BATCH_SIZE + 1}/{(NUM_RECORDS//BATCH_SIZE)+1}")
#         time.sleep(1)
#     except Exception as e:
#         print(f"🚨 Batch error: {str(e)}")
#         exit(1)

# def benchmark(operation):
#     """Benchmarking utility"""
#     process = psutil.Process()
#     start_time = time.time()
#     cpu_before = process.cpu_percent(interval=None)
#     mem_before = process.memory_info().rss / (1024 ** 2)
    
#     result = operation()
    
#     latency = (time.time() - start_time) * 1000
#     cpu_usage = process.cpu_percent(interval=None) - cpu_before
#     mem_usage = (process.memory_info().rss / (1024 ** 2)) - mem_before
#     throughput = len(result) / (latency / 1000) if latency > 0 else 0
    
#     return (latency, cpu_usage, mem_usage, throughput)

# def run_benchmarks():
#     results = {op: [] for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]}
    
#     for config in CONFIGURATIONS:
#         collection_name = f"cohere_benchmark_{config['name'].lower().replace(' ', '_')}"
        
#         if client.collection_exists(collection_name):
#             client.delete_collection(collection_name)
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=config["vector_config"]
#         )

#         # Define operations
#         def insert():
#             points = [
#                 PointStruct(
#                     id=ids[i],
#                     vector=vectors[i],
#                     payload={"source": s, "destination": d, "protocol": p}
#                 ) for i, (s, d, p) in enumerate(zip(sources, destinations, protocols))
#             ]
#             client.upsert(collection_name=collection_name, points=points)
#             return points

#         def search():
#             return client.search(
#                 collection_name=collection_name,
#                 query_vector=vectors[0],
#                 limit=1
#             )

#         def update():
#             update_ids = ids[:10]
#             new_docs = [f"Updated {doc}" for doc in documents[:10]]
#             response = co.embed(texts=new_docs, model=MODEL_NAME, input_type="search_document", truncate="END")
#             points = [
#                 PointStruct(
#                     id=uid,
#                     vector=vec,
#                     payload={"source": "updated", "destination": "updated", "protocol": "UPDATED"}
#                 ) for uid, vec in zip(update_ids, response.embeddings)
#             ]
#             client.upsert(collection_name=collection_name, points=points)
#             return points

#         def delete():
#             delete_ids = ids[:10]
#             client.delete(collection_name=collection_name, points_selector=delete_ids)
#             return delete_ids

#         # Run benchmarks
#         results["INSERT"].append((config["name"], *benchmark(insert)))
#         results["SEARCH"].append((config["name"], *benchmark(search)))
#         results["UPDATE"].append((config["name"], *benchmark(update)))
#         results["DELETE"].append((config["name"], *benchmark(delete)))

#         client.delete_collection(collection_name)

#     # Format results
#     headers = ["Operation"] + [config["name"] for config in CONFIGURATIONS]
#     table_data = []
#     for op in ["INSERT", "SEARCH", "UPDATE", "DELETE"]:
#         row = [op]
#         for metric in results[op]:
#             row.append(
#                 f"Latency: {metric[1]:.2f}ms\n"
#                 f"CPU: {metric[2]:.2f}%\n"
#                 f"Memory: {metric[3]:.2f}MB\n"
#                 f"Throughput: {metric[4]:.2f} ops/s"
#             )
#         table_data.append(row)
    
#     print("\nBenchmark Results:")
#     print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

# if __name__ == "__main__":
#     run_benchmarks()

