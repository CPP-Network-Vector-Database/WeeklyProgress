import chromadb
import time
import psutil
import multiprocessing
import uuid
import os
import random
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import csv
import gc
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('embeddingModel')
parser.add_argument('distanceMetric')
parser.add_argument('maxSeqLength', type = int)
parser.add_argument('num_records', type = int)

args = parser.parse_args()

embedding_model = SentenceTransformer(args.embeddingModel, device="cpu")
embedding_model.max_seq_length = args.maxSeqLength

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts, num_workers=0).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name=args.embeddingModel+'_'+args.distanceMetric+'_'+str(args.maxSeqLength), embedding_function=SentenceTransformerEmbeddingFunction(), metadata={"hnsw:space": args.distanceMetric})

def display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage, operations):
    total_time = end_time - start_time
    throughput = operations / total_time if total_time > 0 else 0
    mem_per_op = mem_usage / operations if operations > 0 else 0
    cpu_per_op = cpu_percentage / operations if operations > 0 else 0
    sys_mem_per_op = memory_usage_percentage / operations if operations > 0 else 0

    print("--- Performance Metrics ---")
    print(f"Total operations: {operations}")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Total memory usage: {mem_usage:.2f} MB")
    print(f"Total CPU time usage: {cpu_percentage:.2f}%")
    print(f"Total system memory usage: {memory_usage_percentage:.2f}%")
    print(f"Throughput: {throughput:.2f} operations/sec\n")
    print("--- Performance Per Operation ---")
    print(f"Time per operation: {total_time / operations:.6f} sec/op" if operations > 0 else "Time per operation: N/A")
    print(f"Memory per operation: {mem_per_op:.6f} MB/op" if operations > 0 else "Memory per operation: N/A")
    print(f"CPU per operation: {cpu_per_op:.6f}%" if operations > 0 else "CPU per operation: N/A")
    print(f"System memory per operation: {sys_mem_per_op:.6f}%" if operations > 0 else "System memory per operation: N/A")
    print("-------------------------\n")

def read_documents_from_file(file_path):
    documents = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.num_records:
                break
            frame_num = row["frame.number"]
            ip_src = row["ip.src"]
            ip_dst = row["ip.dst"]
            srcport = row["tcp.srcport"]
            dstport = row["tcp.dstport"]
            protocol = row["_ws.col.protocol"]
            combined = f"{frame_num},{ip_src},{ip_dst},{srcport},{dstport},{protocol}"
            documents.append(combined)
    return documents

all_ids = []
document_chunks = [] 
id_chunks = [] 
shuffled_document_chunks = [] 


def split_documents(documents, num_workers):
    base_chunk_size = len(documents) // num_workers
    remainder = len(documents) % num_workers
    chunks = []
    start = 0
    for i in range(num_workers):
        end = start + base_chunk_size + (1 if i < remainder else 0)
        chunks.append(documents[start:end])
        start = end
    return chunks



def insert_chunk(chunk_number):
    global collection, embedding_function, document_chunks
    chunk = document_chunks[chunk_number]
    pid = os.getpid()
    print(f"Inserting chunk in PID {pid}")
    ids = [str(uuid.uuid4()) for _ in range(len(chunk))]
    metadata = [{"document": doc} for doc in chunk]
    process = psutil.Process(pid)
    total_cores = psutil.cpu_count(logical=False)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    embeddings = embedding_function(chunk)    
    collection.add(documents=chunk, ids=ids, embeddings=embeddings, metadatas=metadata)
    
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 ** 2)
    cpu_end = process.cpu_times()
    mem_usage = end_mem - start_mem
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()
    
    del embeddings, metadata, chunk
    gc.collect()
    return {"pid": pid, "start_time": start_time, "end_time": end_time, 
            "mem_usage": mem_usage, "cpu_percentage": cpu_percentage, 
            "memory_usage_percentage": memory_usage_percentage, "ids": ids}

def query_chunk(chunk_number):
    global collection, embedding_function, document_chunks
    chunk = document_chunks[chunk_number]
    pid = os.getpid()
    n_results = 5
    print(f"Querying chunk in PID {pid}")
    process = psutil.Process(pid)
    total_cores = psutil.cpu_count(logical=False)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    embeddings = embedding_function(chunk)
    query_result = collection.query(query_embeddings=embeddings, n_results=n_results, include=["documents", "distances"])
    
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 ** 2)
    cpu_end = process.cpu_times()
    mem_usage = end_mem - start_mem
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()

    num_queries = len(chunk)
    if num_queries > 0 and query_result.get('distances'):
        distance_sums = [0.0] * n_results
        actual_results_count = len(query_result['distances'])

        if actual_results_count > 0:
            valid_distance_lists = 0
            for distances_for_one_query in query_result['distances']:
                if len(distances_for_one_query) == n_results:
                    valid_distance_lists += 1
                    for i in range(n_results):
                        if isinstance(distances_for_one_query[i], (int, float)):
                             distance_sums[i] += distances_for_one_query[i]

            if valid_distance_lists > 0:
                 avg_distances = [dist_sum / valid_distance_lists for dist_sum in distance_sums]
    
    del embeddings, chunk, query_result
    gc.collect()
    return {"pid": pid, "start_time": start_time, "end_time": end_time, 
            "mem_usage": mem_usage, "cpu_percentage": cpu_percentage, 
            "memory_usage_percentage": memory_usage_percentage, "avg_distances": avg_distances, "num_documents": num_queries}

def update_chunk(chunk_number):
    global collection, embedding_function, id_chunks, shuffled_document_chunks
    new_chunk = shuffled_document_chunks[chunk_number]
    id_chunk = id_chunks[chunk_number]
    new_metadata = [{"document": doc} for doc in new_chunk]
    pid = os.getpid()
    print(f"Updating chunk in PID {pid}")
    process = psutil.Process(pid)
    total_cores = psutil.cpu_count(logical=False)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    new_embeddings = embedding_function(new_chunk)
    collection.update(ids=id_chunk, documents=new_chunk,
                        embeddings=new_embeddings, metadatas=new_metadata)
    
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 ** 2)
    cpu_end = process.cpu_times()
    mem_usage = end_mem - start_mem
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()
    
    del new_chunk, new_embeddings, new_metadata, id_chunk
    gc.collect()
    return {"pid": pid, "start_time": start_time, "end_time": end_time, 
            "mem_usage": mem_usage, "cpu_percentage": cpu_percentage, 
            "memory_usage_percentage": memory_usage_percentage}

def delete_chunk(chunk_number):
    global collection, embedding_function, id_chunks
    pid = os.getpid()
    print(f"Deleting chunk in PID {pid}")
    id_chunk = id_chunks[chunk_number]
    process = psutil.Process(pid)
    total_cores = psutil.cpu_count(logical=False)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    collection.delete(ids=id_chunk)
    
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 ** 2)
    cpu_end = process.cpu_times()
    mem_usage = end_mem - start_mem
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()
    
    del id_chunk
    gc.collect()

    return {"pid": pid, "start_time": start_time, "end_time": end_time, 
            "mem_usage": mem_usage, "cpu_percentage": cpu_percentage, 
            "memory_usage_percentage": memory_usage_percentage}

def parallel_operation(operation, num_operations, num_workers):
    global all_ids
    chunk_indices = range(num_workers)
    with multiprocessing.Pool(processes=num_workers) as pool:
        if operation == "insert":
            results = pool.map(insert_chunk, chunk_indices)
            for result in results:
                all_ids.extend(result["ids"])
        elif operation == "query":
            results = pool.map(query_chunk, chunk_indices)
            n_results = len(results[0]["avg_distances"])
            final_distance_average = [0.0] * n_results
            for result in results:
                avg_distances = result["avg_distances"]
                num_documents = result["num_documents"]
                if avg_distances:
                    for i in range(n_results):
                        if isinstance(avg_distances[i], (int, float)):
                            final_distance_average[i] += avg_distances[i] * num_documents
            for i in range(n_results):
                final_distance_average[i] /= num_operations
            print("\nAverage distances for each query:", final_distance_average)
                
        elif operation == "update":
            results = pool.map(update_chunk, chunk_indices)
        elif operation == "delete":
            results = pool.map(delete_chunk, chunk_indices)
        else:
            raise ValueError("Invalid operation. Choose from: insert, query, update, delete.")

    start_time = min(result["start_time"] for result in results)
    end_time = max(result["end_time"] for result in results)
    mem_usage = sum(result["mem_usage"] for result in results)
    cpu_percentage = sum(result["cpu_percentage"] for result in results)
    memory_usage_percentage = sum(result["memory_usage_percentage"] for result in results)
    print(f"{operation.capitalize()} Performance:")
    display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage, num_operations)

file_path = "./ip_flow_dataset.csv"
documents = read_documents_from_file(file_path)
num_operations = len(documents)

num_cores = psutil.cpu_count(logical = False)-2
num_workers = min(num_cores, num_operations)
document_chunks = split_documents(documents, num_workers)
shuffled_document_chunks = [random.sample(chunk, len(chunk)) for chunk in document_chunks]

parallel_operation("insert", num_operations, num_workers)

count = collection.count()
print("\nNumber of records in collection after insert:", count, "\n\n")

parallel_operation("query", num_operations, num_workers)

count = collection.count()
print("\nNumber of records in collection after query:", count, "\n\n")

id_chunks = split_documents(all_ids, num_workers)

parallel_operation("update", num_operations, num_workers)

count = collection.count()
print("\nNumber of records in collection after update:", count, "\n\n")

parallel_operation("delete", num_operations, num_workers)

count = collection.count()
print("\nNumber of records in collection after delete:", count, "\n\n")