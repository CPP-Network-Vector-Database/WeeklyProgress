import chromadb
import time
import gc
import psutil
import tracemalloc
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import uuid
import os

embedding_model = SentenceTransformer("all-distilroberta-v1", device="cpu")
embedding_model.max_seq_length = 128

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts, num_workers=psutil.cpu_count(logical=True)).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="distilroberta_l2_128", embedding_function=SentenceTransformerEmbeddingFunction(), metadata={"hnsw:space": "l2"})

def display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage, operations):    
    print("\n--- Performance Metrics ---")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Memory usage: {mem_usage:.2f} MB")
    print(f"CPU time usage: {cpu_percentage:.2f}%")
    print(f"Memory usage (system): {memory_usage_percentage:.2f}%")
    print(f"Throughput: {operations / (end_time - start_time):.2f} operations/sec")
    print("-------------------------")

def insert_record():
    source_ip = input("Enter Source IP: ").strip()
    destination_ip = input("Enter Destination IP: ").strip()
    source_port = input("Enter Source Port: ").strip()
    destination_port = input("Enter Destination Port: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source_ip}, {destination_ip}, {source_port}, {destination_port},  {protocol}"
    doc_id = str(uuid.uuid4())

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    collection.add(
        documents=[document], 
        ids=[doc_id], 
        embeddings=embedding_function([document]),
        metadatas=[{"source_ip": source_ip, "destination_ip": destination_ip, "source_port": source_port, "destination_port": destination_port, "protocol": protocol}]
    )

    end_time = time.perf_counter()

    
    end_mem = process.memory_info().rss / (1024 ** 2)
    mem_usage = end_mem - start_mem
    cpu_end = process.cpu_times()
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()

    display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage, 1)

    print("Record inserted successfully.")

def delete_record():
    source_ip = input("Enter Source IP: ").strip()
    destination_ip = input("Enter Destination IP: ").strip()
    source_port = input("Enter Source Port: ").strip()
    destination_port = input("Enter Destination Port: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source_ip}, {destination_ip}, {source_port}, {destination_port},  {protocol}"
    flag = False

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)

    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()

    start_time = time.perf_counter()
    
    results = collection.get(where={"$and": [
        {"source_ip": {"$eq": source_ip}},
        {"destination": {"$eq": destination_ip}},
        {"source_port": {"$eq": source_port}},
        {"destination_port": {"$eq": destination_port}},
        {"protocol": {"$eq": protocol}}
    ]})

    if results and "ids" in results and results["ids"]:
        collection.delete(ids=results["ids"])
        flag = True
    

    end_time = time.perf_counter()

    
    end_mem = process.memory_info().rss / (1024 ** 2)
    mem_usage = end_mem - start_mem
    cpu_end = process.cpu_times()
    
    if flag:
        print("Record deleted successfully.")
    else:
        print("No matching record found.")
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()

    display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage,1)


def update_record():
    source_ip = input("Enter Source IP: ").strip()
    destination_ip = input("Enter Destination IP: ").strip()
    source_port = input("Enter Source Port: ").strip()
    destination_port = input("Enter Destination Port: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source_ip}, {destination_ip}, {source_port}, {destination_port},  {protocol}"

    new_source_ip = input("Enter Source IP: ").strip()
    new_destination_ip = input("Enter Destination IP: ").strip()
    new_source_port = input("Enter Source Port: ").strip()
    new_destination_port = input("Enter Destination Port: ").strip()
    new_protocol = input("Enter Protocol: ").strip()
    new_document = f"{new_source_ip}, {new_destination_ip}, {new_source_port}, {new_destination_port},  {new_protocol}"

    flag = False

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()

    results = collection.get(where={"$and": [
        {"source_ip": {"$eq": source_ip}},
        {"destination_ip": {"$eq": destination_ip}},
        {"source_port": {"$eq": source_port}},
        {"destination_port": {"$eq": destination_port}},
        {"protocol": {"$eq": protocol}}
    ]})

    if results["ids"]:
        doc_id = results["ids"][0]  # Get document ID
        collection.update(
            ids=[doc_id], 
            documents=[new_document], 
            embeddings=embedding_function([new_document]),
            metadatas=[{"source_ip": new_source_ip, "destination_ip": new_destination_ip , "source_port": new_source_port, "destination_port": new_destination_port,"protocol": new_protocol}]
        )
        flag = True
    

    end_time = time.perf_counter()

    
    end_mem = process.memory_info().rss / (1024 ** 2)
    mem_usage = end_mem - start_mem
    cpu_end = process.cpu_times()

    if flag:
        print("Record updated successfully.")
    else:
        print("No matching record found.")

    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()

    display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage, 1)

def query_record():
    source_ip = input("Enter Source IP: ").strip()
    destination_ip = input("Enter Destination IP: ").strip()
    source_port = input("Enter Source Port: ").strip()
    destination_port = input("Enter Destination Port: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source_ip}, {destination_ip}, {source_port}, {destination_port},  {protocol}"

    try:
        n_results = int(input("Enter the number of results to return: ").strip())
        if n_results <= 0:
            print("Please enter a valid positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)
    start_mem = process.memory_info().rss / (1024 ** 2)
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    results = collection.query(
            query_embeddings=embedding_function([document]), 
            n_results=n_results, 
            include=["documents", "distances"]
        )

    end_time = time.perf_counter()

    
    end_mem = process.memory_info().rss / (1024 ** 2)
    mem_usage = end_mem - start_mem
    cpu_end = process.cpu_times()

    print("\nResults:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"Document: {doc} | Distance: {dist:.4f}")
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage_percentage = process.memory_percent()

    display_performance(start_time, end_time, mem_usage, memory_usage_percentage, cpu_percentage, 1)

def menu():
    while True:
        print("\n--- ChromaDB CRUD Operations ---")
        print("1. Insert Record")
        print("2. Query Record")
        print("3. Update Record")
        print("4. Delete Record")
        print("5. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            insert_record()
        elif choice == '2':
            query_record()
        elif choice == '3':
            update_record()
        elif choice == '4':
            delete_record()
        elif choice == '5':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()
