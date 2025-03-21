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
        return embedding_model.encode(texts, num_workers=1).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="distilroberta", embedding_function=SentenceTransformerEmbeddingFunction())

def display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage, operations):    
    print("\n--- Performance Metrics ---")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Memory usage: {current / 1024 / 1024:.4f} MB (Current), {peak / 1024 / 1024:.4f} MB (Peak)")
    print(f"CPU time usage: {cpu_percentage:.2f}%")
    print(f"Memory usage (system): {memory_usage:.2f}%")
    print(f"Throughput: {operations / (end_time - start_time):.2f} operations/sec")
    print("-------------------------")

def insert_record():
    source = input("Enter Source: ").strip()
    destination = input("Enter Destination: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source}, {destination}, {protocol}"
    doc_id = str(uuid.uuid4())

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)
    tracemalloc.start()
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    collection.add(
        documents=[document], 
        ids=[doc_id], 
        embeddings=embedding_function([document]),
        metadatas=[{"source": source, "destination": destination, "protocol": protocol}]
    )

    end_time = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_end = process.cpu_times()
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage = process.memory_percent()

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage, 1)

    print("Record inserted successfully.")

def delete_record():
    source = input("Enter Source: ").strip()
    destination = input("Enter Destination: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source}, {destination}, {protocol}"
    flag = False

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)

    tracemalloc.start()
    cpu_start = process.cpu_times()

    start_time = time.perf_counter()
    
    results = collection.get(where={"$and": [
        {"source": {"$eq": source}},
        {"destination": {"$eq": destination}},
        {"protocol": {"$eq": protocol}}
    ]})

    if results and "ids" in results and results["ids"]:
        collection.delete(ids=results["ids"])
        flag = True
    

    end_time = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_end = process.cpu_times()
    
    if flag:
        print("Record deleted successfully.")
    else:
        print("No matching record found.")
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage = process.memory_percent()

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage,1)


def update_record():
    source = input("Enter Source: ").strip()
    destination = input("Enter Destination: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source}, {destination}, {protocol}"

    new_source = input("Enter new Source: ").strip()
    new_destination = input("Enter new Destination: ").strip()
    new_protocol = input("Enter new Protocol: ").strip()
    new_document = f"{new_source}, {new_destination}, {new_protocol}"

    flag = False

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)
    tracemalloc.start()
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()

    results = collection.get(where={"$and": [
        {"source": {"$eq": source}},
        {"destination": {"$eq": destination}},
        {"protocol": {"$eq": protocol}}
    ]})

    if results["ids"]:
        doc_id = results["ids"][0]  # Get document ID
        collection.update(
            ids=[doc_id], 
            documents=[new_document], 
            embeddings=embedding_function([new_document]),
            metadatas=[{"source": new_source, "destination": new_destination, "protocol": new_protocol}]
        )
        flag = True
    

    end_time = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_end = process.cpu_times()

    if flag:
        print("Record updated successfully.")
    else:
        print("No matching record found.")

    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage = process.memory_percent()

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage, 1)

def query_record():
    source = input("Enter Source: ").strip()
    destination = input("Enter Destination: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source}, {destination}, {protocol}"

    try:
        n_results = int(input("Enter the number of results to return: ").strip())
        if n_results <= 0:
            print("Please enter a valid positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

    process = psutil.Process(os.getpid())
    total_cores = psutil.cpu_count(logical=True)
    tracemalloc.start()
    cpu_start = process.cpu_times()
    start_time = time.perf_counter()
    
    results = collection.query(
            query_embeddings=embedding_function([document]), 
            n_results=n_results, 
            include=["documents", "distances"]
        )

    end_time = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_end = process.cpu_times()

    print("\nResults:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"Document: {doc} | Distance: {dist:.4f}")
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage = process.memory_percent()

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage, 1)

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
