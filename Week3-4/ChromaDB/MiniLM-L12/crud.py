import chromadb
import time
import gc
import psutil
import tracemalloc
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import uuid

embedding_model = SentenceTransformer("all-MiniLM-L12-v2", device="cpu")
embedding_model.max_seq_length = 128

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts, num_workers=1).tolist()

embedding_function = SentenceTransformerEmbeddingFunction()

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="miniLM_L12", embedding_function=SentenceTransformerEmbeddingFunction())

def display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage):    
    print("\n--- Performance Metrics ---")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Memory usage: {current / 1024 / 1024:.4f} MB (Current), {peak / 1024 / 1024:.4f} MB (Peak)")
    print(f"CPU time usage: {cpu_percentage:.2f}%")
    print(f"Memory usage (system): {memory_usage:.2f}%")
    print("-------------------------")

def insert_record():
    source = input("Enter Source: ").strip()
    destination = input("Enter Destination: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source}, {destination}, {protocol}"

    process = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)
    start_time = time.time()
    tracemalloc.start()
    cpu_start = process.cpu_times()
    
    collection.add(documents=[document], ids=[str(uuid.uuid4())], embeddings=embedding_function([document]))
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_end = process.cpu_times()
    
    cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores
    memory_usage = process.memory_percent()

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage)

    print("Record inserted successfully.")

def delete_record():
    source = input("Enter Source: ").strip()
    destination = input("Enter Destination: ").strip()
    protocol = input("Enter Protocol: ").strip()
    document = f"{source}, {destination}, {protocol}"
    flag = False

    process = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)
    start_time = time.time()
    tracemalloc.start()
    cpu_start = process.cpu_times()
    
    # Query the collection to find a matching document
    results = collection.query(
        query_embeddings=embedding_function([document]),
        n_results=1,
        include=["documents", "distances"]  # Remove "ids" since it's not supported
    )

    if results["documents"]:
        retrieved_doc = results["documents"][0][0]  # Get the first matching document
        
        if retrieved_doc == document:  # Ensure exact match
            matching_docs = collection.get()  # Get all stored records
            
            for idx, doc in enumerate(matching_docs["documents"]):
                if doc == retrieved_doc:
                    doc_id = matching_docs["ids"][idx]  # Get corresponding ID
                    collection.delete(ids=[doc_id])
                    flag = True
                    break

    end_time = time.time()
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

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage)


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

    process = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)
    start_time = time.time()
    tracemalloc.start()
    cpu_start = process.cpu_times()

    # Query to find the document using embeddings
    results = collection.query(
        query_embeddings=embedding_function([document]),
        n_results=1,
        include=["documents", "distances"]  # Removed "ids"
    )

    if results["documents"]:
        retrieved_doc = results["documents"][0][0]  # Get the first matching document
        
        if retrieved_doc == document:  # Ensure exact match
            matching_docs = collection.get()  # Get all stored records
            
            for idx, doc in enumerate(matching_docs["documents"]):
                if doc == retrieved_doc:
                    doc_id = matching_docs["ids"][idx]  # Get corresponding ID
                    collection.update(ids=[doc_id], documents=[new_document], embeddings=embedding_function([new_document]))
                    flag = True
                    break

    end_time = time.time()
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

    display_performance(start_time, end_time, current, peak, memory_usage, cpu_percentage)


def menu():
    while True:
        print("\n--- ChromaDB CRUD Operations ---")
        print("1. Insert Record")
        print("2. Delete Record")
        print("3. Update Record")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            insert_record()
        elif choice == '2':
            delete_record()
        elif choice == '3':
            update_record()
        elif choice == '4':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()
