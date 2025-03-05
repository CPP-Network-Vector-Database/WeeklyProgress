import logging
import time
import psutil
import tracemalloc
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction

logging.getLogger("chromadb").setLevel(logging.ERROR)

embedding_model = SentenceTransformer("all-mpnet-base-v2")

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts).tolist()

client = chromadb.PersistentClient(path="./chromadb")

embedding_function = SentenceTransformerEmbeddingFunction()
collection = client.get_or_create_collection(name="mpnet", embedding_function=embedding_function)

def query_collection():
    print("\n--- ChromaDB Query CLI ---")
    process = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)  
    
    while True:
        query_text = input("Enter your query (or 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break
        try:
            n_results = int(input("Enter the number of results to return: ").strip())
            if n_results <= 0:
                print("Please enter a valid positive number.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        
        start_time = time.time()
        tracemalloc.start()
        cpu_start = process.cpu_times()

        
        results = collection.query(
            query_embeddings=embedding_function([query_text]), 
            n_results=n_results, 
            include=["documents", "distances"]
        )

        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_end = process.cpu_times()

        
        cpu_usage = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        cpu_percentage = ((cpu_usage / (end_time - start_time)) * 100) / total_cores

        
        memory_usage = process.memory_percent()

        
        print("\nResults:")
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            print(f"Document: {doc} | Distance: {dist:.4f}")
        
        print("\n--- Performance Metrics ---")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print(f"Memory usage: {current / 1024 / 1024:.4f} MB (Current), {peak / 1024 / 1024:.4f} MB (Peak)")
        print(f"CPU time usage: {cpu_percentage:.2f}%")
        print(f"Memory usage (system): {memory_usage:.2f}%")
        print("-------------------------")

if __name__ == '__main__':
    query_collection()
