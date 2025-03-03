import time
import psutil
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Benchmark function
def benchmark_query(query_embedding):
    """
    Perform a query and measure performance metrics.
    :param query_embedding: Embedding of the query.
    :return: Latency, memory usage, CPU usage, and query response.
    """
    start_time = time.time()
    try:
        response = client.search(collection_name="my_collection", query_vector=query_embedding, limit=5)
    except Exception as e:
        print(f"Error during query: {e}")
        return None

    latency = time.time() - start_time

    # Measure system-wide memory and CPU usage
    memory_usage = psutil.virtual_memory().used / 1024 / 1024  # in MB
    cpu_usage = psutil.cpu_percent(interval=1.0)

    return latency, memory_usage, cpu_usage, response

# CLI-based UI
def cli_ui():
    """
    Command-line interface for querying the Qdrant database.
    """
    print("Qdrant Query CLI")
    print("Type your query and press Enter. Type 'exit' to quit.")
    query_count = 0
    total_latency = 0
    start_time = time.time()

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        # Generate query embedding
        try:
            query_embedding = model.encode([query])[0]
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            continue

        # Benchmark and query
        result = benchmark_query(query_embedding)
        if result is None:
            continue  # Skip if there was an error

        latency, memory, cpu, response = result

        # Update metrics
        query_count += 1
        total_latency += latency

        # Display results
        print("\n--- Results ---")
        print(f"Query: {query}")
        print(f"Latency: {latency:.4f}s, Memory: {memory:.2f}MB, CPU: {cpu:.2f}%")
        print("Top 5 results:")
        for idx, result in enumerate(response, start=1):
            print(f"\nResult {idx}:")
            print(f"  ID: {result.id}")
            print(f"  Score: {result.score:.4f}")
            if 'text' in result.payload:
                print(f"  Text: {result.payload['text'][:200]}...")  # Show first 200 characters
            else:
                print("  Text: [Payload does not contain 'text' field]")

        # Display throughput
        if query_count > 0:
            elapsed_time = time.time() - start_time
            throughput = query_count / elapsed_time
            print(f"\nThroughput: {throughput:.2f} queries/sec")

if __name__ == "__main__":
    cli_ui()
