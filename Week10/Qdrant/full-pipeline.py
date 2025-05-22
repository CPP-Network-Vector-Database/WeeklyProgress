import pandas as pd
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantPerformanceTracker:
    def __init__(self, model_name='all-mpnet-base-v2', collection_name="packet_collection"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.client = QdrantClient(":memory:")  # In-memory storage for testing
        self.collection_name = collection_name
        self.next_id = 0
        self.inserted_ids = []

        # Initialize performance tracking lists
        self.insertion_sizes = []
        self.deletion_sizes = []
        self.update_sizes = []
        self.query_sizes = []
        
        self.insertion_times = []
        self.deletion_times = []
        self.update_times = []
        self.query_times = []
        
        self.insertion_cpu = []
        self.deletion_cpu = []
        self.update_cpu = []
        self.query_cpu = []
        
        self.insertion_memory = []
        self.deletion_memory = []
        self.update_memory = []
        self.query_memory = []

    def _measure_performance(self, func, *args, **kwargs):
        pid = os.getpid()
        process = psutil.Process(pid)
        
        start_cpu = process.cpu_percent(interval=None)
        start_mem = process.memory_info().rss / (1024 ** 2)
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_cpu = process.cpu_percent(interval=None)
        end_mem = process.memory_info().rss / (1024 ** 2)
        end_time = time.time()
        
        cpu_usage = end_cpu - start_cpu
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage

    def create_collection(self, vector_size, distance=models.Distance.COSINE):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance
            )
        )

    def track_insertion(self, new_packet_texts):
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        ids = list(range(self.next_id, self.next_id + len(new_embeddings)))
        self.next_id += len(new_embeddings)
        self.inserted_ids.extend(ids)
        
        points = [
            models.PointStruct(
                id=id,
                vector=vector.tolist(),
                payload={"text": text}
            ) for id, vector, text in zip(ids, new_embeddings, new_packet_texts)
        ]
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self.client.upsert, 
            collection_name=self.collection_name,
            points=points
        )
        
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, num_deletions):
        if num_deletions > len(self.inserted_ids):
            num_deletions = len(self.inserted_ids)
        delete_ids = self.inserted_ids[:num_deletions]
        del self.inserted_ids[:num_deletions]
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self.client.delete,
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=delete_ids
            )
        )
        
        self.deletion_sizes.append(num_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)

    def track_update(self, num_updates, new_packet_texts):
        if num_updates > len(self.inserted_ids):
            num_updates = len(self.inserted_ids)
        update_ids = self.inserted_ids[:num_updates]
        del self.inserted_ids[:num_updates]
        
        # Delete old points
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=update_ids)
        )
        
        # Insert new points
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        new_ids = list(range(self.next_id, self.next_id + num_updates))
        self.next_id += num_updates
        self.inserted_ids.extend(new_ids)
        
        points = [
            models.PointStruct(
                id=id,
                vector=vec.tolist(),
                payload={"text": text}
            ) for id, vec, text in zip(new_ids, new_embeddings, new_packet_texts)
        ]
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points
        )
        
        self.update_sizes.append(num_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage)

    def track_query(self, query_texts, k=5):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        total_time = 0
        total_cpu = 0
        total_mem = 0
        
        for query in query_embeddings:
            _, time_taken, cpu_usage, mem_usage = self._measure_performance(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query.tolist(),
                limit=k
            )
            total_time += time_taken
            total_cpu += cpu_usage
            total_mem += mem_usage
        
        avg_time = total_time / len(query_embeddings)
        avg_cpu = total_cpu / len(query_embeddings)
        avg_mem = total_mem / len(query_embeddings)
        
        self.query_sizes.append(len(query_embeddings))
        self.query_times.append(avg_time)
        self.query_cpu.append(avg_cpu)
        self.query_memory.append(avg_mem)

    def plot_performance_metrics(self, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # Time Plots
        plt.subplot(1, 3, 1)
        plt.plot(self.insertion_sizes, self.insertion_times, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_times, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_times, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_times, label='Query', marker='x')
        plt.title('Execution Time')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)

        # CPU Usage Plots
        plt.subplot(1, 3, 2)
        plt.plot(self.insertion_sizes, self.insertion_cpu, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_cpu, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_cpu, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_cpu, label='Query', marker='x')
        plt.title('CPU Usage')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('CPU Percentage')
        plt.legend()
        plt.grid(True)

        # Memory Usage Plots
        plt.subplot(1, 3, 3)
        plt.plot(self.insertion_sizes, self.insertion_memory, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_memory, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_memory, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_memory, label='Query', marker='x')
        plt.title('Memory Usage')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Performance Metrics for {self.model_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()

def main():
    # ========== REDUCED PARAMETERS FOR QUICK TEST ==========
    csv_path = "ip_flow_dataset.csv"  # Update this path
    modelList = ['all-MiniLM-L6-v2']  # Single fast model
    test_batches = [100, 500]         # Reduced batch sizes
    test_queries = 10                 # Limited queries
    initial_samples = 100             # Initial dataset size
    # ======================================================
    
    # Load minimal data
    df = pd.read_csv(csv_path, nrows=1000)  # Limit to first 1000 rows
    
    # Create packet text (simplified)
    df["packet_text"] = df.apply(lambda row: " ".join(str(x) for x in row), axis=1)

    plots_dir = "QdrantQuickTestResults"
    os.makedirs(plots_dir, exist_ok=True)

    for mod in modelList:
        print(f"\nTesting model: {mod}")
        clean_name = mod.replace('/', '_').replace('-', '_')
        
        # Initialize tracker and model
        tracker = QdrantPerformanceTracker(mod)
        model = SentenceTransformer(mod)
        
        # Create initial collection with tiny dataset
        initial_texts = df["packet_text"].head(initial_samples).tolist()
        initial_embeddings = model.encode(initial_texts)
        tracker.create_collection(vector_size=initial_embeddings.shape[1])
        tracker.track_insertion(initial_texts)
        
        # Run reduced test batches
        for num_ops in test_batches:
            print(f"  Processing {num_ops} operations...")
            
            # Insert test data
            new_texts = [f"TEST_IP_{i}" for i in range(num_ops)]
            tracker.track_insertion(new_texts)
            
            # Delete half
            tracker.track_deletion(num_ops//2)
            
            # Update quarter
            update_texts = [f"UPDATE_{i}" for i in range(num_ops//4)]
            tracker.track_update(num_ops//4, update_texts)
            
            # Query with sample texts
            query_texts = ["TEST_QUERY_1", "TEST_QUERY_2"][:test_queries]
            tracker.track_query(query_texts)
        
        # Save graphs
        plot_path = f"{plots_dir}/{clean_name}_quick_test.png"
        tracker.plot_performance_metrics(save_path=plot_path)

if __name__ == "__main__":
    main()
