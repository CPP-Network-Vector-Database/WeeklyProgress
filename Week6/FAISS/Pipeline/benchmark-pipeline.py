import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

class FAISSPerformanceTracker:
    def __init__(self, model_name = 'all-mpnet-base-v2'): # sentence model to be passed can be changed 
        self.model = SentenceTransformer(model_name) # load the model
        
        # Performance tracking lists- to keep track of time, memory, and cpu usage as the number of insertions/deletions/updates change
        self.insertion_sizes = []
        self.deletion_sizes = []
        self.update_sizes = []
        
        self.insertion_times = []
        self.deletion_times = []
        self.update_times = []
        
        self.insertion_cpu = []
        self.deletion_cpu = []
        self.update_cpu = []
        
        self.insertion_memory = []
        self.deletion_memory = []
        self.update_memory = []

    def _measure_performance(self, func, *args, **kwargs): # this returns the resultof the function
        pid = os.getpid()
        process = psutil.Process(pid)
        
        # Initial resource usage
        start_cpu = process.cpu_percent(interval=None)
        start_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        start_time = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Final resource usage
        end_cpu = process.cpu_percent(interval=None)
        end_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        end_time = time.time()
        
        # Compute differences
        cpu_usage = end_cpu - start_cpu
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage # same fucntion as was beingused before

    def track_insertion(self, index, new_packet_texts):
        # Normalize embeddings because all sentenceTransformer based models need normalization
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        # Track metrics by appending it to the previously initialized lists
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, index, num_deletions):
        # Select random indices to delete (the number is passed by us as an argument)
        delete_indices = np.random.choice(index.ntotal, num_deletions, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, index, delete_indices
        )
        
        # Track metrics
        self.deletion_sizes.append(num_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)

    def track_update(self, index, num_updates, new_packet_texts):

        # Select random indices to update
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_packet_texts
        )
        
        # Track metrics
        self.update_sizes.append(num_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage) # same as before

    def _insert_embeddings(self, index, new_embeddings):
        index.add(new_embeddings)
        return index

    def _delete_embeddings(self, index, delete_indices):
        index.remove_ids(delete_indices)
        return index

    def _update_embeddings(self, index, update_indices, new_packet_texts):
        index.remove_ids(update_indices)
        
        # Compute and normalize new embeddings
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Add new embeddings
        index.add(new_embeddings)
        return index

    def plot_performance_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # Time Plots
        plt.subplot(1, 3, 1)
        plt.plot(self.insertion_sizes, self.insertion_times, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_times, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_times, label='Update', marker='^')
        plt.title('Execution Time')
        plt.xlabel('Number of Embeddings')
        plt.ylabel('Time (seconds)')
        plt.legend()
        
        # CPU Usage Plots
        plt.subplot(1, 3, 2)
        plt.plot(self.insertion_sizes, self.insertion_cpu, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_cpu, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_cpu, label='Update', marker='^')
        plt.title('CPU Usage')
        plt.xlabel('Number of Embeddings')
        plt.ylabel('CPU Percentage')
        plt.legend()
        
        # Memory Usage Plots
        plt.subplot(1, 3, 3)
        plt.plot(self.insertion_sizes, self.insertion_memory, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_memory, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_memory, label='Update', marker='^')
        plt.title('Memory Usage')
        plt.xlabel('Number of Embeddings')
        plt.ylabel('Memory (MB)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Read CSV 
    csv_path = "/home/pes1ug22am100/Documents/WeeklyProgress/Week6/FAISS/dirA.125910-packets.csv"
    df = pd.read_csv(csv_path, header=None, names=["timestamp", "src_ip", "dst_ip", "protocol", "size"]) # can use nrows = x to select a few rows only

    # Packet data to text format for BERT processing
    df["packet_text"] = df["src_ip"] + " " + df["dst_ip"] + " " + df["protocol"] + " " + df["size"].astype(str)

    # Generate embeddings
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(df["packet_text"].tolist(), convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Creating FAISS IVFFlat index
    dimension = embeddings.shape[1]
    nlist = min(100, int(np.sqrt(len(embeddings))))
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)

    # Initialize performance tracker
    tracker = FAISSPerformanceTracker()

    # Track performance for various operations and get the graph at the end of it
    for num_ops in [50, 100, 250, 500, 1000]:
        # Insertion tracking
        new_packet_texts = [f"192.168.1.{i} 192.168.1.{i+1} TCP {i*10}" for i in range(num_ops)]
        tracker.track_insertion(index, new_packet_texts)

        # Deletion tracking
        tracker.track_deletion(index, num_ops)

        # Update tracking
        update_texts = [f"10.0.0.{i} 10.0.0.{i+1} UDP {i*5}" for i in range(num_ops)]
        tracker.track_update(index, num_ops, update_texts)

    tracker.plot_performance_metrics()

if __name__ == "__main__":
    main()