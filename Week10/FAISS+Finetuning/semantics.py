import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

class FAISSPerformanceTracker:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Performance tracking lists
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

    def track_insertion(self, index, new_packet_texts):
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, index, num_deletions):
        delete_indices = np.random.choice(index.ntotal, num_deletions, replace=False)
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, index, delete_indices
        )
        
        self.deletion_sizes.append(num_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)

    def track_update(self, index, num_updates, new_packet_texts):
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_packet_texts
        )
        
        self.update_sizes.append(num_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage)

    def _insert_embeddings(self, index, new_embeddings):
        index.add(new_embeddings)
        return index

    def _delete_embeddings(self, index, delete_indices):
        index.remove_ids(delete_indices)
        return index

    def _update_embeddings(self, index, update_indices, new_packet_texts):
        index.remove_ids(update_indices)
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        index.add(new_embeddings)
        return index

    def plot_performance_metrics(self, save_path=None):
        plt.figure(figsize=(15, 5))
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()

    def query_top_k(self, index, query_texts, k=5):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        distances, indices = index.search(query_embeddings, k)
        return distances, indices
    
    def track_query(self, index, query_texts, k=5):
        num_queries = len(query_texts)
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_texts, k
        )
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)

    def _query_embeddings(self, index, query_texts, k):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        index.nprobe = 10
        return index.search(query_embeddings, k)

def create_semantic_packet_text(row):
    src = f"source IP {row['ip.src']}" if pd.notna(row['ip.src']) else "an unknown source"
    dst = f"destination IP {row['ip.dst']}" if pd.notna(row['ip.dst']) else "an unknown destination"
    protocol = f"using {row['_ws.col.protocol']} protocol" if pd.notna(row['_ws.col.protocol']) else ""
    src_port = f"from port {row['tcp.srcport']}" if pd.notna(row['tcp.srcport']) else ""
    dst_port = f"to port {row['tcp.dstport']}" if pd.notna(row['tcp.dstport']) else ""
    length = f"with packet size {row['frame.len']} bytes" if pd.notna(row['frame.len']) else ""
    
    return f"A network packet from {src} {src_port} to {dst} {dst_port} {protocol} {length}"

def generate_semantic_queries():
    return [
        "Find all TCP packets",
        "Show me packets from source IP 192.168.1.1",
        "Find large packets over 1000 bytes",
        "Show me UDP traffic to port 53",
        "Find all HTTPS packets",
        "Show me packets between 192.168.1.1 and 192.168.1.2",
        "Find small packets under 100 bytes"
    ]

def main(): 
    csv_path = "/Week8-9/FAISS/ip_flow_dataset.csv" 
    
    df = pd.read_csv(
        csv_path,
        header=0,
        names=[
            "frame.number", "frame.time", "ip.src", "ip.dst",
            "tcp.srcport", "tcp.dstport", "_ws.col.protocol", "frame.len"
        ],
        dtype=str,
        skiprows=1
    )

    # Create semantic packet descriptions
    df["packet_text"] = df.apply(create_semantic_packet_text, axis=1)

    modelList = [
        'paraphrase-MiniLM-L12-v2', 
        "all-MiniLM-L6-v2", 
        'distilbert-base-nli-stsb-mean-tokens', 
        'microsoft/codebert-base', 
        'bert-base-nli-mean-tokens', 
        'sentence-transformers/average_word_embeddings_komninos', 
        'all-mpnet-base-v2'
    ]
    
    plots_dir = "PipelineResults"
    os.makedirs(plots_dir, exist_ok=True)
    
    for mod in modelList:
        print(f"\nProcessing model: {mod}")
        
        clean_model_name = mod.replace('/', '_').replace('-', '_')
        model = SentenceTransformer(mod)
        embeddings = model.encode(df["packet_text"].tolist(), convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        dimension = embeddings.shape[1]
        nlist = min(100, int(np.sqrt(len(embeddings))))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        
        tracker = FAISSPerformanceTracker(mod)
        
        for num_ops in [2500, 5000, 7500, 10000, 20000, 30000]:
            print(f"  Running {num_ops} operations...")
            
            # Generate semantic test data
            new_packet_texts = [
                f"A network packet from source IP 192.168.1.{i} to destination IP 192.168.1.{i+1} using TCP protocol from port {i*10}"
                for i in range(num_ops)
            ]
            
            tracker.track_insertion(index, new_packet_texts)
            tracker.track_deletion(index, num_ops)
            
            update_texts = [
                f"A network packet from source IP 10.0.0.{i} to destination IP 10.0.0.{i+1} using UDP protocol with packet size {i*5} bytes"
                for i in range(num_ops)
            ]
            tracker.track_update(index, num_ops, update_texts)
            
            # Test semantic queries
            query_texts = generate_semantic_queries()[:min(5, num_ops)]
            tracker.track_query(index, query_texts, k=5)
            
            if num_ops == 1000:
                test_query = ["Find all packets from source IP 192.168.1.1 to destination IP 192.168.1.2"]
                distances, indices = tracker.query_top_k(index, test_query, k=5)
                print("  Top 5 Neighbors (indices):", indices[0])
                print("  Cosine Similarities:", distances[0])
        
        plot_filename = f"{plots_dir}/{clean_model_name}.png"
        tracker.plot_performance_metrics(save_path=plot_filename)

if __name__ == "__main__":
    main()