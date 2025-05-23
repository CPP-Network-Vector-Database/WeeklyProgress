import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetModel
from transformers import BertModel, BertTokenizer

class CustomEmbeddings:
    def __init__(self, data, categorical_cols, continuous_cols):
        self.data = data
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._preprocess_data()
        
    def _preprocess_data(self):
        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
            
        # Scale continuous columns
        if self.continuous_cols:
            self.data[self.continuous_cols] = self.scaler.fit_transform(self.data[self.continuous_cols])
    
    def get_mlp_embeddings(self, embedding_dim=128): # embeddings with a single MLP
        # Prepare data
        X_cat = self.data[self.categorical_cols].values
        X_cont = self.data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.data), 0))
        X = np.concatenate([X_cat, X_cont], axis=1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Define MLP model
        input_dim = X.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Generate embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = model(inputs)
                embeddings.append(outputs.numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def get_tabtransformer_embeddings(self, embedding_dim=128): # now use tab transformer 
        # Prepare data
        X_cat = self.data[self.categorical_cols].values
        X_cont = self.data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.data), 0))
        
        # Convert to PyTorch tensors
        X_cat_tensor = torch.LongTensor(X_cat)
        X_cont_tensor = torch.FloatTensor(X_cont)
        
        # Define TabTransformer-like model
        class TabTransformerEmbedder(nn.Module):
            def __init__(self, num_categories, num_continuous, embedding_dim):
                super().__init__()
                self.embedding_layers = nn.ModuleList([
                    nn.Embedding(num_categories[col], embedding_dim) 
                    for col in range(len(num_categories))
                ])
                self.continuous_projection = nn.Linear(num_continuous, embedding_dim)
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim, nhead=4, dim_feedforward=256
                )
                self.final_projection = nn.Linear(embedding_dim * (len(num_categories) + 1), embedding_dim)
                
            def forward(self, x_cat, x_cont):
                # Embed categorical features
                cat_embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding_layers)]
                cat_embeddings = torch.stack(cat_embeddings, dim=1)
                
                # Project continuous features
                cont_embeddings = self.continuous_projection(x_cont).unsqueeze(1)
                
                # Combine and process with transformer
                combined = torch.cat([cat_embeddings, cont_embeddings], dim=1)
                transformed = self.transformer(combined)
                
                # Pool and project to final embedding
                pooled = transformed.mean(dim=1)
                return pooled
        
        # Get number of categories for each categorical column
        num_categories = [len(self.label_encoders[col].classes_) for col in self.categorical_cols]
        
        # Initialize model
        model = TabTransformerEmbedder(
            num_categories=num_categories,
            num_continuous=X_cont.shape[1],
            embedding_dim=embedding_dim
        )
        
        # Generate embeddings
        dataset = TensorDataset(X_cat_tensor, X_cont_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x_cat, x_cont = batch
                outputs = model(x_cat, x_cont)
                embeddings.append(outputs.numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def get_autoencoder_embeddings(self, embedding_dim=128): # embeddings with an autoencoder
        # Prepare data
        X_cat = self.data[self.categorical_cols].values
        X_cont = self.data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.data), 0))
        X = np.concatenate([X_cat, X_cont], axis=1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Define Autoencoder model
        input_dim = X.shape[1]
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, embedding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        model = Autoencoder()
        
        # Train autoencoder (simplified version- just for poc)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Quick training- will change in a real scenario
        for epoch in range(5):  # Normally would be higher
            for batch in loader:
                inputs = batch[0]
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
        
        # Generate embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                encoded, _ = model(inputs)
                embeddings.append(encoded.numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def get_contrastive_embeddings(self, embedding_dim=128):
        # This is a simplified version again- we'd need proper positive/negative pairs
        # Prepare data
        X_cat = self.data[self.categorical_cols].values
        X_cont = self.data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.data), 0))
        X = np.concatenate([X_cat, X_cont], axis=1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Define contrastive model
        input_dim = X.shape[1]
        class ContrastiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_net = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, embedding_dim)
                )
                
            def forward(self, x):
                return self.embedding_net(x)
        
        model = ContrastiveModel()
        
        criterion = nn.CosineEmbeddingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(5):  # Normally would be higher
            for batch in loader:
                inputs = batch[0]
                # Create artificial pairs
                idx1 = torch.randperm(inputs.size(0))
                idx2 = torch.randperm(inputs.size(0))
                inputs1 = inputs[idx1]
                inputs2 = inputs[idx2]
                
                # Labels: 1 for similar, -1 for dissimilar
                labels = torch.where(torch.rand(inputs1.size(0)) > 0.5, 1, -1).float()
                
                optimizer.zero_grad()
                emb1 = model(inputs1)
                emb2 = model(inputs2)
                loss = criterion(emb1, emb2, labels)
                loss.backward()
                optimizer.step()
        
        # Generate embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = model(inputs)
                embeddings.append(outputs.numpy())
        
        return np.concatenate(embeddings, axis=0)

class FAISSPerformanceTracker:
    def __init__(self, embedding_name):
        self.embedding_name = embedding_name
        
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
        
        return result, execution_time, cpu_usage, mem_usage

    def track_insertion(self, index, new_embeddings):
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        # Track metrics
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, index, num_deletions):
        # Select random indices to delete
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

    def track_update(self, index, num_updates, new_embeddings):
        # Select random indices to update
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_embeddings
        )
        
        # Track metrics
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

    def _update_embeddings(self, index, update_indices, new_embeddings):
        index.remove_ids(update_indices)
        index.add(new_embeddings)
        return index

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

        plt.suptitle(f'Performance Metrics for {self.embedding_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()

    def query_top_k(self, index, query_embeddings, k=5):
        distances, indices = index.search(query_embeddings, k)
        return distances, indices
    
    def track_query(self, index, query_embeddings, k=5):
        num_queries = len(query_embeddings)
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_embeddings, k
        )
        # Track metrics
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)

    def _query_embeddings(self, index, query_embeddings, k):
        index.nprobe = 10
        return index.search(query_embeddings, k)

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

    # Convert relevant columns to numeric
    df["frame.len"] = pd.to_numeric(df["frame.len"])
    df["tcp.srcport"] = pd.to_numeric(df["tcp.srcport"].fillna(0))
    df["tcp.dstport"] = pd.to_numeric(df["tcp.dstport"].fillna(0))
    
    # Define categorical and continuous columns
    categorical_cols = ["ip.src", "ip.dst", "_ws.col.protocol"]
    continuous_cols = ["frame.len", "tcp.srcport", "tcp.dstport"]
    
    # Initialize custom embeddings
    custom_embedder = CustomEmbeddings(df, categorical_cols, continuous_cols)
    
    # Define embedding methods to test
    embedding_methods = {
        "MLP": custom_embedder.get_mlp_embeddings,
        "TabTransformer": custom_embedder.get_tabtransformer_embeddings,
        "Autoencoder": custom_embedder.get_autoencoder_embeddings,
        "Contrastive": custom_embedder.get_contrastive_embeddings
    }
    
    plots_dir = "PipelineResults"
    os.makedirs(plots_dir, exist_ok=True)
    
    for method_name, embed_method in embedding_methods.items():
        print(f"\nProcessing embedding method: {method_name}")
        
        # Generate embeddings
        embeddings = embed_method(embedding_dim=128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Creating FAISS IVFFlat index
        dimension = embeddings.shape[1]
        nlist = min(100, int(np.sqrt(len(embeddings))))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        
        # Initialize performance tracker
        tracker = FAISSPerformanceTracker(method_name)
        
        # Track performance for various operations
        for num_ops in [2500, 5000, 7500, 10000, 20000, 30000]:
            print(f"  Running {num_ops} operations...")
            
            # Generate synthetic data for operations
            new_embeddings = np.random.randn(num_ops, dimension).astype('float32')
            new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            
            # Insertion tracking
            tracker.track_insertion(index, new_embeddings)
            
            # Deletion tracking
            tracker.track_deletion(index, num_ops)
            
            # Update tracking
            tracker.track_update(index, num_ops, new_embeddings)
            
            # Query tracking
            query_embeddings = np.random.randn(num_ops, dimension).astype('float32')
            query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            tracker.track_query(index, query_embeddings, k=5)
            
            # Sample query results
            if num_ops == 1000:
                test_query = np.random.randn(1, dimension).astype('float32')
                test_query = test_query / np.linalg.norm(test_query, axis=1, keepdims=True)
                distances, indices = tracker.query_top_k(index, test_query, k=5)
                print("  Top 5 Neighbors (indices):", indices[0])
                print("  Cosine Similarities:", distances[0])
        
        plot_filename = f"{plots_dir}/{method_name}.png"
        tracker.plot_performance_metrics(save_path=plot_filename)

if __name__ == "__main__":
    main()