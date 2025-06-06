import streamlit as st
import pandas as pd
import numpy as np
import time
import psutil
import os
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import uuid
from sklearn.metrics import precision_score

class QdrantPerformanceTracker:
    def __init__(self, model_name='distilbert-base-nli-stsb-mean-tokens'):
        self.model_name = model_name
        self.model = None
        self.load_model()
        
        # Initialize Qdrant client
        self.client = QdrantClient(":memory:")
        
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
        
        self.load_latency = None
        self.index_construction_time = None
        self.recall_rates = []
    
    def load_model(self):
        start_time = time.time()
        self.model = SentenceTransformer(self.model_name)
        self.load_latency = time.time() - start_time
    
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
        
        cpu_usage = max(0, end_cpu - start_cpu)
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage
    
    def create_collection(self, collection_name="packets", vector_size=768):
        start_time = time.time()
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            self.index_construction_time = time.time() - start_time
            return True
        except Exception as e:
            st.error(f"Error creating collection: {str(e)}")
            return False
    
    def track_insertion(self, collection_name, new_packet_texts):
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": text}
            )
            for embedding, text in zip(new_embeddings, new_packet_texts)
        ]
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, collection_name, points
        )
        
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)
        
        return len(new_embeddings)
    
    def track_deletion(self, collection_name, num_deletions):
        points = self.client.scroll(collection_name, limit=num_deletions)
        actual_deletions = len(points[0])
        
        if actual_deletions == 0:
            return 0
        
        delete_ids = [point.id for point in points[0]]
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, collection_name, delete_ids
        )
        
        self.deletion_sizes.append(actual_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)
        
        return actual_deletions
    
    def track_update(self, collection_name, num_updates, new_packet_texts):
        points = self.client.scroll(collection_name, limit=num_updates)
        actual_updates = min(num_updates, len(points[0]))
        
        if actual_updates == 0:
            return 0
        
        update_texts = new_packet_texts[:actual_updates]
        update_embeddings = self.model.encode(update_texts, convert_to_numpy=True)
        update_embeddings = update_embeddings / np.linalg.norm(update_embeddings, axis=1, keepdims=True)
        
        update_points = [
            PointStruct(
                id=points[0][i].id,
                vector=update_embeddings[i].tolist(),
                payload={"text": update_texts[i]}
            )
            for i in range(actual_updates)
        ]
        
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, collection_name, update_points
        )
        
        self.update_sizes.append(actual_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage)
        
        return actual_updates
    
    def track_query(self, collection_name, query_texts, k=5, ground_truth=None):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        num_queries = len(query_texts)
        
        result, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, collection_name, query_embeddings, k
        )
        
        # Calculate recall if ground truth is provided
        if ground_truth is not None:
            distances, indices = result
            recall = self._calculate_recall(indices, ground_truth)
            self.recall_rates.append(recall)
        
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)
        
        return result
    
    def _calculate_recall(self, retrieved_ids, ground_truth_ids):
        # Convert to sets for easier comparison
        retrieved_set = set(retrieved_ids.flatten())
        ground_truth_set = set(ground_truth_ids)
        
        # Calculate number of relevant items retrieved
        relevant_retrieved = len(retrieved_set.intersection(ground_truth_set))
        
        # Calculate recall
        if len(ground_truth_set) > 0:
            recall = relevant_retrieved / len(ground_truth_set)
        else:
            recall = 0.0
            
        return recall
    
    def _insert_embeddings(self, collection_name, points):
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        return True
    
    def _delete_embeddings(self, collection_name, point_ids):
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=point_ids
            )
        )
        return True
    
    def _update_embeddings(self, collection_name, points):
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        return True
    
    def _query_embeddings(self, collection_name, query_embeddings, k):
        results = []
        for embedding in query_embeddings:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=embedding.tolist(),
                limit=k
            )
            results.append(search_result)
        
        distances = []
        indices = []
        for result in results:
            batch_distances = []
            batch_indices = []
            for hit in result:
                batch_distances.append(1 - hit.score)
                batch_indices.append(hit.id)
            distances.append(batch_distances)
            indices.append(batch_indices)
        
        return np.array(distances), np.array(indices)
    
    def get_collection_count(self, collection_name):
        try:
            return self.client.get_collection(collection_name).points_count
        except:
            return 0
    
    def create_performance_plot(self):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Execution Time', 'CPU Usage', 'Memory Usage', 'Recall Rate'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time plot
        if self.insertion_sizes:
            fig.add_trace(go.Scatter(x=self.insertion_sizes, y=self.insertion_times, 
                                   mode='lines+markers', name='Insertion', line=dict(color='blue')), row=1, col=1)
        if self.deletion_sizes:
            fig.add_trace(go.Scatter(x=self.deletion_sizes, y=self.deletion_times, 
                                   mode='lines+markers', name='Deletion', line=dict(color='red')), row=1, col=1)
        if self.update_sizes:
            fig.add_trace(go.Scatter(x=self.update_sizes, y=self.update_times, 
                                   mode='lines+markers', name='Update', line=dict(color='green')), row=1, col=1)
        if self.query_sizes:
            fig.add_trace(go.Scatter(x=self.query_sizes, y=self.query_times, 
                                   mode='lines+markers', name='Query', line=dict(color='orange')), row=1, col=1)
        
        # CPU plot
        if self.insertion_sizes:
            fig.add_trace(go.Scatter(x=self.insertion_sizes, y=self.insertion_cpu, 
                                   mode='lines+markers', name='Insertion', line=dict(color='blue'), 
                                   showlegend=False), row=1, col=2)
        if self.deletion_sizes:
            fig.add_trace(go.Scatter(x=self.deletion_sizes, y=self.deletion_cpu, 
                                   mode='lines+markers', name='Deletion', line=dict(color='red'), 
                                   showlegend=False), row=1, col=2)
        if self.update_sizes:
            fig.add_trace(go.Scatter(x=self.update_sizes, y=self.update_cpu, 
                                   mode='lines+markers', name='Update', line=dict(color='green'), 
                                   showlegend=False), row=1, col=2)
        if self.query_sizes:
            fig.add_trace(go.Scatter(x=self.query_sizes, y=self.query_cpu, 
                                   mode='lines+markers', name='Query', line=dict(color='orange'), 
                                   showlegend=False), row=1, col=2)
        
        # Memory plot
        if self.insertion_sizes:
            fig.add_trace(go.Scatter(x=self.insertion_sizes, y=self.insertion_memory, 
                                   mode='lines+markers', name='Insertion', line=dict(color='blue'), 
                                   showlegend=False), row=2, col=1)
        if self.deletion_sizes:
            fig.add_trace(go.Scatter(x=self.deletion_sizes, y=self.deletion_memory, 
                                   mode='lines+markers', name='Deletion', line=dict(color='red'), 
                                   showlegend=False), row=2, col=1)
        if self.update_sizes:
            fig.add_trace(go.Scatter(x=self.update_sizes, y=self.update_memory, 
                                   mode='lines+markers', name='Update', line=dict(color='green'), 
                                   showlegend=False), row=2, col=1)
        if self.query_sizes:
            fig.add_trace(go.Scatter(x=self.query_sizes, y=self.query_memory, 
                                   mode='lines+markers', name='Query', line=dict(color='orange'), 
                                   showlegend=False), row=2, col=1)
        
        # Recall plot
        if self.recall_rates:
            fig.add_trace(go.Scatter(x=list(range(len(self.recall_rates))), y=self.recall_rates,
                                   mode='lines+markers', name='Recall', line=dict(color='purple')), row=2, col=2)
        
        fig.update_xaxes(title_text="Number of Operations", row=1, col=1)
        fig.update_xaxes(title_text="Number of Operations", row=1, col=2)
        fig.update_xaxes(title_text="Number of Operations", row=2, col=1)
        fig.update_xaxes(title_text="Query Number", row=2, col=2)
        
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="CPU Percentage", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        fig.update_yaxes(title_text="Recall Rate", row=2, col=2)
        
        fig.update_layout(
            title=f'Performance Metrics for {self.model_name} with Qdrant',
            height=800,
            showlegend=True
        )
        
        return fig

def load_csv_data(csv_path):
    try:
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
        
        df["packet_text"] = (
            df["ip.src"].fillna('') + " " +
            df["ip.dst"].fillna('') + " " +
            df["_ws.col.protocol"].fillna('') + " " +
            df["tcp.srcport"].fillna('') + " " +
            df["tcp.dstport"].fillna('') + " " +
            df["frame.len"].fillna('')
        )
        
        return df["packet_text"].tolist()
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def initialize_session_state():
    if 'tracker' not in st.session_state:
        model_options = [
            'distilbert-base-nli-stsb-mean-tokens',
            'all-MiniLM-L6-v2',
            'bert-base-nli-mean-tokens',
            'roberta-base-nli-stsb-mean-tokens'
        ]
        selected_model = st.sidebar.selectbox("Select Embedding Model:", model_options)
        st.session_state.tracker = QdrantPerformanceTracker(model_name=selected_model)
        st.session_state.collection_name = "network_packets"
    
    if 'data_texts' not in st.session_state:
        csv_path = st.sidebar.text_input(
            "CSV File Path:", 
            value="ip_flow_dataset.csv",
            help="Enter the path to your CSV file"
        )
        
        if st.sidebar.button("Load CSV Data") or 'csv_loaded' not in st.session_state:
            packet_texts = load_csv_data(csv_path)
            
            if packet_texts:
                st.sidebar.success(f"Loaded {len(packet_texts)} entries from CSV")
                
                with st.spinner("Creating embeddings from CSV data..."):
                    inserted_count = st.session_state.tracker.track_insertion(
                        st.session_state.collection_name,
                        packet_texts
                    )
                
                st.session_state.data_texts = packet_texts.copy()
                st.session_state.csv_loaded = True
                st.session_state.ground_truth = [str(i) for i in range(len(packet_texts))][:100]  # Sample ground truth
                st.rerun()
            else:
                st.sidebar.warning("Could not load CSV, using sample data")
                sample_texts = [
                    "192.168.1.1 192.168.1.2 TCP 80 443",
                    "10.0.0.1 10.0.0.2 UDP 53 8080",
                    "172.16.0.1 172.16.0.2 HTTP 80 8080",
                    "192.168.0.1 192.168.0.2 HTTPS 443 443"
                ]
                
                st.session_state.tracker.create_collection(
                    st.session_state.collection_name,
                    vector_size=st.session_state.tracker.model.get_sentence_embedding_dimension()
                )
                
                inserted_count = st.session_state.tracker.track_insertion(
                    st.session_state.collection_name,
                    sample_texts
                )
                
                st.session_state.data_texts = sample_texts.copy()
                st.session_state.csv_loaded = True
                st.session_state.ground_truth = [str(i) for i in range(len(sample_texts))]

def main():
    st.set_page_config(page_title="Qdrant Performance Tracker", layout="wide")
    st.title("Qdrant Performance Tracker with Multiple Embeddings")
    st.markdown("Perform CRUD operations on network packet data and visualize performance metrics")
    
    initialize_session_state()
    
    if 'csv_loaded' not in st.session_state:
        st.info("Please load your CSV data using the sidebar controls first!")
        return
    
    st.sidebar.header("Operations")
    operation = st.sidebar.selectbox(
        "Choose Operation:",
        ["Insert", "Delete", "Update", "Query", "View Data"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{operation} Operation")
        
        if operation == "Insert":
            st.write("Add new packet data to the database")
            num_entries = st.number_input("Number of entries to insert:", min_value=1, max_value=10000, value=10)
            new_data = st.text_area(
                "Enter packet data (one per line):",
                placeholder="192.168.1.10 192.168.1.20 TCP 80 8080\n10.0.0.5 10.0.0.10 UDP 53 53",
                height=100
            )
            
            if st.button("Insert Data"):
                if new_data.strip():
                    new_texts = [line.strip() for line in new_data.split('\n') if line.strip()]
                    if len(new_texts) != num_entries:
                        st.warning(f"Provided {len(new_texts)} entries, but requested {num_entries}. Using provided data.")
                else:
                    new_texts = [f"192.168.{i//255}.{i%255} 10.0.{i//255}.{i%255} TCP {80+i} {8000+i}" 
                               for i in range(num_entries)]
                
                with st.spinner("Inserting data..."):
                    inserted_count = st.session_state.tracker.track_insertion(
                        st.session_state.collection_name,
                        new_texts
                    )
                    st.session_state.data_texts.extend(new_texts)
                
                st.success(f"Inserted {inserted_count} entries successfully!")
                st.info(f"Total entries in database: {st.session_state.tracker.get_collection_count(st.session_state.collection_name)}")
        
        elif operation == "Delete":
            current_count = st.session_state.tracker.get_collection_count(st.session_state.collection_name)
            st.write("Remove entries from the database")
            num_to_delete = st.number_input("Number of entries to delete:", min_value=1, max_value=max(1, current_count), value=1)
            
            if st.button("Delete Data"):
                if current_count > 0:
                    with st.spinner("Deleting data..."):
                        deleted_count = st.session_state.tracker.track_deletion(
                            st.session_state.collection_name,
                            num_to_delete
                        )
                        if len(st.session_state.data_texts) >= deleted_count:
                            st.session_state.data_texts = st.session_state.data_texts[:-deleted_count]
                    
                    st.success(f"Deleted {deleted_count} entries successfully!")
                    st.info(f"Remaining entries in database: {st.session_state.tracker.get_collection_count(st.session_state.collection_name)}")
                else:
                    st.warning("No data to delete!")
        
        elif operation == "Update":
            current_count = st.session_state.tracker.get_collection_count(st.session_state.collection_name)
            st.write("Update existing entries in the database")
            num_to_update = st.number_input("Number of entries to update:", min_value=1, max_value=max(1, current_count), value=1)
            update_data = st.text_area(
                "Enter new packet data (one per line):",
                placeholder="10.10.10.1 10.10.10.2 HTTPS 443 443\n172.16.1.1 172.16.1.2 SSH 22 22",
                height=100
            )
            
            if st.button("Update Data"):
                if current_count > 0:
                    if update_data.strip():
                        new_texts = [line.strip() for line in update_data.split('\n') if line.strip()]
                        if len(new_texts) < num_to_update:
                            st.warning(f"Provided {len(new_texts)} entries, but requested {num_to_update}. Using provided data.")
                    else:
                        new_texts = [f"10.10.{i//255}.{i%255} 172.16.{i//255}.{i%255} HTTPS {443+i} {9000+i}" 
                                   for i in range(num_to_update)]
                    
                    with st.spinner("Updating data..."):
                        updated_count = st.session_state.tracker.track_update(
                            st.session_state.collection_name,
                            num_to_update,
                            new_texts
                        )
                    
                    st.success(f"Updated {updated_count} entries successfully!")
                    st.info(f"Total entries in database: {st.session_state.tracker.get_collection_count(st.session_state.collection_name)}")
                else:
                    st.warning("No data to update!")
        
        elif operation == "Query":
            current_count = st.session_state.tracker.get_collection_count(st.session_state.collection_name)
            st.write("Search for similar packet data")
            query_text = st.text_input(
                "Enter query:",
                placeholder="192.168.1.1 192.168.1.2 TCP 80 443"
            )
            k = st.slider("Number of results (k):", 1, min(20, max(1, current_count)), 5)
            
            if st.button("Search"):
                if query_text.strip() and current_count > 0:
                    with st.spinner("Searching..."):
                        results = st.session_state.tracker.track_query(
                            st.session_state.collection_name,
                            [query_text],
                            k,
                            ground_truth=st.session_state.ground_truth if 'ground_truth' in st.session_state else None
                        )
                    
                    st.success(f"Query completed!")
                    
                    if results and len(results[0]) > 0 and len(results[0][0]) > 0:
                        distances, indices = results
                        st.write("**Search Results:**")
                        
                        result_points = []
                        for point_id in indices[0]:
                            point = st.session_state.tracker.client.retrieve(
                                st.session_state.collection_name,
                                ids=[point_id]
                            )
                            if point:
                                result_points.append(point[0])
                        
                        results_df = pd.DataFrame({
                            'Rank': range(1, len(distances[0]) + 1),
                            'ID': indices[0],
                            'Similarity Score': [f"{1-d:.4f}" for d in distances[0]],
                            'Data': [point.payload.get('text', 'No text') if point else 'No data' 
                                   for point in result_points]
                        })
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        st.warning("No results found!")
                elif not query_text.strip():
                    st.warning("Please enter a query!")
                else:
                    st.warning("Database is empty!")
        
        elif operation == "View Data":
            current_count = st.session_state.tracker.get_collection_count(st.session_state.collection_name)
            st.write("Current database statistics")
            st.metric("Total Entries", current_count)
            st.metric("Model", st.session_state.tracker.model_name)
            st.metric("CSV Data Loaded", "Yes" if 'csv_loaded' in st.session_state else "No")
            
            if st.session_state.data_texts:
                st.write("**Sample Data:**")
                sample_size = min(10, len(st.session_state.data_texts))
                sample_df = pd.DataFrame({
                    'Index': range(1, sample_size + 1),
                    'Packet Data': st.session_state.data_texts[:sample_size]
                })
                st.dataframe(sample_df, use_container_width=True)
                
                if len(st.session_state.data_texts) > sample_size:
                    st.write(f"... and {len(st.session_state.data_texts) - sample_size} more entries")
                
                if st.checkbox("Show Data Analysis"):
                    st.write("**Data Analysis:**")
                    protocols = []
                    for text in st.session_state.data_texts[:1000]:
                        parts = text.split()
                        if len(parts) >= 3:
                            protocols.append(parts[2])
                    
                    if protocols:
                        protocol_counts = pd.Series(protocols).value_counts()
                        st.write("**Top Protocols:**")
                        st.bar_chart(protocol_counts.head(10))
    
    with col2:
        st.subheader("Performance Metrics")
        
        has_data = (len(st.session_state.tracker.insertion_times) > 0 or 
                   len(st.session_state.tracker.deletion_times) > 0 or 
                   len(st.session_state.tracker.update_times) > 0 or 
                   len(st.session_state.tracker.query_times) > 0 or
                   st.session_state.tracker.load_latency is not None or
                   st.session_state.tracker.index_construction_time is not None)
        
        if has_data:
            fig = st.session_state.tracker.create_performance_plot()
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Performance Summary:**")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                if st.session_state.tracker.load_latency is not None:
                    st.metric("Model Load Latency", f"{st.session_state.tracker.load_latency:.4f}s")
                if st.session_state.tracker.insertion_times:
                    avg_insert_time = np.mean(st.session_state.tracker.insertion_times)
                    st.metric("Avg Insert Time", f"{avg_insert_time:.4f}s")
                if st.session_state.tracker.query_times:
                    avg_query_time = np.mean(st.session_state.tracker.query_times)
                    st.metric("Avg Query Time", f"{avg_query_time:.4f}s")
            
            with metrics_col2:
                if st.session_state.tracker.index_construction_time is not None:
                    st.metric("Index Construction Time", f"{st.session_state.tracker.index_construction_time:.4f}s")
                if st.session_state.tracker.deletion_times:
                    avg_delete_time = np.mean(st.session_state.tracker.deletion_times)
                    st.metric("Avg Delete Time", f"{avg_delete_time:.4f}s")
                if st.session_state.tracker.update_times:
                    avg_update_time = np.mean(st.session_state.tracker.update_times)
                    st.metric("Avg Update Time", f"{avg_update_time:.4f}s")
                if st.session_state.tracker.recall_rates:
                    avg_recall = np.mean(st.session_state.tracker.recall_rates)
                    st.metric("Avg Recall Rate", f"{avg_recall:.4f}")
        else:
            st.info("Performance graphs will appear here after you perform operations")
            st.write("Try inserting, deleting, updating, or querying data to see performance metrics!")

    st.markdown("---")

if __name__ == "__main__":
    main()
