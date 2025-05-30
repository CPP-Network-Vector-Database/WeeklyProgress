import streamlit as st
import pandas as pd
from similarity_pipeline import QdrantPerformanceTracker  # Assuming your class is in this file
import matplotlib.pyplot as plt
import os

def main():
    st.title("Qdrant Vector Database Performance Tracker")
    st.write("""
    This application tracks and visualizes the performance of Qdrant vector database operations 
    including insertions, deletions, updates, and queries.
    """)

    # Sidebar configuration
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Embedding Model",
        options=['all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1'],
        index=0
    )
    
    initial_samples = st.sidebar.number_input(
        "Initial Dataset Size", 
        min_value=100, 
        max_value=10000, 
        value=1000
    )
    
    test_batches = st.sidebar.text_input(
        "Batch Sizes (comma separated)", 
        value="2500,5000,7500,10000"
    )
    
    test_queries = st.sidebar.number_input(
        "Number of Test Queries", 
        min_value=1, 
        max_value=10, 
        value=3
    )
    
    # Parse batch sizes
    try:
        test_batches = [int(x.strip()) for x in test_batches.split(",") if x.strip().isdigit()]
    except:
        st.error("Invalid batch sizes format. Please enter comma-separated integers.")
        test_batches = [2500, 5000, 7500, 10000]
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=["csv"],
        help="Dataset should contain network flow data with multiple columns"
    )
    
    if not uploaded_file:
        st.warning("Please upload a CSV file to proceed")
        return
    
    # Load data
    try:
        df = pd.read_csv(uploaded_file, nrows=initial_samples)
        if len(df.columns) < 2:
            st.error("Uploaded file doesn't have enough columns to create meaningful text")
            return
            
        df["packet_text"] = df.apply(lambda row: " ".join(str(x) for x in row), axis=1)
        st.success(f"Successfully loaded {len(df)} records")
        
        # Show sample data
        if st.checkbox("Show sample data"):
            st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return
    
    # Initialize tracker
    tracker = QdrantPerformanceTracker(model_name)
    
    # Operation selection
    st.header("Operations to Perform")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        do_insert = st.checkbox("Insert", value=True)
    with col2:
        do_delete = st.checkbox("Delete", value=True)
    with col3:
        do_update = st.checkbox("Update", value=True)
    with col4:
        do_query = st.checkbox("Query", value=True)
    
    # Percentage controls
    if do_delete:
        delete_percent = st.slider(
            "Percentage to delete", 
            min_value=1, 
            max_value=100, 
            value=30
        ) / 100
    
    if do_update:
        update_percent = st.slider(
            "Percentage to update", 
            min_value=1, 
            max_value=100, 
            value=20
        ) / 100
    
    # Query text input
    if do_query:
        default_queries = [
            "192.168.1.1 TCP 80",
            "10.0.0.1 UDP 53",
            "PROTO_TCP"
        ]
        
        query_texts = []
        for i in range(test_queries):
            query = st.text_input(
                f"Query {i+1}", 
                value=default_queries[i] if i < len(default_queries) else ""
            )
            if query:
                query_texts.append(query)
    
    # Run tests button
    if st.button("Run Performance Tests"):
        if not (do_insert or do_delete or do_update or do_query):
            st.warning("Please select at least one operation to perform")
            return
        
        if do_query and not query_texts:
            st.warning("Please enter at least one query text")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create collection
        status_text.text("Initializing collection...")
        initial_embeddings = tracker.model.encode(df["packet_text"].head(100).tolist())
        tracker.create_collection(vector_size=initial_embeddings.shape[1])
        progress_bar.progress(5)
        
        # Insert initial data
        status_text.text("Inserting initial data...")
        tracker.track_insertion(df["packet_text"].head(100).tolist())
        progress_bar.progress(10)
        
        total_steps = len(test_batches) * (do_insert + do_delete + do_update + do_query)
        current_step = 0
        
        # Run tests
        results = []
        for i, num_ops in enumerate(test_batches):
            batch_results = {"Batch Size": num_ops}
            status_text.text(f"Processing batch size: {num_ops} ({i+1}/{len(test_batches)})")
            
            # Insert synthetic data
            if do_insert:
                current_step += 1
                progress = 10 + (current_step / total_steps * 90)
                progress_bar.progress(int(progress))
                
                new_texts = [f"TEST_IP_{i} PROTO_TCP" for i in range(num_ops)]
                tracker.track_insertion(new_texts)
                batch_results["Insert Time"] = tracker.insertion_times[-1]
                batch_results["Insert CPU"] = tracker.insertion_cpu[-1]
                batch_results["Insert Memory"] = tracker.insertion_memory[-1]
            
            # Delete
            if do_delete:
                current_step += 1
                progress = 10 + (current_step / total_steps * 90)
                progress_bar.progress(int(progress))
                
                tracker.track_deletion(int(num_ops * delete_percent))
                batch_results["Delete Time"] = tracker.deletion_times[-1]
                batch_results["Delete CPU"] = tracker.deletion_cpu[-1]
                batch_results["Delete Memory"] = tracker.deletion_memory[-1]
            
            # Update
            if do_update:
                current_step += 1
                progress = 10 + (current_step / total_steps * 90)
                progress_bar.progress(int(progress))
                
                update_texts = [f"UPDATE_{i} PROTO_UDP" for i in range(int(num_ops * update_percent))]
                tracker.track_update(int(num_ops * update_percent), update_texts)
                batch_results["Update Time"] = tracker.update_times[-1]
                batch_results["Update CPU"] = tracker.update_cpu[-1]
                batch_results["Update Memory"] = tracker.update_memory[-1]
            
            # Query
            if do_query:
                current_step += 1
                progress = 10 + (current_step / total_steps * 90)
                progress_bar.progress(int(progress))
                
                tracker.track_query(query_texts, k=3)
                batch_results["Query Time"] = tracker.query_times[-1]
                batch_results["Query CPU"] = tracker.query_cpu[-1]
                batch_results["Query Memory"] = tracker.query_memory[-1]
            
            results.append(batch_results)
        
        progress_bar.progress(100)
        status_text.text("Tests completed!")
        
        # Show results
        st.header("Performance Results")
        
        # Plot metrics
        st.subheader("Performance Metrics")
        fig = tracker.plot_performance_metrics()
        st.pyplot(fig)
        
        # Raw data
        st.subheader("Raw Performance Data")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Query results
        if do_query:
            st.subheader("Query Results")
            # You would need to modify your tracker to store and expose query results
            # For now, we'll just show that queries were executed
            st.write(f"Executed {len(query_texts)} queries with results printed to console")
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='qdrant_performance_results.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()