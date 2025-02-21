## FAISS vs. Milvus

### Code Struct
*   **`test_installation.py`**: Verified the correct installation of Milvus- attempts to establish a connection
*   **`plain_milvus_demo.py`**: Testing basic Milvus operations, including connecting to the server, creating a collection, and inserting/querying data
*   **`milvus_x_faiss.py`**: Compares FAISS and Milvus performance using random synthetically generated data, providing a controlled baseline
*   **`milvus_x_ipflow_faiss.py`**: Benchmarks FAISS and Milvus for IP flow matching, simulating a practical network-based use case
*   **`milvus_x_faiss_book.py`**: Evaluates FAISS and Milvus using text data from Project Gutenberg (Pride and Prejudice), this was to focus on semantic search capabilities
*   **`multiple_queries.py`**: For multiple repeated queries to assess performance trends and stability over time- this was to see how caching would work for these two
*   **`app.py`**: A Streamlit application for UI
*   **`app3.py`**: Modified Streamlit application- prints chunks retrieved

### Dependencies

The following Python packages are required:

*   `faiss-cpu` (or `faiss-gpu` for GPU support)
*   `pymilvus`
*   `sentence-transformers`
*   `requests`
*   `matplotlib`
*   `streamlit`
*   `pandas`
*   `psutil`

Install the dependencies using pip:

```bash
pip install faiss-cpu pymilvus sentence-transformers requests matplotlib streamlit pandas psutil
```

Reference: [https://milvus.io/docs/](https://milvus.io/docs/)

### Embedding Generation

1.  **Text Chunking**: In `milvus_x_faiss_book.py`, `app.py`, and `app3.py`, the text from Project Gutenberg is divided into chunks of `chunk_size=300` characters. The `split_document` function performs this:

    ```python
    def split_document(text, chunk_size=300):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    ```

    Large documents cannot be directly embedded. Chunking allows us to create vector representations of smaller units of text.

2.  **Sentence Transformers**: Used the `SentenceTransformer("all-MiniLM-L6-v2")` model to convert each text chunk into a vector embedding.

    ```python
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = np.array(embedding_model.encode(chunks), dtype="float32")
    ```

    This is because raw text is unsuitable for similarity searches.
    Sentence Transformers are pre-trained models that map sentences (or text chunks) to a high-dimensional vector space.
    The key is that sentences with similar meanings are placed closer together in this space. 

    Working:

    The `embedding_model.encode(chunks)` function processes each chunk through the transformer network, producing a dense vector representation. These vectors capture the semantic meaning of the text, enabling similarity searches based on content.

4.  **Vector Representation**: The output `vectors` is a NumPy array where each row represents a text chunk, and each column corresponds to a dimension in the embedding space. The dimensionality (`dim`) is determined by the `all-MiniLM-L6-v2` model (typically 384 dimensions for text).

### Querying Mechanisms
1.  **FAISS Indexing + Milvus Storage:**
    *   **FAISS Index Creation**:  FAISS indices are created using different algorithms (FlatL2, IVFFlat, HNSW, IVFPQ) to evaluate their impact on search performance.

        ```python
        indices = {
            "FlatL2": faiss.IndexFlatL2(dim),
            "IVFFlat": faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100),
            "HNSW": faiss.IndexHNSWFlat(dim, 32),
            "IVFPQ": faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, 100, 8, 8)
        }
        ```

        *FlatL2* performs a brute-force search, *IVFFlat* uses an inverted file index for faster search, *HNSW* employs a graph-based approach, and *IVFPQ* combines inverted files with product quantization.

    *   **Milvus Storage**: The generated vectors are stored in a Milvus collection. This provides persistent storage and scalability.

        ```python
        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ], description="FAISS vs. Milvus Benchmark")

        collection = Collection(collection_name, schema)
        collection.insert([vectors.tolist()])
        collection.load()
        ```

    *   **Query Execution (FAISS)**: Queries are executed directly against the FAISS index:

        ```python
        def faiss_search(index, queries):
            index.search(queries, k=5)
        ```

        The `index.search` method returns the `k` nearest neighbors to the query vector based on the distance metric used by the index (L2 distance in this case).

2.  **Milvus-Only Querying:**
    *   In this approach, vectors are stored and searched entirely within Milvus, relying on Milvus's built-in indexing capabilities.
    *   **Query Execution (Milvus)**: Similarity searches are performed using Milvus's `collection.search` method:

        ```python
        def milvus_search(collection, queries):
            search_param = {"metric_type": "L2", "params": {"nprobe": 10}}
            collection.search(queries.tolist(), "vector", search_param, limit=5)
        ```

        The `search_param` dictionary specifies the search parameters, including the distance metric (`L2`) and the number of partitions to search (`nprobe`).

### Changes in `multiple_queries.py`
How it works:

1.  **Defines Query Texts**: Establishes a set of sample query texts that simulate real-world search requests.

    ```python
    query_texts = ["A young lady of remarkable beauty and intelligence", "A conversation about love and marriage"]
    query_vectors = np.array(embedding_model.encode(query_texts), dtype="float32")
    ```

2.  **Run Repeated Queries**: Executes each query multiple times (`num_queries = 50`) against both the FAISS-indexed data and the Milvus collection.

3.  **Benchmark Search**: The `benchmark_search` function (defined in the code) records query time, CPU usage, and memory usage for each query execution. 

    ```python
    def benchmark_search(index, queries, search_fn):
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().used / (1024 ** 2)

        search_fn(index, queries)  # Run the actual search

        query_time = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().used / (1024 ** 2)

        return {
            "Query Time (s)": query_time,
            "CPU Usage (%)": cpu_after - cpu_before,
            "Memory Usage (MB)": mem_after - mem_before
        }
    ```

### Internal Differences Between FAISS and Milvus

| Feature          | FAISS                                       | Milvus                                                       |
| ---------------- | ------------------------------------------- | ------------------------------------------------------------ |
| Architecture     | Library for indexing and searching vectors | Vector database for storage, indexing, and querying           |
| Data Management  | Primarily in-memory                      | Persistent storage with support for disk-based indexing      |
| Scalability      | Limited by available RAM                     | Designed for large-scale, distributed deployments           |
| Indexing Methods | Optimized for speed with diverse algorithms  | Supports multiple indexing methods, optimized for scalability |
| Query Execution  | Fast nearest neighbor search               | Provides a comprehensive query interface                     |

### Performance Metrics
*   **Search Time**: The time required to perform similarity searches.
*   **CPU Usage**: The CPU utilization during search operations.
*   **Memory Usage**: The memory consumption during search operations.

### Results and Analysis
#### How it was done:
* Time: Use `time.time()` to record the start and end times of the search. The difference between these times provides the query execution time in seconds (`query_time = time.time() - start_time`).
* CPU Usage: CPU usage is measured before and after the search operation using `psutil.cpu_percent(interval=None)`. Then we calculate the difference.
* Memory usage: Measured before and after the search operation using `psutil.virtual_memory().used`.

#### Graphs: 
*   **Query Time Across Queries**:
![query_time](https://github.com/user-attachments/assets/fe607290-8d05-4c8f-b501-ae1784cc1d17)

*   **CPU Usage Across Queries**:
![cpu_usage](https://github.com/user-attachments/assets/f7770dd3-0e2e-4fe1-a804-8147e8fc7bbd)

*   **Memory Usage Across Queries**:
![memory_usage](https://github.com/user-attachments/assets/12b6bcbd-d2de-4ccc-9892-83b797693f3f)

### Takeaways
* FAISS-FlatL2 is slowest but most consistent in query time. It's a brute-force method, guaranteeing accuracy but taking longer.
* Approximate methods (FAISS-IVFFlat, HNSW, IVFPQ, Milvus) are faster but less consistent. They trade accuracy for speed.
* All methods except FAISS-FlatL2 utilize high CPU resources. They are CPU-bound, relying heavily on processing power.
* We're getting negative CPU usage- does that mean it's freeing it up? I didn't understand.
