## Progress

### 1. **Similarity Score Calculation**
- Modified `main.py` (refer to line 72 onwards) to display similarity scores for benchmarked queries.
- Weaviate uses **cosine distance** by default to measure vector similarity.
  - Cosine similarity is calculated as:  
    Cosine Similarity = 1 - Cosine Distance
  - Interpretation:
    - **Lower distance → Higher similarity**
    - **Higher distance → Lower similarity**

---

### 2. **Handling Update & Delete Operations in Weaviate**
Weaviate initially limited **update** and **delete** operations to the first 100 matching records. This was addressed using **batch processing** (`query.py`).

- **Delete Operation**
  - Repeatedly fetches records matching the condition and deletes them until no more are found.

- **Update Operation**
  - A naive batch update approach could lead to an **infinite loop** due to Weaviate’s **lack of real-time index refresh**.
  - To fix this, we modify the filter condition to **exclude already updated records**, ensuring the update process terminates correctly.

---

### 3. **Garbage Collection in Weaviate**
- Weaviate is written in **Go**, a **garbage-collected** language.
- This means:
  - Memory is **not immediately available** for reuse.
  - The **garbage collector runs asynchronously**, which may introduce memory overhead.
  - This factor could **affect final inference performance** due to inefficient memory cleanup.

---

### 4. **Vector Indexing in Weaviate**
Weaviate supports multiple vector indexing methods:

| Index Type          | Description |
|---------------------|-------------|
| **HNSW (Default)** | Hierarchical Navigable Small World - optimized for large-scale search. |
| **Flat Indexing**  | Brute-force text matching, unsuitable for semantic search. |
| **Dynamic Indexing** | Hybrid approach - uses Flat Indexing for small datasets, switches to HNSW for larger datasets. |

- **Why We Stick to HNSW?**
  - **Dynamic Indexing** introduces additional overhead.
  - Since our dataset is always large, we avoid unnecessary complexity by **directly using HNSW**.

---

