# Results of Latency Benchmarks – Pgvector Database

## Dataset: 32 K rows IP Flow

## Time Taken by different CRUD Operations

> **Formula**:  
> **Total Time = Planning Time + Execution Time**

| **Operation**             | **IVF**            | **HNSW**           | **Without Indexing**   |
|--------------------------|--------------------|--------------------|------------------------|
| **Create: Insertion**    |                    |                    |                        |
| • Planning Time          | 3.721 ms           | 3.598 ms           | 3.961 ms               |
| • Execution Time         | 1.387 ms            | 26.896 ms            | 0.267 ms                |
| **Read: Cosine Similarity** |                |                    |                        |
| • Planning Time          | 0.053 ms           | 0.051 ms           | 0.060 ms               |
| • Execution Time         | 29.842 ms             | 0.444 ms              | 15.788 ms                 |
| **Delete: Deletion of 10k records** |         |                    |                        |
| • Planning Time          | 0.220 ms           | 0.503 ms           | 0.118 ms               |
| • Execution Time         | 86.857 ms            | 127.105 ms            | 46.248 ms                |

---

## Screenshots of Results

### Insertion

![Insertion IVF](Screenshots/insert_ivf.png)
![Insertion HNSW](Screenshots/insert_hnsw.png)
![Insertion Without Indexing](Screenshots/insert_without.png)

### Cosine Similarity (Read)

![Cosine IVF](Screenshots/cosine_ivf.png)
![Cosine HNSW](Screenshots/cosine_hnsw.png)
![Cosine Without Indexing](Screenshots/cosine_without.png)

### Deletion

![Delete IVF](Screenshots/delete_ivf.png)
![Delete HNSW](Screenshots/delete_hnsw.png)
![Delete Without Indexing](Screenshots/delete_without.png)
