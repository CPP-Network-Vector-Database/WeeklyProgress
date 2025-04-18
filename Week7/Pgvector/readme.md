# Results of Query latency of Pgvector database
## Time Taken 
| **Dataset: 32 K rows IP Flow**       | **IVF**        | **HNSW**      | **Without Indexing**|
|-------------------|--------------------|-----------------|-----------------|
| **Query Insertion**         |  16.75 s  |  24.70 s  | 16.21 s |
| **Query Cosine Similarity**  | 113 ms |  74 ms  | 130 ms | 