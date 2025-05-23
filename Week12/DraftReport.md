# Evaluating Vector Databases for IP Flow Embeddings

In this project, we evaluated a set of vector databases on how well they handle exactly network data. Network data, like IP flows is structured, high-volume, and comes with real-world constraints- so the goal was to test how well modern vector stores hold up when used for this kind of workload.  

---

## 1. Abstract

This project takes a deep dive into evaluating how well different vector databases manage IP flow embeddings. Instead of working with the usual suspects- natural language text or images- we focused on network data, which comes with its own quirks: rapid ingestion needs, large scale, and structured feature sets.

We tested six popular vector databases using embeddings generated from transformer-based models like BERT and DistilBERT. The evaluation looked at things like indexing time, query latency, memory usage, recall scores, and general system overhead. 

The goal isn’t to crown a single winner, but to give a clear sense of where each system fits in, and what trade-offs come into play when the data isn’t from text or vision tasks- but from a network.

---

## 2. Introduction

As networks continue to scale and get more complex, we’re generating more flow-level data than ever. IP flows- defined by packet exchanges between endpoints- are key to monitoring, anomaly detection, and traffic shaping. But with so much data being generated in real time, the challenge becomes one of storage and efficient retrieval.

That’s where vector databases come in. These systems are built to handle high-dimensional vector data and support similarity search at scale. But most of the literature and tools are geared toward NLP or vision tasks.

So, in this project, we decided to flip the script a little. We generated embeddings from IP flow records, stored them in different vector databases, and tested them across a range of benchmarks- indexing speed, query time, memory/disk usage, and how accurate the similarity search results were. The idea was to get a sense of how these tools hold up when used in a more infrastructure-heavy, less text-focused setting.

---

## 3. IP Flow Embeddings
(add more here pls)
For our dataset, we used IP flow records. These include fields like source IP, destination IP, source port, destination port, timestamp, protocol, byte count, packet count, and so on.

We generated embeddings using two approaches:
- **Handcrafted feature vectors**: Standardized and normalized flow-level features converted into fixed-length vectors.
- **Pretrained language model embeddings**: Using BERT and DistilBERT to encode flow metadata into semantic vectors.

These embeddings were then used as the core dataset for benchmarking the vector databases.

---

## 4. Vector Databases Evaluated

We evaluated the following vector databases:

- **FAISS**  
- **Milvus**  
- **Qdrant**  
- **Weaviate**  
- **Chroma**  
- **PgVector**

Each was tested using their supported ANN indexing methods- like IVF, HNSW, and flat- and tuned only minimally, to replicate what a practical deployment might look like with defaults or near-defaults.

---

## 5. Benchmark Methodology

Each system was tested across the following criteria:

- **Indexing Time**
- **Query Latency (mean, p95, p99)**
- **Recall@k**
- **Memory Usage (RAM during insert and query)**
- **Disk Footprint**
- **CPU Utilization**
- **Ease of Use / Developer Experience**

All experiments were done under identical hardware and data conditions.

---

## 6. Results

### Performance Summary

| Database | Indexing Time | Query p95 | Recall@10 | RAM Usage | Disk Size | Notes         |
|----------|----------------|------------|------------|------------|------------|----------------|
| FAISS    | ___            | ___        | ___        | ___        | ___        | ____________   |
| Qdrant   | ___            | ___        | ___        | ___        | ___        | ____________   |
| Milvus   | ___            | ___        | ___        | ___        | ___        | ____________   |
| Weaviate | ___            | ___        | ___        | ___        | ___        | ____________   |
| Chroma   | ___            | ___        | ___        | ___        | ___        | ____________   |
| PgVector | ___            | ___        | ___        | ___        | ___        | ____________   |

We also plotted:
- Query latency vs. recall
- Indexing time across databases
- RAM and disk usage
- Radar charts for trade-offs

---

## 7. Analysis and Discussion

---

## 8. Recommendations


| Use Case                    | Recommended DB | Why                                    |
|----------------------------|----------------|----------------------------------------|
| High-performance lookups   | -----          | ????              |
| Real-time threat detection | ------         | ????       |
| Lightweight/local setup    | ------         | ????  |
| Semantic + structured search | --------     | ????   |
| Production + managed infra | --------       | ????       |

---

## 9. Limitations


---

## 10. Future Work



---

## 11. Conclusion



---

## 12. Appendix

- Scripts used for benchmarking and logging
- Details of the embedding pipeline
- Configuration files for each DB
- Full tables of raw metrics
