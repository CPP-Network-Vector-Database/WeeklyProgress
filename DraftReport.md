# Evaluating Vector Databases for IP Flow Embeddings

In this project, we evaluated a comprehensive set of vector databases on their ability to handle network data effectively. Network data, particularly IP flows, presents unique challenges with its structured format, high-volume ingestion requirements, and real-time processing constraints. Our goal was to assess how well modern vector stores perform when applied to this specialized domain, moving beyond traditional text and image embedding workloads.

---

## 1. Abstract

This project provides a comprehensive evaluation of six prominent vector databases in their capacity to manage IP flow embeddings. Unlike conventional evaluations that focus on natural language processing or computer vision applications, we specifically examined network data characteristics: rapid ingestion requirements, large-scale operations, and structured feature representations.

We benchmarked six vector databases using embeddings generated from multiple transformer-based models including BERT, DistilBERT, and specialized network embedding approaches. Our evaluation encompassed indexing performance, query latency, memory utilization, recall accuracy, and overall system overhead across different embedding strategies.

Rather than identifying a single optimal solution, this research illuminates the specific strengths and trade-offs of each system when applied to network infrastructure data, providing practical guidance for network monitoring, anomaly detection, and traffic analysis applications.

---

## 2. Introduction

Modern network infrastructures generate unprecedented volumes of flow-level data as traffic complexity and scale continue to expand. IP flows, characterized by packet exchanges between network endpoints, serve as fundamental building blocks for network monitoring, anomaly detection, security analysis, and traffic optimization. The challenge lies not merely in collecting this data, but in storing and retrieving it efficiently for real-time analysis and historical investigation.

Vector databases have emerged as powerful solutions for managing high-dimensional data and enabling similarity-based searches at scale. However, the majority of existing research and tooling focuses on natural language processing and computer vision applications, leaving a significant gap in understanding their performance characteristics for network data.

This project addresses that gap by systematically evaluating how vector databases perform when tasked with managing IP flow embeddings. We generated embeddings from authentic IP flow records, deployed them across six different vector database systems, and conducted comprehensive benchmarking across multiple performance dimensions including indexing speed, query latency, resource utilization, and search accuracy.

---

## 3. IP Flow Embeddings

### 3.1 Dataset Characteristics

Our evaluation dataset consists of IP flow records captured from simulated communication between containers. Each flow record contains standard network telemetry fields including source and destination IP addresses, port numbers, timestamps, protocol identifiers, byte counts, packet counts, flow duration, and TCP flags.

### 3.2 Embedding Generation Approaches

We employed multiple embedding strategies to capture different aspects of network flow semantics:

**Pretrained Language Model Embeddings**: Flow metadata was serialized into text representations and processed through transformer models including BERT and DistilBERT. This method captures implicit semantic relationships between network flows that may not be apparent through traditional feature engineering.

**Specialized Network Embeddings**: We incorporated domain-specific embedding models designed for network data, including approaches that leverage graph neural networks and specialized architectures for IP address and protocol representations. **will write about later**

### 3.3 Embedding Models Evaluated

| Model | Type | Dimensionality | Key Characteristics |
|-------|------|----------------|-------------------|
| **paraphrase-MiniLM-v2** | Sentence Transformer | 384 | Efficient semantic similarity, optimized for paraphrase detection |
| **MiniLM-L6-v2** | Sentence Transformer | 384 | Balanced speed-accuracy trade-off, general-purpose embedding |
| **bert-base-nli-mean-tokens** | BERT-based | 768 | Strong natural language inference capabilities |
| **mpnet-base-v2** | MPNet | 768 | Superior contextual sentence meaning capture |
| **microsoft-codebert-base** | CodeBERT | 768 | Code understanding specialization, relevant for protocol analysis |
| **average_word_embeddings_komn** | Word2Vec-based | 300 | Lightweight word representations, efficient processing |
| **ip2vec** | Custom Word2Vec | 100 | Network-specific embeddings using Word2Vec foundation |

---

## 4. Vector Databases Evaluated

We selected six representative vector databases that span different architectural approaches, deployment models, and performance characteristics:

### 4a. FAISS (Facebook AI Similarity Search) [https://github.com/facebookresearch/faiss]

**Overview**: FAISS represents Meta's approach to efficient similarity search and dense vector clustering. Originally developed for large-scale computer vision applications, FAISS has become a foundational library for vector operations across numerous domains.

**Architecture**: FAISS operates as a library rather than a standalone database system, providing optimized implementations of various approximate nearest neighbor (ANN) algorithms. It supports both CPU and GPU acceleration, with particular strength in batch processing scenarios.

**Index Types Evaluated**:
- **IndexFlatL2**: Exact brute-force search using L2 distance. This method is the slowest but most consistent in query time, guaranteeing 100% accuracy as it performs exhaustive search across all vectors. While reliable, it becomes impractical for large-scale deployments due to its linear time complexity.

- **IndexIVFFlat**: Inverted file system with flat quantizer. This approximate method trades some accuracy for significantly improved speed. It shows less consistent query times compared to FlatL2 but provides controllable speed-accuracy trade-offs through parameter tuning. Crucially, IVFFlat supports dynamic updates and deletions without requiring complete index reconstruction.

- **IndexHNSWFlat**: Hierarchical Navigable Small World implementation, excelling in high-recall scenarios. While providing excellent query performance, HNSW requires complete index rebuilding for updates or deletions, making it unsuitable for dynamic network monitoring scenarios.

**Performance Characteristics**: Our evaluation revealed distinct performance patterns across index types. FAISS-FlatL2 demonstrated the slowest but most consistent query times due to its brute-force approach. All approximate methods (IVFFlat, HNSW, IVFPQ) delivered faster performance but with less consistent timing and high CPU resource utilization, making them CPU-bound operations. **should graphs be added??**

**Index Selection for Network Data**: After comprehensive evaluation, we selected IndexIVFFlat as our primary index for IP flow embeddings. This decision was driven by its unique capability to handle updates and deletions without requiring complete index reconstruction- a critical requirement for dynamic network monitoring where flow records are continuously added and expired flows need removal. Unlike HNSW and FlatL2, which require complete reindexing for any modifications, IVFFlat provides the operational flexibility essential for production network environments.

**Network Data Suitability**: FAISS's batch-oriented design aligns well with network flow analysis patterns for the bulk processing of flow records. Its extensive index variety allows fine-tuning for specific network monitoring use cases, from real-time anomaly detection to large-scale forensic analysis. The library's mature optimization makes it particularly suitable for high-throughput network monitoring applications.

### 4b. Milvus

### 4c. Qdrant


### 4d. Weaviate


### 4e. Chroma


### 4f. PgVector


---

## 5. Benchmark Methodology

### 5.1 Testing Environment

All evaluations were conducted on standardized hardware configurations to ensure fair comparison: **fill in the specifics of the common machine**

### 5.2 Dataset Configuration


### 5.3 Evaluation Metrics

---

## 6. Results

### 6.1 Performance Summary

| Database | Indexing Time (min) | CPU Usage (%) | Avg Query Latency (ms) | RAM Usage (GB) | Disk Size (GB) | Recall@10 |
|----------|-------------------|---------------|----------------------|---------------|---------------|-----------|
| FAISS    | ___              | ___            | ___                  | ___           | ___           | ___      |
| Qdrant   | ___              | ___            | ___                  | ___           | ___           | ___      |
| Milvus   | ___              | ___            | ___                  | ___           | ___           | ___      |
| Weaviate | ___              | ___            | ___                  | ___           | ___           | ___      |
| Chroma   | ___              | ___            | ___                  | ___           | ___           | ___      |
| PgVector | ___              | ___            | ___                  | ___           | ___           | ___      |

### 6.2 Embedding Model Performance

| Model | Best Database | Avg Similarity Score | Use Case Recommendation |
|-------|---------------|---------------------|------------------------|
| paraphrase-MiniLM-v2 | ___ | ___ | ___ |
| MiniLM-L6-v2 | ___ | ___ | ___ |
| bert-base-nli-mean-tokens | ___ | ___ | ___ |
| mpnet-base-v2 | ___ | ___ | ___ |
| microsoft-codebert-base | ___ | ___ | ___ |
| average_word_embeddings_komn | ___ | ___ | ___ |
| ip2vec | ___ | ___ | ___ |

---

## 7. Analysis and Discussion

### 7.1 Performance Characteristics


### 7.2 Network Data Insights

---

## 8. Recommendations

| Use Case | Recommended DB | Embedding Model | Why |
|----------|----------------|-----------------|-----|
|  | ___ | ___ | ___ |
|  | ___ | ___ | ___ |
|  | ___ | ___ | ___ |
|  | ___ | ___ | ___ |
|  | ___ | ___ | ___ |
|  | ___ | ___ | ___ |

---

## 9. Limitations

---

## 10. Future Work


---

## 11. Conclusion

---

## 12. Appendix

### A. Implementation Scripts


### B. Configuration Details


### C. Raw Performance Data

### D. Embedding Analysis
