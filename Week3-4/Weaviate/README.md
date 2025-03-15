
## Tasks Carried Out

- **Obtained vector sizes:**  
  Collected and compared vector dimensions for different embedding models.
  
- **Tried different embedding models:**  
  Tested several embedding models with Weaviate to evaluate their performance and semantic response quality.

- **Interpreted CPU utilization:**  
  Tried to interpret cpu utilization better by considering cores of the cpu and a shorter timeframe to capture cpu utilization. (worked out sometimes but still shows 0% for couple of the embeddings models)

- **Benchmarked embedding models:**  
  Executed benchmark tests using the same query and document across all models to measure:
  - Memory usage (percentage and in MB)
  - Total system CPU usage (adjusted for multiple cores)
  - Query duration (in seconds)
  
## Embedding Model Performance Table

| Embedding Model            | Memory Usage (in %) | Memory Usage (in MB) | Total System CPU Usage (in %) | Query Duration (in seconds) | Vector Dimensions | Remark                                                                      |
|----------------------------|---------------------|----------------------|-------------------------------|-----------------------------|-------------------|-----------------------------------------------------------------------------|
| all-MiniLM-L6-v2           | 17.18               | 672.66               | 1                             | 0.0226                      | 384               |                                                                             |
| all-MiniLM-L12-v2          | 18.3                | 716.15               | 1                             | 0.0484                      | 384               | More semantically meaningful responses but slightly more resource intensive |
| all-distilroberta-v1       | 21.03               | 818.73               | 0                             | 0.063                       | 768               |                                                                             |
| multi-qa-MiniLM-L6-cos-v1   | 17.1                | 669                  | 0                             | 0.0235                      | 384               | Very similar to L6-v2                                                       |
| paraphrase-MiniLM-L3-v2    | 16.58               | 649.34               | 1.95                          | 0.0181                      | 384               | Lightest model from sentence-transformers                                   |
| e5-small-v2                | 18.27               | 715.37               | 0.9                           | 0.1161                      | 384               | Model optimized for search and retrieval (for semantic queries)             |

## Observations

- **E5 models:**  
  E5 models ("Embedding for Everything") are embedding models specifically for retrieval, ranking, and semantic search tasks.

- **CPU Utilization:**  
  Considering the number of cores seems to make a more clear picture of cpu utilization but it still comes as zero for certain models. Might have to consider graph plotting to notice spikes in usage for better interpretation or use a larger dataset(many queries at once) for a noticeable difference.

- **Query Duration:**  
  the MiniLM-L12 seems to give more semantically appropriate responses but at a slightly higher resouce cost. Will have to consider this tradeoff while trying queries on IP flows.

- **384 embed size:**  
  this vector size seems to be constant across the lightest variants of all models. This is apparently due to the fact that these models are compressed versions of the transformer architectures. They just reduce the number of hidden layers and attention heads.
  

## Additional Progress

- **GPU Usage in VirtualBox:** VirtualBox does not provide direct GPU access to the virtual machine (VM). Even though the host machine has a GPU, the VM will not automatically use it unless configured for GPU passthrough. So we can be sure that the GPU is not utilized while creating the embeddings.  

- **CPU Utilization Monitoring:** Interpreted CPU utilization by considering cores and created a separate daemon to monitor CPU usage during document ingestion and query processing.  

- **Visualization:** Used Matplotlib to plot CPU usage logs captured by the daemon.  

- **Performance Metrics:** Captured the time taken to ingest a document.  

- **Benchmarking:** Added a feature to the CLI to benchmark multiple queries at once, allowing for an average of recorded metrics to provide a clearer picture of compute utilization.  


---

