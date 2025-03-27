This week, my AAIs were: 
- Work with other vector databases and FAISS
- Make the update and deletion functions more efficient
- Work on finetuning bert for ip flow embeddings generation

### Work with BERT finetuning
- Have done a thorough literature survey of custom made embedding models for IP flows
- learnt the theory for embedding generation through bert
- Have worked on a basic implementation of embedding generation with base bert model (uncased)

### Making the update and deletion functions more efficient
#### Initial approach:
-  deleting 50 random packets
    - Problem!!! FAISS does not support direct deletion of individual embeddings.
    - Workaround:
        - Remove entries from the metadata CSV.
        - Rebuild the FAISS index without the deleted embeddings.
- finding 50 packets and randomly updating it
    - Similar to delete + insert.
    - Just remove old embeddings from FAISS, recompute new ones, and reinsert.

#### Updated approach:  
The reason this approach was used was because the FlatL2 index was initially used to store the data. While FlatL2 provides fast brute-force similarity search, it introduces unnecessary overhead when performing updates and deletions. This led to kernel crashes when handling larger-scale modifications.  

By switching to **IVFFlatL2**, we improved the efficiency of insertions, updates, and deletions. IVFFlatL2 partitions the dataset into clusters (**nlist**), which reduces the number of comparisons needed for search and update operations.  

##### Key improvements w IVFFlatL2:  
- Instead of modifying the entire dataset during insertions, deletions, or updates, IVFFlatL2 only operates on the relevant clusters.  
- Since search and modifications happen within specific clusters, the computational load is significantly lower than FlatL2.  
- This method prevents kernel crashes and allows batch operations on **hundreds or even thousands** of embeddings instead of just 50 at a time.  

### Results
| Embedding Model                      | Embedding Size | Action                  | CPU Usage  | Memory Usage | Time Taken  |
|--------------------------------------|---------------|-------------------------|------------|--------------|-------------|
| paraphrase-MiniLM-v2                 | 384 dimensions | Query new packet       | 97.80%     | 0.01MB       | 0.0019s     |
|                                       |               | Add 50 new packets      | 99.20%     | 0.05MB       | 0.0010s     |
|                                       |               | Delete 50 packets       | 270.70%    | 0.11MB       | 0.0041s     |
|                                       |               | Update 50 new packets   | 126.90%    | 0.12MB       | 0.1266s     |
| MiniLM-L6-v2                          | 384 dimensions | Query new packet       | 98.20%     | 0.25MB       | 0.0024s     |
|                                       |               | Add 50 new packets      | 99.60%     | 0.38MB       | 0.0036s     |
|                                       |               | Delete 50 packets       | 100.30%    | 0.38MB    | 0.0067s     |
|                                       |               | Update 50 new packets   | 156.0%    | 98.0MB    | 0.0389s     |
| bert-base-nli-mean-tokens             | 768 dimensions | Query new packet       | 228.40%    | 0.04MB       | 0.0044s     |
|                                       |               | Add 50 new packets      | 101.10%    | 0.12MB       | 0.0014s     |
|                                       |               | Delete 50 packets       | 291.20%    | 0.15MB       | 0.0037s     |
|                                       |               | Update 50 new packets   | 113.10%    | 0.12MB       | 0.2922s     |
| distilbert                            | 768 dimensions | Query new packet       | 99.10%     | 0.00MB       | 0.0039s     |
|                                       |               | Add 50 new packets      | 111.20%    | 0.25MB       | 0.0016s     |
|                                       |               | Delete 50 packets       | 119.60%    | 0.17MB       | 0.0055s     |
|                                       |               | Update 50 new packets   | 117.50%    | 0.12MB       | 0.1877s     |
| mpnet-base-v2                         | 768 dimensions | Query new packet       | 195.70%    | 0.00MB       | 0.0051s     |
|                                       |               | Add 50 new packets      | 110.20%    | 0.12MB       | 0.0016s     |
|                                       |               | Delete 50 packets       | 119.30%    | 0.00MB       | 0.0045s     |
|                                       |               | Update 50 new packets   | 112.70%    | 0.12MB       | 0.3021s     |
| microsoft-codebert-base               | 768 dimensions | Query new packet       | 172.50%    | 0.12MB       | 0.0175s     |
|                                       |               | Add 50 new packets      | 164.30%    | 0.75MB       | 0.0019s     |
|                                       |               | Delete 50 packets       | 170.20%    | 0.38MB       | 0.0061s     |
|                                       |               | Update 50 new packets   | 113.60%    | 0.38MB       | 0.3261s     |
| average_word_embeddings_komn          | 768 dimensions | Query new packet       | 101.30%    | 0.00MB       | 0.0004s     |
|                                       |               | Add 50 new packets      | 119.20%    | 0.00MB       | 0.0013s     |
|                                       |               | Delete 50 packets       | 242.20%    | 0.99MB       | 0.0045s     |
|                                       |               | Update 50 new packets   | 173.40%    | 0.00MB       | 0.0293s     |

