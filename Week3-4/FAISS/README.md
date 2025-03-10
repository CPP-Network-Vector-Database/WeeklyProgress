## Trying Different Embeddings
The embedding models used were all BERT based, as that is what I had communicated to the team that I'd take. 

#### Input: 
- [CSV with IP flow](https://www.kaggle.com/datasets/namitaachyuthanpesu/pcap-2019-dira-125910)

#### Embedding models: 
- [**paraphrase-MiniLM-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)- Chosen for efficient semantic similarity.  
- [**MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)- Selected for balanced speed and accuracy.  
- [**bert-base-nli-mean-tokens**](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)- Picked for strong natural language inference.  
- [**mpnet-base-v2**](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)- Used for capturing contextual sentence meaning.  
- [**microsoft-codebert-base**](https://huggingface.co/microsoft/codebert-base)- Ideal for code understanding and embeddings.  
- [**average_word_embeddings_komn**](https://huggingface.co/sentence-transformers/average_word_embeddings_komninos)- Preferred for lightweight word representations.
- [**ip2vec**](https://github.com/DavidHarar/IP2Vec)- using word2vec as base (yet to update table)

#### Output:
- packet_embeddings.index
        - FAISS index contains 1653188 embeddings
- packet_metadata.csv
- Performance results

#### The actions tested were: 
- querying a random packet (to get the top k neighbors)
    - Convert the query packet into text format.
    - Generate its BERT embedding.
    - Normalize the embedding (since FAISS works best with normalized vectors).
    - Search the FAISS index for the k-nearest neighbors.
    - Return the top-k results with their distances (lower = more similar).
- inserting 50 random new packets
    - Convert 50+ new packets into embeddings.
    - Normalize them.
    - Add to the FAISS index.
-  deleting 50 random packets
    - Problem!!! FAISS does not support direct deletion of individual embeddings.
    - Workaround:
        - Remove entries from the metadata CSV.
        - Rebuild the FAISS index without the deleted embeddings.
- finding 50 packets and randomly updating it
    - Similar to delete + insert.
    - Just remove old embeddings from FAISS, recompute new ones, and reinsert.
 
#### Results
| Embedding Model                   | Embedding Size   | Action                   | CPU Usage | Memory Usage  | Time Taken  | Note  |
|------------------------------------|------------------|--------------------------|-----------|--------------|-------------|------------------------------------------------------|
| paraphrase-MiniLM-v2              | 384 dimensions   | Query new packet         | 98.50%    | 0.25MB       | 0.2331s     | Fastest model with lowest memory usage. Suitable for large-scale queries. |
|                                    |                  | Add 50 new packets       | 100.10%   | 0.15MB       | 2.6068s     | |
|                                    |                  | Delete 50 packets        | 100.20%   | 4843.25MB    | 3.3047s     | |
|                                    |                  | Update 50 new packets    | 100.30%   | 4843.15MB    | 6.8423s     | |
| MiniLM-L6-v2                      | 384 dimensions   | Query new packet         | 97.00%    | 0.25MB       | 0.2160s     | Similar to paraphrase-MiniLM-v2, slightly better in update operations. Good trade-off between speed and memory. |
|                                    |                  | Add 50 new packets       | 99.90%    | 0.19MB       | 2.4035s     | |
|                                    |                  | Delete 50 packets        | 100.30%   | 4843.21MB    | 3.1418s     | |
|                                    |                  | Update 50 new packets    | 100.20%   | 4843.37MB    | 6.4888s     | |
| bert-base-nli-mean-tokens         | 768 dimensions   | Query new packet         | 100.80%   | 0.25MB       | 0.4665s     | Kernel crashes under heavy operations. High memory usage. Not ideal for large-scale data. |
|                                    |                  | Add 50 new packets       | 100.00%   | 130.85MB     | 5.0796s     | |
|                                    |                  | Delete 50 packets        | 100.00%   | 9667.32MB    | 6.7680s     | |
|                                    |                  | Update 50 new packets    | Kernel Crashed multiple times | | | |
| Kernel Crashed multiple times      | 768 dimensions   | Query new packet         | 91.50%    | 0.12MB       | 0.4592s     | High memory consumption. Frequent crashes when updating packets. |
|                                    |                  | Add 50 new packets       | 99.90%    | 53.38MB      | 4.9438s     | |
|                                    |                  | Delete 50 packets        | 100.20%   | 9540.68MB    | 6.4405s     | |
|                                    |                  | Update 50 new packets    | Kernel Crashed multiple times | | | |
| mpnet-base-v2                     | 768 dimensions   | Query new packet         | 100.10%   | 0.25MB       | 0.4495s     | High accuracy but slow. Kernel crashes in updates. Significant memory overhead. |
|                                    |                  | Add 50 new packets       | 100.00%   | 158.05MB     | 4.8790s     | |
|                                    |                  | Delete 50 packets        | 99.90%    | 9531.78MB    | 9.7175s     | |
|                                    |                  | Update 50 new packets    | Kernel Crashed multiple times | | | |
| microsoft-codebert-base            | 768 dimensions   | Query new packet         | 101.20%   | 0.25MB       | 0.4247s     | Heavy memory usage and crashes under updates, but performs well in querying. |
|                                    |                  | Add 50 new packets       | 100.20%   | 180.59MB     | 5.0016s     | |
|                                    |                  | Delete 50 packets        | 100.00%   | 9666.62MB    | 6.6994s     | |
|                                    |                  | Update 50 new packets    | Kernel Crashed multiple times | | | |
| average_word_embeddings_komn       | 768 dimensions   | Query new packet         | 99.10%    | 0.25MB       | 0.1808s     | Fastest for querying & adding packets, lowest memory footprint. Best for low-resource environments. |
|                                    |                  | Add 50 new packets       | 100.00%   | 0.11MB       | 1.8612s     | |
|                                    |                  | Delete 50 packets        | 100.10%   | 3783.74MB    | 2.4665s     | |
|                                    |                  | Update 50 new packets    | 100.10%   | 3783.66MB    | 5.0252s     | |


#### Points to focus on: 
- FAISS is NOT a vector database, it's a library used with vector databases for easy indexing and retrieval
- Last time I'd used FAISS with milvus for the different indexing methods, but for this, I wanted to see how FAISS perfroms as a standalone library
- FAISS doesn't support deletion, so what I did was:
    - remove the entries from the embeddings csv
    - rebuild the faiss index without those entries
- BERT embeddings need normalization
 
  
### Fixing the old issue of negative CPU and Memory usage
tl;dr: pick up the PID of the process and only use psutils for that rather than making a measure of the whole system

This measure function helps accurately track CPU, memory usage, and execution time for any function. Previously, negative values appeared due to incorrect timing in resource measurement. 

#### How it works:  
1. Captures CPU and memory usage before running the function.  
2. Runs the function with the given arguments.  
3. Captures CPU and memory usage again after execution.  
4. Calculates the difference to get actual resource usage.  

#### Why this fixes the issue: 
- `cpu_percent(interval=None)` ensures we don’t get misleading CPU readings.  
- Memory is tracked correctly in MB, avoiding fluctuations.  
- Execution time is measured precisely.  
