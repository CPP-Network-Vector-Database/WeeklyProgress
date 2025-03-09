## Trying Different Embeddings
I took the BERT based models for testing out different embeddings and used a [dataset that has extracted pcap data into a csv](https://www.kaggle.com/datasets/namitaachyuthanpesu/pcap-2019-dira-125910)

#### Input: 
- [CSV with IP flow](https://www.kaggle.com/datasets/namitaachyuthanpesu/pcap-2019-dira-125910)

#### Output:
- packet_embeddings.index
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
- deleting 50 random packets
    - Problem!!! FAISS does not support direct deletion of individual embeddings.
    - Workaround:
        - Remove entries from the metadata CSV.
        - Rebuild the FAISS index without the deleted embeddings.
- finding 50 packets and randomly updating it
      - Similar to delete + insert.
      - Just remove old embeddings from FAISS, recompute new ones, and reinsert.

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
