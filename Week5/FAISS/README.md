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
