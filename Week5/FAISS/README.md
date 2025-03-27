This week, my AAIs were: 
- Work with other vector databases and FAISS
- Test updates after large insertions
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
The reason this approach was used was because the flatL2 index was being used to store the data. Making the update and delete functions more efficient allowed for more insertions, updates, adn deletions without kernel crashes. 
I believe this is because the brute force approach, though fast, introduces unnecessary overhead when it comes to updates and deletions, but IVFFlatL2, though slower than FlatL2, can perform better in CRUD operations because it **partitions the dataset into clusters** (nlist), **reducing the number of comparisons** needed for search and updates. 

**Instead of modifying the entire dataset during insertions, deletions, or updates, IVFFlatL2 only operates on the relevant clusters**, making these operations more efficient. 
This significantly reduces memory overhead and computational load, preventing kernel crashes and allowing me to work with MUCH mroe than 50 at a time. 
