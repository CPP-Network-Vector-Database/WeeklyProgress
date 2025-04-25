### Putting together a model pipeline + querying after variable number of operations
The pipeline was written to evaluate (more comprehensively) how different sentence-transformer models behave under insertions, deletions, updates, and queries when plugged into a FAISS index. 

Basically, for each model, I ran a full performance suite across different operation sizes- 50, 100, 250, 500, and 1000 ops- and tracked how long each operation took, how much memory it used, and what kind of CPU load it introduced.

This time, I didn’t just run the ops and measure insert/delete/update timings like before, I also ran queries immediately after each of those operations, and logged their metrics as well. That helped capture how the FAISS index responds to incremental structural changes over time. In earlier versions I only tested query times before and after everything, but now it’s done per step- much more granular and informative.
> one mistake I made here, while plotting was that the graph for the queries is a bit wonky- it doesn't move across the graph as the number of operations changes. I got confused with how many plots there were and didn't realize this until after all the models were trained. Each model takes quite a bit of time, so I'll fix this by next week and then run the suite again

The CSV file was first cleaned and converted into a textual format by joining IPs, ports, protocol, and length fields into a single string- one line per flow, as we've been doing. These strings were then embedded using sentence-transformers (7 models tested in total). The embeddings were normalized so cosine similarity could be approximated through L2 distance (since FAISS doesn’t support cosine directly unless everything is pre-normalized).

A FAISS IndexIVFFlat was used for each model. Once trained, it was filled with the embeddings, and operations began. For insertion, synthetic packet text flows were generated to simulate new data. For deletions and updates, random entries were selected from the existing index. The update operation first removed embeddings and then re-added new ones in their place, essentially mimicking a replace.

One new thing I added this time was a cosine similarity check (as was suggested earlier) after the 1000th operation batch. After all insert/update/delete/query rounds, I manually queried with a specific flow ("192.168.1.1 192.168.1.2 TCP 100") and recorded the top-5 nearest neighbors and their cosine similarity scores. This gives a sanity check for how well the index is maintaining meaningful structure even after a large number of modifications.

Each model's entire run generated a set of performance metrics, and all of them were plotted in one go- time, CPU, memory -across each batch size for each operation type. The plots are saved in the PipelineResults/ folder, named after the model.

> Another thing I noticed was that when I ran the pipeline on the GPU it took 10(ish) minutes compared to the ~40 minutes that the embeddings took on the CPU. 
> Does this mean we have to look into optimizing for the GPU? Or since we're looking at this being more on the edge, should we just work with the CPU? 

#### Cosine Similarity metrics per model: 
| Model Name                         | Average Cosine Similarity |
|-----------------------------------|-------------------|
| paraphrase-MiniLM-L12-v2         | 0.0129            |
| all-MiniLM-L6-v2                 | 0.1636            |
| distilbert-base-nli-stsb-mean-tokens | 0.1393        |
| microsoft/codebert-base          | 0.0016            |
| bert-base-nli-mean-tokens        | 0.1744            |
| average_word_embeddings_komninos | 4.31e-16          |
| all-mpnet-base-v2                | 0.0703            |



Key takeaways: 
* We can see that all the top-5 neighbors are super close in index values- this is the beauty of ivfflat!!
    - It clusters the vector space into "nlist" partitions, and only searches the top clusters closest to the query vector (based on a coarse quantizer).  
    - So when the index is already populated with a huge number of flows, and you query with a normalized embedding, FAISS looks inside a narrow cluster, which often contains very close index IDs- divide and conquer
    - This behavior is actually what makes it fast- we trade off a tiny bit of accuracy (recall) for speed by not searching the entire space, just the "best clusters."

* Interestingly, average word embeddings returned the closest neighbors in terms of cosine similarity (almost identical values: ~4e-16 for all 5 results).  
    - One reason could be that this model is super shallow and just averages word vectors- so the resulting embeddings fall into a very tight region in vector space.  
    - That makes them almost indistinguishable when normalized, so even small changes in query result in very little change in cosine distance.  
    - Basically: less expressive embeddings = less spread = everything looks "close."

* On the flip side, all-MiniLM gave the farthest cosine similarities among all models- top scores were only around 0.16-0.20 
    - Probably because these deeper transformer-based models capture finer-grained differences between flows (maybe based on token structure, ordering, or internal attention heads).  
    - So even if the flows look similar at a surface level, the model embeds them in well-separated clusters, hence the cosine similarity stays low.  
