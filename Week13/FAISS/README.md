tl;dr
### Work done this week
-  UI demo related
    - Added user option to choose model
    - updated metrics table that can be exported as a csv
    - Updated code is in the deployment repository: https://github.com/CPP-Network-Vector-Database/Deployment/tree/main/FAISS/UI-Demo
- Added documentation for FAISS for final report: https://github.com/CPP-Network-Vector-Database/WeeklyProgress/blob/main/DraftReport.md
- Set up rough pipeline for all vector DBs (except FAISS) with Langtrace: https://github.com/CPP-Network-Vector-Database/Deployment/tree/main/CRUD-ops

### Issues that need attention
- FAISS and langtrace don't work together- how to look at the observability for FAISS
- How to export finetuned models for other vector databases
    - was working on the demo this week so couldn't fully focus on the finetuned embedding models
 
---

### The HNSW Problem with FAISS
The fundamental issue with HNSW and FAISS comes from the architectural limitations that make dynamic operations messy
- How it works: 
  - HNSW creates an intricate graph-like structure where vectors are interconnected through a web of relationships
  - Each vector maintains connections to its neighbors, forming the backbone of the search mechanism
  - This connectivity is essential for the algorithm's efficiency but creates rigidity in the structure

- Why updates and delted don't work:
  - HNSW's heavy reliance on interconnected relationships means vectors cannot be updated or deleted without potentially breaking the entire graph structure
  - According to the [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), removing a node (vector) or changing its position would require costly restructuring of the entire graph
  - FAISS does not implement dynamic restructuring capabilities for HNSW indexes

- How we *could* solve it?
  - The only method to "delete"/"update" data is to rebuild the entire index from scratch without the unwanted vectors
  - This approach was tested with flat indexing in earlier implementations and resulted in significant latency issues
  - For use cases requiring frequent updates or deletions (such as network admin operations), this approach becomes computationally prohibitive
  - The metadata overhead continues to increase with each rebuild operation

#### How other vector DBs handle the updates/deletions
I tried to read about how other vector DBs are able to handle this, since FAISS clearly has an issue with it: 
- pgvector
    - pgvector implements a custom HNSW solution that supports updates and deletions through standard SQL operations
    - When UPDATE or DELETE operations are performed, pgvector manages the necessary modifications to the HNSW index structure
    - The system handles graph restructuring without requiring complete index rebuilds
    - Reference: [AWS Aurora PostgreSQL pgvector announcement](https://aws.amazon.com/about-aws/whats-new/2023/10/amazon-aurora-postgresql-pgvector-v0-5-0-hnsw-indexing/)

- Qdrant
  - Qdrant utilizes a segmented architecture that separates data into mutable and immutable segments
  - Updates are applied to mutable segments, maintaining flexibility for dynamic operations
  - Once segments reach a predetermined size threshold, they transition to immutable status
  - Reference: [Qdrant dedicated vector search article](https://qdrant.tech/articles/dedicated-vector-search/)

- Weaviate
  - Weaviate uses a custom HNSW implementation with immutable document ID assignment
  - Updates are handled by deleting the old vector and inserting a new one
  - Deletions utilize a tombstoning approach where vectors are marked as deleted rather than immediately removed
  - Asynchronous bulk cleanup operations handle the actual removal from the index
  - Reference: [DB-Engines blog post](https://db-engines.com/en/blog_post/87)

- FAISS
  - FAISS HNSW implementation lacks support for updates or deletions due to design constraints
  - While highly efficient for read-heavy workloads, FAISS requires complete index rebuilding to reflect data changes
  - This limitation makes FAISS unsuitable for applications requiring frequent dynamic operations

### Benchmarking challenges
- Recall rate limitation (with unlabelled data)
  - Recall rate is calculated as the intersection of true nearest neighbors and retrieved nearest neighbors, divided by the true nearest neighbors
  - This metric requires knowledge of the actual nearest neighbors for any given query (ground truth0
  - With unlabeled datasets, determining true nearest neighbors becomes impossible
  - Without ground truth data, recall rate cannot be computed accurately, making it unsuitable for our benchmarking methodology

- Langtrace observability doesn't work with FAISS
  - FAISS is fundamentally designed as a library for efficient similarity search and clustering of dense vectors
  - The system provides algorithms for searching large-scale datasets that may exceed RAM capacity
  - FAISS does not provide persistent storage capabilities for vectors
  - This ephemeral nature creates challenges for observability tools like Langtrace to effectively monitor and hook into FAISS operations

### Hybrid db considerations (??)
Would combining FAISS with another vector DB be the way through this? 
- Integrating FAISS with systems like Weaviate or Milvus would not provide accurate performance metrics for FAISS in isolation
- Such combinations would benchmark FAISS behavior when wrapped and managed by another system
- Multiple layers of abstraction and additional logic would be introduced
- Performance results would reflect the hybrid system rather than pure FAISS capabilities
- This approach would compromise the ability to understand FAISS-specific performance characteristics
