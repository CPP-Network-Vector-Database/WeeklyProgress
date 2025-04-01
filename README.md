# WeeklyProgress

## CPP Meeting 1: 07/02/25
Everyone picks one open source database:
| **Person**   | **Database**  |
|--------------|---------------|
| Anshu        | Weaviate      |
| Vishal       | ChromaDB      |
| Ajay         | PgVector      |
| Namita       | FAISS         |
| Varshini     | Milvus        |
| Smera        | Qdrant        |


Once the database is picked: (to be done by next meeting)
* Read about the database chosen
* Get python client library
* Install database and client library into VM
- Write script:
  - Connect API (connect to database)
  - Get a successful connection object
  - Load some data- some random data- not necessarily IP flow
  - Run some queries and make sure you're getting the intended output
- Linux VM works well
- Will need to go through the API manual for the client
- To read about: workflows- making queries and working with data- not just loading it

## CPP Meeting 2: 14/02/25
AAIs for next week:
- Create a repo in github for the project: folders for weekly progress
    - Each week gets a readme: captures all the AAIs for the week
- Start with a large document, progress with benchmarking script (many page doc-> feed into db)
- How long is a query taking? CPU percentage? Memory percentage?
- Try to make the script output the benchmark numbers- start with these though
- For next week build a UI(CLI) to query
- Implement how data flow embeddings are placed in database. Establish use case for applications

## CPP Meeting 3: 21/02/25

#### Feedback:
- **Anshu:**
  - Vector dimensions for the created embeddings
  - Figure out how to obtain meaningful interpretation for CPU usage
  - Have different embeddings generators to embed and benchmark them
- **Vishal:**
  - More records in the database
  - Try benchmarking on more queries
- **Ajay:**
  - Have custom scripts for benchmarking and compare that with standard benchmarking tools
- **Namita:**
  - Understand how psutil works for the negative memory and CPU usage data
  - How brute force works better than other indexing methods
- **Varshini:**
  - Add more queries
  - Plot graphs
- **Smera:**
  - Remove all unnecessary API calls to reduce overhead
  - Visualize the metrics obtained

#### Team AAIs:
- Work with different embeddings- not just sentenceTransformers
  - Each person takes 3-4 unique ones
- Try measuring CPU usage, memory, and latency with different embeddings and interpret results
- More operations on data- not just loading into database
  - CRUD operations
  - Impact of these on query performance
  - What would large updates/deletions/insertions do to the queries?

#### Next meetings:
- **March 4th update via mail:**
  - No meeting that week
  - Post-mail update: feedback + next set of AAIs
- **Online meeting on March 19th (Wednesday)**

## CPP Meeting 4: 14/03/25  

### Feedback  

| **Person**  | **Feedback**  |
|------------|--------------|
| **Anshu**  | - Work with IP flows instead of documents  <br> - Baseline for memory usage (to record delta)  <br> - Record throughput  <br> - CRUD operations  |
| **Vishal** | - Check if updation affects retrieval  <br> - Optimize updation and deletion  |
| **Ajay**   | None  |
| **Namita** | - Check metrics for queries after large updates and deletions  <br> - Work with Milvus/Weaviate and FAISS  <br> - Optimize the update and delete mechanisms  <br> - *(Additional)* Work with fine-tuning BERT for IP Flow embeddings  |
| **Varshini** | None  |
| **Smera**  | - Implement CRUD for different embeddings  <br> - Look into Qdrant config  <br> - Figure out why latency is so high  |

#### Team AAIs  
- Compile all previous week results in a single table  
- Work with throughput calculation as an additional metric  

#### Next Meetings  
- **March 21st update via mail**  
  - No meeting that week
  - Post-mail update: Feedback + next set of AAIs

## CPP Meeting 5: 21/03/25  

### Feedback  

| **Person**  | **Feedback**  |
|------------|--------------|
| **Anshu**  | - Display similarity scores  <br> - Explore indexing methods  |
| **Vishal** | - Explore different values for `max_seq_length`, `num_workers`, `batch_size`  <br> - Use other efficient methods to track memory  <br> - Store source and destination port  <br> - Try different similarity methods  |
| **Ajay**   | - Explore various Vector Indexing Methods  <br> - Improve insertion operation performance by introducing multithreading  |
| **Namita** | - Push `.ipynb` code to `.py`  <br> - Work with Milvus/Weaviate and FAISS  <br> - Finish fine-tuning BERT for IP Flow embeddings  |
| **Varshini** | - Print distance similarity score  <br> - Correct the metrics  |
| **Smera**  | None  |  

## CPP Meeting 6: 28/03/25  

### Feedback  

| **Person**  | **Feedback**  |
|------------|--------------|
| **Anshu**  | - Tabularize and plot results  <br> - Try to implement a more efficient HNSW  |
| **Vishal** | - Perform CRUD operations for large batches  <br> - Tabulate and plot the results  |
| **Ajay**   | - Tabulate the results  <br> - Perform vector operations and measure query latency  <br> - Work on the indexing mechanism  <br> - Explore an embedding model specialized for 5-tuple network packets  |
| **Namita** | - Make use of similarity metrics to measure accuracy  <br> - Make a custom tokenizer for better fine-tuning of BERT  |
| **Varshini** | - Tabulate results  <br> - Explore different indexing methods  |
| **Smera**  | - Compute similarity scores  <br> - Implement the same on VM  <br> - Visualize the results  |  

### Team AAIs  
- Make use of a common dataset  
  - This should include the application name along with the IP  
