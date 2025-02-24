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

## CPP Meeting 3 - 21/02/25

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
