


# Week 2 Progress


## Overview

- **Schema Creation:**  
  Created a Weaviate class called `DocumentChunk` with a `content` property to store text chunks.
  
- **Document Ingestion:**  
  Extract text from a PDF document, split the text into chunks, generate vector embeddings using a SentenceTransformer model, and insert these chunks into Weaviate.

- **Semantic Querying:**  
  Convert a user's query into a vector and perform a semantic search against the ingested document chunks.

- **Benchmarking:**  
  Measured query performance including execution time, CPU usage, and memory consumption.

- **Database Reset:**  
  Delete the entire schema and all associated objects from Weaviate to reset the database.

## File Descriptions

### 1. `schema.py`
- **Purpose:**  
  Defines and creates the `DocumentChunk` class in the Weaviate schema with a `content` property.
  
- **Key Points:**
  - Checks if the class already exists before creating it.
  - Sets the vectorizer to `"none"` since embeddings are provided externally.
  


### 2. `ingest.py`
- **Purpose:**  
  Provides functions to:
  - Extract text from a PDF using PyPDF2.
  - Split the text into smaller chunks.
  - Generate embeddings for each chunk with SentenceTransformer.
  - Insert the chunks along with their embeddings into the Weaviate vector DB.
  


### 3. `query.py`
- **Purpose:**  
  Contains the logic to:
  - Encode a query string into a vector.
  - Execute a semantic search against the `DocumentChunk` class in Weaviate.
  - Return the most relevant document chunks based on the query.
  
### 4. `benchmark.py`
- **Functionalities:**
    - **Query Execution Duration:**  
    The script uses Python's built-in `time` module to record the start and end times of a query execution. The difference between these times gives you the duration of the query.

    - **CPU Usage Measurement:**  
    Using the `psutil` module, the script calculates the CPU percentage used during the query execution. This provides insight into how much processing power is consumed when running queries.

    - **Memory Usage Measurement:**  
    The script also measures the memory consumption of the current process  in megabytes (MB) using `psutil` to report the memory consumption of the query.
   

### 5. `delete.py`
- **Purpose:**  
  Deletes all schema classes and their associated objects from the Weaviate instance to reset the vector database.
  
- **Key Points:**
  - Includes a confirmation prompt to prevent accidental deletion.
  

### 6. `main.py`
- **Purpose:**  
  Acts as the command-line interface (CLI)
  
- **Functionality:**
  - **Ingest Documents:**  
    Example:  
    ```bash
    python main.py ingest /path/to/document.pdf
    ```
  - **Query the Database:**  
    Example:  
    ```bash
    python main.py query "search query"
    ```
  - **Benchmark Queries:**  
    Example:  
    ```bash
    python main.py benchmark "search query"
    ```
  

Dependencies:
```bash
pip install weaviate-client sentence-transformers PyPDF2 psutil
```


## Additional Notes

- The vectorizer for the `DocumentChunk` class is set to `"none"` since embeddings are generated externally using the SentenceTransformer model.
- The functionalities are modularized into separate files for maintainability and ease of extension.
- The CLI interface provided by `main.py` ties all the functionalities together.
