# start off with an import of the Milvus client
from pymilvus import MilvusClient
from pymilvus import model
import random  # For generating random vectors if we need it

# 1. Connect to Milvus
# So basically how it works is:
#    - Milvus Lite stores data in a local file.
#    - Then we have to instantiate a MilvusClient, specifying a file name to store all data.
#    - If the file exists, it will load the existing data.
#    - If the file doesn't exist, it will create a new database file.
MILVUS_DB_NAME = "bob_database.db"  # this is what we're naming the db
client = MilvusClient(MILVUS_DB_NAME)
print(f"Successfully connected to Milvus at {MILVUS_DB_NAME}")


# 2. Define Collection Parameters
#    - A collection is like a table in a relational database.
#    - It stores vectors and associated metadata.
COLLECTION_NAME = "bob_collection"
VECTOR_DIMENSION = 768  # Dimensionality of the vectors - basically how many scalars you've packed in. Starting with 768 cuz it's used for sentence embedding usually

# 3. Create a Collection
#    - Creates a collection if one doesn't already exist.
#    - Drops the collection first if it exists
if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")

client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=VECTOR_DIMENSION,
)
print(f"Collection '{COLLECTION_NAME}' created with dimension {VECTOR_DIMENSION}.")


# 4. Prepare Data
#    - Generate sample data to insert into the collection.
#    - We'll use text data and convert it into vectors using a pre-trained embedding model.

# Text data to be converted into vectors
DOCUMENTS = [
    "Snails have been around for over 500 million years, predating dinosaurs.",
    "Bees use a complex dance, the 'waggle dance', to communicate the location of food sources.",
    "The average garden snail can travel about 1.3 meters per hour.",
    "Bees are essential pollinators, responsible for pollinating approximately one-third of the food we eat.",
    "Some snails can hibernate for up to three years.",
    "A single honeybee will produce about 1/12th teaspoon of honey in its lifetime.",
]

# Create embeddings for the documents using a default embedding function
# install with: pip install "pymilvus[model]" - we've already done this though
embedding_fn = model.DefaultEmbeddingFunction()
vectors = embedding_fn.encode_documents(DOCUMENTS)

# Create a list of dictionaries, where each dictionary represents an entity.
# Each entity has an ID, a vector representation, the original text, and a subject label.
data = [
    {"id": i, "vector": vectors[i], "text": DOCUMENTS[i], "subject": "history" if i < 3 else "biology"}
    for i in range(len(DOCUMENTS))
]

print(f"Prepared {len(data)} entities for insertion.")


# 5. Insert Data- we'll insert the prepared data into the Milvus collection.
insert_result = client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"Data insertion result: {insert_result}")

# 6. Perform Semantic Search- we convert the search query into a vector and search for similar vectors in the collection.
SEARCH_QUERY = "How fast can garden snails travel?"
query_vectors = embedding_fn.encode_queries([SEARCH_QUERY])

# Search the collection for the most similar vectors to the query vector.
search_results = client.search(
    collection_name=COLLECTION_NAME,  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(f"Search results for query '{SEARCH_QUERY}': {search_results}")


# 7. Vector Search with Metadata Filtering
#   - Search for vectors similar to the query vector, but only return results that match a specific metadata filter.
FILTER_QUERY = "tell me garden snail related information"
filter_vectors = embedding_fn.encode_queries([FILTER_QUERY])

# Perform a vector search with a metadata filter.
filtered_search_results = client.search(
    collection_name=COLLECTION_NAME,
    data=filter_vectors,
    filter="subject == 'waggle dance'",  # Filter results to only include entities with the subject "biology".
    limit=2,
    output_fields=["text", "subject"],
)

print(f"Filtered search results for query '{FILTER_QUERY}': {filtered_search_results}")


# 8. Query
#    - Retrieves all entities matching a criteria, such as a filter expression or matching some IDs.
query_results = client.query(
    collection_name=COLLECTION_NAME,
    filter="subject == 'honey'",
    output_fields=["text", "subject"],
)

print(f"Query results for entities with subject 'honey': {query_results}")

#    - Directly retrieve entities by primary key:
query_results_by_id = client.query(
    collection_name=COLLECTION_NAME,
    ids=[0, 2],
    output_fields=["vector", "text", "subject"],
)

print(f"Query results for entities with id 0 and 2: {query_results_by_id}")


# 9. Delete Entities- this is when you wanna delete entities from the collection based on their IDs or a filter expression
# here's we're deleting by primary key
delete_result = client.delete(collection_name=COLLECTION_NAME, ids=[0, 2])
print(f"Delete results for entities with id 0 and 2: {delete_result}")

# Delete entities by a filter expression
delete_result_by_filter = client.delete(
    collection_name=COLLECTION_NAME,
    filter="subject == 'honey'",
)
print(f"Delete results for entities with subject 'honey': {delete_result_by_filter}")


# 10. Drop the Collection (Clean Up)- Deletes the collection and all its data. No more garbage waiting on system
client.drop_collection(collection_name=COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' dropped.")

print("K we checked out milvus cool")

