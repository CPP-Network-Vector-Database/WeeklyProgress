from pymilvus import (
     connections,
     utility,
     FieldSchema,
     CollectionSchema,
     DataType,
     Collection,
)
import random

# Step 1: Connect to Milvus
connections.connect("default", host="localhost", port="19530")
print("Connected to Milvus.")

# Step 2: Define Schema
fields = [
     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True,auto_id=False),
     FieldSchema(name="random", dtype=DataType.DOUBLE),
     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8),
]
schema = CollectionSchema(fields, description="Milvus example with search and filtering.")

# Step 3: Drop collection if it exists
collection_name = "hello_milvus"
if utility.has_collection(collection_name):
     print(f"Dropping existing collection: {collection_name}")
     utility.drop_collection(collection_name)

# Step 4: Create a new collection
hello_milvus = Collection(collection_name, schema)
print(f"Collection '{collection_name}' created.")

# Step 5: Insert Data
entities = [
     [i for i in range(3000)],  # Primary key
     [float(random.uniform(-20, -10)) for _ in range(3000)],  # Random float values
     [[random.random() for _ in range(8)] for _ in range(3000)],  # 8D embeddings
]
insert_result = hello_milvus.insert(entities)
print(f"Inserted {len(entities[0])} entities.")
print(f"Number of entities in collection: {hello_milvus.num_entities}") 

# Step 6: Create Index
index_params = {
     "index_type": "IVF_FLAT",
     "metric_type": "L2",
     "params": {"nlist": 128},
}
hello_milvus.create_index("embeddings", index_params)
print("Index created.")

# Step 7: Load Collection for Search
hello_milvus.load()
print("Collection loaded into memory.")

# Step 8: Prepare Search Query
vectors_to_search = entities[-1][-2:]  # Take last two vectors from
inserted data
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

# Step 9: Perform Search (Without filtering)
print("Performing search (no filter)")
search_result = hello_milvus.search(vectors_to_search, "embeddings",
search_params, limit=3, output_fields=["random"])
print("Search Results:", search_result)

# Step 10: Query with Filtering
print("Querying with filter: random > -14")
query_result = hello_milvus.query(expr="random > -14",
output_fields=["random", "embeddings"])
print(f"Found {len(query_result)} results with random > -14")

# Step 11: Search with Filtering
print("Performing search with filter: random > -12")
filtered_search_result = hello_milvus.search(
     vectors_to_search, "embeddings", search_params, limit=3,
expr="random > -12", output_fields=["random"]
)
print("Filtered Search Results:", filtered_search_result)

# Step 12: Delete Entities
if insert_result.primary_keys:
     ids = insert_result.primary_keys[:2]  
     expr = f"pk in {ids}"
     print(f"Deleting entities with IDs: {ids}")
     hello_milvus.delete(expr)
     print("Entities deleted.")

# Step 13: Drop Collection
hello_milvus.release()
utility.drop_collection(collection_name)
print(f"Collection '{collection_name}' dropped.")


	
