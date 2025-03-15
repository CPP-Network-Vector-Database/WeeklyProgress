from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)

client.recreate_collection(
    collection_name="glove_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE)  # Adjust 'size' based on your embeddings
)
