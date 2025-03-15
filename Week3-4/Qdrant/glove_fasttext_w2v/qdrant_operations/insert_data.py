from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

def store_embeddings(collection_name, embeddings, metadata_list=None):
    client = QdrantClient("localhost", port=6333)  

    points = []
    for i, emb in enumerate(embeddings):
        if isinstance(emb, str):
            raise ValueError(f"Embedding at index {i} is a string, expected a numeric list: {emb}")

        points.append(
            PointStruct(id=i, vector=emb.tolist() if hasattr(emb, "tolist") else emb, payload=metadata_list[i] if metadata_list else {})
        )

    batch_size = 500  # Adjust batch size as needed
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])
