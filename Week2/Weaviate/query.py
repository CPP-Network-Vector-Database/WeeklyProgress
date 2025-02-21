from sentence_transformers import SentenceTransformer
import weaviate

client = weaviate.Client("http://localhost:8080")
# loading the same embedding model to encode query text
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_query(query_text, limit=2):
    query_vector = embed_model.encode(query_text).tolist()
    result = client.query.get("DocumentChunk", ["content"]).with_near_vector({"vector": query_vector}).with_limit(limit).do()
    return result


