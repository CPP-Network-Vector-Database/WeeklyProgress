from qdrant_client import QdrantClient

def query_qdrant(query_embedding, collection_name="network_traffic", top_k=5):
    """
    Query Qdrant with an embedding and retrieve the top-k closest matches.

    Args:
        query_embedding (list): The embedding vector to search for.
        collection_name (str): The name of the Qdrant collection.
        top_k (int): Number of closest matches to return.

    Returns:
        list: Top-k results from Qdrant.
    """
    client = QdrantClient("localhost", port=6333)

    # Debugging
    print(f"Type of query_embedding: {type(query_embedding)}")
    if isinstance(query_embedding, list):
        print(f"First 5 values: {query_embedding[:5]}")
    else:
        raise ValueError("query_embedding must be a list of floats.")

    # Ensure it's a valid vector
    query_embedding = list(map(float, query_embedding))

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    return search_results
