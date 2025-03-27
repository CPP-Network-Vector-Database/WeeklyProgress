from sentence_transformers import SentenceTransformer
import weaviate

client = weaviate.Client("http://localhost:8080")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_query_ip_flow(query_text, limit=5):
    query_vector = embed_model.encode(query_text).tolist()
    result = (
        client.query
        .get("IPFlow", ["source_ip", "destination_ip", "protocol", "packet_size", "timestamp"])
        .with_near_vector({"vector": query_vector})
        .with_limit(limit)
        .do()
    )
    return result

def update_ip_flow(source_ip, new_packet_size):
    query = client.query.get("IPFlow", ["_additional { id }"]).with_where({
        "path": ["source_ip"],
        "operator": "Equal",
        "valueString": source_ip
    }).do()
    updated_count = 0
    for obj in query.get("data", {}).get("Get", {}).get("IPFlow", []):
        updated_count += 1
        client.data_object.update({"packet_size": new_packet_size}, class_name="IPFlow", uuid=obj["_additional"]["id"])

    print(f"Updated packet size to {new_packet_size} for {updated_count} IP flows with source IP: {source_ip}")

def delete_ip_flow(protocol_number):
    query = client.query.get("IPFlow", ["_additional { id }"]).with_where({
        "path": ["protocol"],
        "operator": "Equal",
        "valueString": protocol_number 
    }).do()

    deleted_count = 0
    for obj in query.get("data", {}).get("Get", {}).get("IPFlow", []):
        client.data_object.delete(obj["_additional"]["id"], class_name="IPFlow")
        deleted_count += 1

    print(f"Deleted {deleted_count} IP flows with protocol: {protocol_number}")



