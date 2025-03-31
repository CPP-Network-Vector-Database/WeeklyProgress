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
        .with_additional(["distance"]) #
        .with_limit(limit)
        .do()
    )
    return result

def update_ip_flow(source_ip, new_packet_size):
    batch_size=100
    update_count=0
    while True:
        query = client.query.get("IPFlow", ["_additional { id }"]).with_where({
            "operator": "And",
            "operands": [
                {"path": ["source_ip"], "operator": "Equal", "valueString": source_ip},
                {"path": ["packet_size"], "operator": "NotEqual", "valueNumber": new_packet_size}#to fetch only those records which have not been updated.
            ]
        }).with_limit(batch_size).do()

        ip_flows = query.get("data", {}).get("Get", {}).get("IPFlow", [])
        if not ip_flows:
            break   #to stop when no more records are found
        
        for obj in ip_flows:
            client.data_object.update({"packet_size": new_packet_size}, class_name="IPFlow", uuid=obj["_additional"]["id"])
            update_count+=1

    print(f"Updated packet size to {new_packet_size} for {update_count} IP flows with source IP: {source_ip}")

def delete_ip_flow(protocol_number):
    batch_size=100
    deleted_count=0
    while True:
        query = client.query.get("IPFlow", ["_additional { id }"]).with_where({
            "path": ["protocol"],
            "operator": "Equal",
            "valueString": protocol_number 
        }).with_limit(batch_size).do()

        ip_flows=query.get("data", {}).get("Get", {}).get("IPFlow", [])
        if not ip_flows:
            break #to stop when no more records are found

        for obj in ip_flows:
            client.data_object.delete(obj["_additional"]["id"], class_name="IPFlow")
            deleted_count += 1

    print(f"Deleted {deleted_count} IP flows with protocol: {protocol_number}")


