import weaviate

client = weaviate.Client("http://localhost:8080")

ip_flow_schema = {
    "class": "IPFlow",
    "vectorizer": "none",  
    "properties": [
        {"name": "source_ip", "dataType": ["string"]},
        {"name": "destination_ip", "dataType": ["string"]},
        {"name": "protocol", "dataType": ["string"]},
        {"name": "packet_size", "dataType": ["int"]},
        {"name": "timestamp", "dataType": ["string"]},
    ]
}

existing_schema = client.schema.get()
if any(cls["class"] == "IPFlow" for cls in existing_schema.get("classes", [])):
    print("The 'IPFlow' class already exists.")
else:
    client.schema.create_class(ip_flow_schema)
    print("The 'IPFlow' class has been created successfully!")
