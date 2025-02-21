import weaviate


client = weaviate.Client("http://localhost:8080")


document_chunk_schema = {
    "class": "DocumentChunk",
    "vectorizer": "none",
    "properties": [
        {
            "name": "content",
            "dataType": ["text"],
            "description": "The content of the document chunk"
        }
    ]
}


existing_schema = client.schema.get()
if any(cls["class"] == "DocumentChunk" for cls in existing_schema.get("classes", [])):
    print("The 'DocumentChunk' class already exists in the schema.")
else:
    client.schema.create_class(document_chunk_schema)
    print("The 'DocumentChunk' class has been created successfully!")
