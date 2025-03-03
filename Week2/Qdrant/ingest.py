import requests
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Fetch a large document from the web
def fetch_document(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    return text

# Split text into chunks
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate embeddings using SentenceTransformer
def generate_embeddings(texts):
    return model.encode(texts)

# Ingest data into Qdrant
def ingest_data(collection_name, url):
    # Fetch and chunk the document
    text = fetch_document(url)
    chunks = chunk_text(text)

    # Generate embeddings for chunks
    embeddings = generate_embeddings(chunks)

    # Create collection in Qdrant
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,  # all-mpnet-base-v2 embedding size
            distance=models.Distance.COSINE
        )
    )

    # Upload data to Qdrant
    records = [
        models.Record(
            id=idx,
            vector=embedding,
            payload={"text": chunk}
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    client.upload_records(collection_name=collection_name, records=records)

    print(f"Ingested {len(records)} records into collection '{collection_name}'.")

if __name__ == "__main__":
    # Example: Ingest data from a Wikipedia page
    ingest_data(
        collection_name="my_collection",
        url="https://en.wikipedia.org/wiki/Artificial_intelligence"
    )
