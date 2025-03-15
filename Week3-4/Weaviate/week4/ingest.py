import PyPDF2
from sentence_transformers import SentenceTransformer
import weaviate
import time
import psutil

client = weaviate.Client("http://localhost:8080")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text(text, max_length=500):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_embeddings(chunks):
    return [embed_model.encode(chunk).tolist() for chunk in chunks]

def insert_document_chunks(chunks, embeddings):
    start_time = time.time()
    for text_chunk, vector in zip(chunks, embeddings):
        client.data_object.create({"content": text_chunk}, "DocumentChunk", vector=vector)
    duration = time.time() - start_time
    print(f"Ingestion Time: {duration:.4f} seconds")