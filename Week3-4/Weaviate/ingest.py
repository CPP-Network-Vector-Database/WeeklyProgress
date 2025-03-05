import PyPDF2
from sentence_transformers import SentenceTransformer
import weaviate


client = weaviate.Client("http://localhost:8080")
# loading the embedding model
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = SentenceTransformer("intfloat/e5-small-v2")

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
        # adding paragraph if it fits, otherwise starting a new chunk.
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# def create_embeddings(chunks):
#     embeddings = []
#     for chunk in chunks:
#         vec = embed_model.encode(chunk).tolist()
#         embeddings.append(vec)
#     return embeddings

def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        formatted_text = f"passage: {chunk}"  # Required for E5 models
        vec = embed_model.encode(formatted_text).tolist()
        embeddings.append(vec)
    return embeddings


def insert_document_chunks(chunks, embeddings):
    for text_chunk, vector in zip(chunks, embeddings):
        data_object = {"content": text_chunk}
        client.data_object.create(data_object, "DocumentChunk", vector=vector)


