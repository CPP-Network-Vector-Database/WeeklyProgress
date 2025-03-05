import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import pandas as pd
import uuid
from tqdm import tqdm
import time


embedding_model = SentenceTransformer("all-distilroberta-v1", device="cpu")
embedding_model.max_seq_length = 128 


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts, num_workers=1).tolist()  


client = chromadb.PersistentClient(path="./chromadb")
client.heartbeat()


df = pd.read_csv('../pacp_csv/dirA.125910-packets.csv')


df.dropna(subset=["Source", "Destination", "Protocol"], inplace=True)
df.drop_duplicates(subset=["Source", "Destination", "Protocol"], inplace=True)


documents = (df["Source"] + ", " + df["Destination"] + ", " + df["Protocol"]).tolist()
ids = [str(uuid.uuid4()) for _ in documents]


embedding_function = SentenceTransformerEmbeddingFunction()
collection = client.get_or_create_collection(name="distilroberta", embedding_function=embedding_function)


def batch_encode_texts(texts, batch_size=500):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress"):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=False).tolist()
        embeddings.extend(batch_embeddings)
    return embeddings


start_time = time.time()
batch_size = 500 
embeddings = batch_encode_texts(documents, batch_size)
print(f"Embedding completed in {time.time() - start_time:.2f} seconds.")


max_chroma_batch_size = 40000 

for i in tqdm(range(0, len(documents), max_chroma_batch_size), desc="Adding to ChromaDB"):
    batch_docs = documents[i:i + max_chroma_batch_size]
    batch_ids = ids[i:i + max_chroma_batch_size]
    batch_embeddings = embeddings[i:i + max_chroma_batch_size]

    collection.add(documents=batch_docs, ids=batch_ids, embeddings=batch_embeddings)

print("Data successfully added to ChromaDB.")
