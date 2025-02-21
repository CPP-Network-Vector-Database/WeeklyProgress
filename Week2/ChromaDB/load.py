import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import pandas as pd
import uuid

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts).tolist()

client = chromadb.PersistentClient(path="./chromadb")

client.heartbeat()

df = pd.read_csv('pacp_csv/dirA.125910-packets.csv')

df_ss = df.head(27000)

embedding_function = SentenceTransformerEmbeddingFunction()
collection = client.get_or_create_collection(name="my_collection", embedding_function=embedding_function)

documents = (df_ss["Source"] + ", " + df_ss["Destination"] + ", " + df_ss["Protocol"]).tolist()
ids = [str(uuid.uuid4()) for _ in documents]
embeddings = embedding_function(documents)

collection.add(documents=documents, ids=ids, embeddings=embeddings)
