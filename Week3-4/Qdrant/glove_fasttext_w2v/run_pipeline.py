import os
from embeddings.word2vec import train_word2vec, get_word2vec_embedding
from embeddings.glove import load_glove_model, get_glove_embedding
from utils.pcap_processing import extract_pcap_data
from qdrant_operations.insert_data import store_embeddings
from qdrant_operations.query_data import query_qdrant

# Load PCAP file
pcap_file = "network_traffic.pcap"
if not os.path.exists(pcap_file):
    raise FileNotFoundError(f"PCAP file '{pcap_file}' not found!")

packets = extract_pcap_data(pcap_file)

if not packets:
    raise ValueError("No packets extracted! Ensure your PCAP file is valid.")

# Train Word2Vec on extracted packet data
word2vec_model = train_word2vec(packets)

# Load GloVe Model (using 100d version)
glove_model = load_glove_model("glove.6B.100d.txt")

# Generate embeddings
word2vec_embeddings = [get_word2vec_embedding(word2vec_model, pkt) for pkt in packets]
glove_embeddings = [get_glove_embedding(glove_model, pkt) for pkt in packets]

# Ensure embeddings are lists of floats
word2vec_embeddings = [embedding.tolist() if hasattr(embedding, "tolist") else embedding for embedding in word2vec_embeddings]
glove_embeddings = [embedding.tolist() if hasattr(embedding, "tolist") else embedding for embedding in glove_embeddings]

# Store embeddings in Qdrant
store_embeddings("word2vec_collection", word2vec_embeddings)
store_embeddings("glove_collection", glove_embeddings)

# Generate query embedding correctly
query_vector = get_word2vec_embedding(word2vec_model, "192.168.1.1 192.168.1.2 TCP")

# Ensure query_vector is a valid list of floats
if not isinstance(query_vector, list) or not all(isinstance(x, float) for x in query_vector):
    raise ValueError("query_vector must be a list of floats!")

# Corrected function call: Pass query_vector first, then collection name
results = query_qdrant(query_vector, "word2vec_collection")

print("Query Results (Word2Vec):", results)
