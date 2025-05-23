import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_semantic_packet_text(row):
    src = f"source IP {row['ip.src']}" if pd.notna(row['ip.src']) else "an unknown source"
    dst = f"destination IP {row['ip.dst']}" if pd.notna(row['ip.dst']) else "an unknown destination"
    protocol = f"using {row['_ws.col.protocol']} protocol" if pd.notna(row['_ws.col.protocol']) else ""
    src_port = f"from port {row['tcp.srcport']}" if pd.notna(row['tcp.srcport']) else ""
    dst_port = f"to port {row['tcp.dstport']}" if pd.notna(row['tcp.dstport']) else ""
    length = f"with packet size {row['frame.len']} bytes" if pd.notna(row['frame.len']) else ""
    return f"A network packet from {src} {src_port} to {dst} {dst_port} {protocol} {length}"

def load_data_and_build_index(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    df["packet_text"] = df.apply(create_semantic_packet_text, axis=1)
    
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(df["packet_text"], convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return df, model, index

def search_packets(query, df, model, index, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    distances, indices = index.search(query_embedding, k)
    return [(df.iloc[idx]['packet_text'], 1-dist) for idx, dist in zip(indices[0], distances[0])]

def main():
    csv_path = input("Enter path to your packet CSV: ").strip()
    df, model, index = load_data_and_build_index(csv_path)
    
    print("Try queries like:")
    print("- 'Find TCP packets'")
    print("- 'Show packets from 192.168.1.1'")
    print("- 'Find large packets over 1000 bytes'\n")
    
    while True:
        query = input("Enter your search query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
            
        results = search_packets(query, df, model, index)
        print(f"\nTop {len(results)} results for '{query}':")
        for i, (text, similarity) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {similarity:.2f}")
            print(text)
        print()

if __name__ == "__main__":
    main()