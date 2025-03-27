import psycopg2
import pandas as pd
import time
from pgvector.psycopg2 import register_vector 

DB_PARAMS = {
    "dbname": "your_database_name",     # Replace with your database name
    "user": "your_username",       # Replace with your username
    "password": "your_password",  # Replace with your password
    "host": "localhost",
    "port": "5432"
}

def execute_query(query, params=None, fetch=True):
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)  # Register pgvector type for proper handling
    cursor = conn.cursor()

    start_time = time.time()
    cursor.execute(query, params)
    exec_time = time.time() - start_time

    if fetch:
        # Get column names and all rows
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        result_df = pd.DataFrame(data, columns=columns)
    else:
        conn.commit()
        result_df = None

    cursor.close()
    conn.close()
    return result_df, exec_time

def benchmark_query(query, params=None, query_name=""):
    print(f"--- {query_name} ---")
    df, exec_time = execute_query(query, params)
    print(f"Execution Time: {exec_time:.4f} seconds")
    if df is not None:
        print(df)
    print("-" * 80)

def main():
    # 1. Fetch a Sample of All Data
    query_all = "SELECT * FROM pcap_embeddings LIMIT 1000;"
    benchmark_query(query_all, query_name="Fetch All Data")
    
    # 384 dimensions.
    example_vector = "[" + ",".join(["0.02"] * 384) + "]"
    
    # 2. Vector Similarity Search (find similar packets)
    query_similar = """
    SELECT *, vector <=> %s::vector AS similarity
    FROM pcap_embeddings
    ORDER BY similarity ASC
    LIMIT 5;
    """
    benchmark_query(query_similar, (example_vector,), "Vector Similarity Search")
    
    # 3. Outlier Query (anomaly detection: farthest vectors)
    query_outliers = """
    SELECT *, vector <=> %s::vector AS distance
    FROM pcap_embeddings
    ORDER BY distance DESC
    LIMIT 5;
    """
    benchmark_query(query_outliers, (example_vector,), "Outlier Query")
    
    # 4. Group by Protocol and Get Average Packet Length
    query_avg_length = """
    SELECT protocol, AVG(length::INTEGER) AS avg_length
    FROM pcap_embeddings
    GROUP BY protocol;
    """
    benchmark_query(query_avg_length, query_name="Average Packet Length by Protocol")
    
    # 5. Detect DDoS Patterns (count packets by source)
    query_ddos = """
    SELECT src, COUNT(*) AS packet_count
    FROM pcap_embeddings
    GROUP BY src
    ORDER BY packet_count DESC
    LIMIT 10;
    """
    benchmark_query(query_ddos, query_name="DDoS Detection - Top Sources")

if __name__ == "__main__":
    main()
