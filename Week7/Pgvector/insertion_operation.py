import csv
import psycopg2
from psycopg2.extras import execute_values
import ast
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Database connection configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT'))
}

# CSV path
CSV_PATH = 'processed_ip_flow_data.csv'

def parse_nullable_int(val):
    return int(val) if val != 'NULL' else None

def parse_vector(vector_str):
    # Turn string '[0.1, 0.2, ...]' into actual Python list
    return ast.literal_eval(vector_str)

def insert_data():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            rows = []

            for row in reader:
                rows.append((
                    int(row['frame_number']),
                    row['frame_time'],
                    row['ip_src'],
                    row['ip_dst'],
                    parse_nullable_int(row['tcp_srcport']),
                    parse_nullable_int(row['tcp_dstport']),
                    row['protocol'],
                    int(row['frame_len']),
                    parse_vector(row['embedding'])
                ))

        query = """
        INSERT INTO network_traffic_hnsw (
            frame_number, frame_time, ip_src, ip_dst,
            tcp_srcport, tcp_dstport, protocol, frame_len, embedding
        ) VALUES %s
        """

        print("Inserting data in table network_traffic_hnsw with hnsw indexing used in the table")
        start_time = time.time()

        execute_values(cur, query, rows)

        conn.commit()
        end_time = time.time()

        print(f"Data inserted successfully!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print("Error: ", e)

    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == '__main__':
    insert_data()
