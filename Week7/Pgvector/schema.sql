-- To Make sure vector support extension is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Monitor Query Metrics
CREATE EXTENSION pg_stat_statements;


-- General Table creation
CREATE TABLE network_traffic (
    frame_number INTEGER PRIMARY KEY,
    frame_time TIMESTAMPTZ,
    ip_src INET,
    ip_dst INET,
    tcp_srcport INTEGER,
    tcp_dstport INTEGER,
    protocol TEXT,
    frame_len INTEGER,
    embedding VECTOR(384)
);


-- Vector Index Table creation


-- Table 1: Using IVFFlat
CREATE TABLE network_traffic_ivf (
    frame_number INTEGER PRIMARY KEY,
    frame_time TIMESTAMPTZ,
    ip_src INET,
    ip_dst INET,
    tcp_srcport INTEGER,
    tcp_dstport INTEGER,
    protocol TEXT,
    frame_len INTEGER,
    embedding VECTOR(384)
);

-- Create IVFFlat Index
CREATE INDEX ivf_flat_idx 
ON network_traffic_ivf USING ivfflat (embedding) 
WITH (lists = 100);    


-- Table 2: Using HNSW
CREATE TABLE network_traffic_hnsw (
    frame_number INTEGER PRIMARY KEY,
    frame_time TIMESTAMPTZ,
    ip_src INET,
    ip_dst INET,
    tcp_srcport INTEGER,
    tcp_dstport INTEGER,
    protocol TEXT,
    frame_len INTEGER,
    embedding VECTOR(384)
);

-- Create HNSW Index
CREATE INDEX hnsw_idx 
ON network_traffic_hnsw USING hnsw (embedding vector_cosine_ops)
WITH (ef_construction = 200, m = 16);