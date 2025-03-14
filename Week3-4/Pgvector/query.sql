CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE pcap_embeddings (
    id SERIAL PRIMARY KEY,
    no INTEGER,
    time TEXT,
    src TEXT,
    dst TEXT,
    protocol TEXT,
    length TEXT,
    info TEXT,
    vector VECTOR(384) -- Adjust based on embedding model
);

-- basic queries
select * from pcap_embeddings;

select count(*) from pcap_embeddings;

select protocol,count(*) 
from pcap_embeddings
group by protocol;

drop table pcap_embeddings;
