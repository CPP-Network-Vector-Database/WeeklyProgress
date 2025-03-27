# Getting Started with Pgvector Database and Understanding Network Packets 

## Overview
This week, the focus was on initial setup of `Pgvector` Database and developing Python scripts for capturing, storing, and embedding network packet data using PostgreSQL with the pgvector extension. The work involved leveraging `psycopg2` for database interaction, `scapy` for real-time packet sniffing, and `sentence-transformers` to generate vector embeddings for deeper network analysis.

## Setup with Docker Compose
A `docker-compose` folder contains all necessary `docker-compose.yml` files for initial setup. Each service has its own `setup.md` file explaining its configuration and usage. This allows for easy deployment of Pgvetor Database, Pgadmin Dashboard and Simulation of Network Packets transfer between multiple hosts in a dedicated network using containers and other dependencies.

## Scripts

### 1. `pgvector_demo.py`
This script sets up a PostgreSQL database table to store student records, demonstrating basic database operations using `psycopg2`. It creates a `student` table with fields such as `student_id`, `first_name`, `last_name`, `age`, `email`, and `enrolled_date`. After creating the table, the script inserts sample student data and retrieves all records to display them. This script is useful for understanding how to interact with PostgreSQL using Python, execute SQL commands, and manage database transactions. It ensures that database connections are properly established and closed, making it a fundamental example for anyone new to PostgreSQL and `psycopg2`.

### 2. `pgvector_scapy.py`
This script captures live network packets using `scapy` and stores metadata in a PostgreSQL database. It creates a `packet_data` table with fields such as `source_ip`, `dest_ip`, `protocol`, `source_port`, `dest_port`, `packet_size`, `ttl`, `ip_flags`, `tcp_flags`, and `raw_data`. The script listens for packets using `sniff()`, extracts relevant fields, and inserts the data into the database. It handles both TCP and UDP packets, capturing essential details for network traffic analysis. This script can be used for monitoring and logging network activity, identifying anomalies, and performing security analysis. It provides a structured way to store network packet data for further investigation.

### 3. `scapy_embedded.py`
This script extends `pgvector_scapy.py` by incorporating vector embeddings for packet data using `sentence-transformers`. It captures packets, extracts metadata such as `src_ip`, `dst_ip`, `protocol`, and payload information, and then generates a numerical vector representation using the `all-MiniLM-L6-v2` model with size of `384 dimension`. These embeddings are stored in a `pgvector`-enabled PostgreSQL database, allowing for efficient similarity searches. This script is particularly useful for advanced network analysis, anomaly detection, and cybersecurity applications where identifying patterns in packet data is critical. By embedding packet payloads and metadata, it enables deeper insights into network traffic beyond traditional logging methods.

