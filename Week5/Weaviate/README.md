# Progress
- Worked with **IP flows** instead of PDF documents.
- Created a better **memory usage metric** by considering the **delta** (see `benchmark.py`).
- Implemented **CRUD operations**.

## Modified Files
- **`schema.py`** & **`ingest.py`**  
  - Updated to accommodate the **IP flow schema**.
- **`main.py`**  
  - Added **subparsers** for `update` and `delete` operations.
- **`query.py`**  
  - Updated to include **`update`** and **`delete`** operations.

## `capture_ip_flow.py`
This script is used to **capture network packets** using the **Scapy** library and store them in `ip_flows.csv`.  
The script captures the following packet data:
- **`source_ip`**  
- **`destination_ip`**  
- **`protocol`**  
- **`packet_size`**  
- **`timestamp`**

##  Libraries Used
Scapy - for sniffing network packets.
```bash
pip install scapy
