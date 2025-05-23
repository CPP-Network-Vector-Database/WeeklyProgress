### UI is ready!

Made a Streamlit app that loads network packet data, converts it to embeddings using DistilBERT (can be changed later), and lets you perform CRUD operations on a FAISS vector database while tracking performance metrics.

#### What it does
Takes the `ip_flow_dataset.csv` as usual and creates embeddings for all the packet data, stores them in FAISS, and gives you a UI to insert/delete/update/query the data. Shows real-time performance graphs for execution time, CPU usage, and memory consumption.

#### Setup

```bash
pip install streamlit pandas faiss-cpu sentence-transformers plotly psutil numpy matplotlib
```

Put the `ip_flow_dataset.csv` file in the same directory, then:

```bash
streamlit run faiss_ui.py
```

#### How to use

1. App loads your CSV data and creates embeddings (takes a minute)
2. Pick an operation from the sidebar (Insert/Delete/Update/Query/View)
3. Do the operation- it'll track performance and show results
4. Check the performance graphs on the right side

#### CSV format

Expects columns: `frame.number`, `frame.time`, `ip.src`, `ip.dst`, `tcp.srcport`, `tcp.dstport`, `_ws.col.protocol`, `frame.len`

Converts to text like: `"192.168.1.1 10.0.0.1 TCP 80 443 1500"`

#### Operations

- **Insert**: Add new packet data (type it in or generate random)
- **Delete**: Remove random entries from database  
- **Update**: Replace random entries with new data
- **Query**: Search for similar packets, get top-k results
- **View**: See database stats and sample data

Performance graphs update after each operation showing how time/CPU/memory scale with operation size.

That's it. Load data, do operations, watch performance graphs.

### Pain points: 
* Should the embedding model be the same or should the user be able to change it? 
* Must make the conversion to sentence form independent of the data format
* Graphs might need to be made more interpretable
* Need help loading the CSV- facing issues connecting it