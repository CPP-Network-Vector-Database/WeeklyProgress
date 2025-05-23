## 1. Custom Embeddings for plain IP flows
### Why BERT is Suboptimal for IP Flow Embeddings
BERT (and other language models) are designed for natural language processing (NLP), where the input is text with semantic meaning. However, IP flow data is fundamentally different:

### Downsides of BERT for IP Flow Embeddings
1. No Semantic Meaning in Raw IPs/Ports  
   - BERT assumes words have contextual meaning, but IP addresses (eg, 192.168.1.1) and port numbers (eg, 443) are arbitrary identifiers, not words.
   - Treating them as text loses their numeric and categorical structure.

2. Poor Handling of Mixed Data Types  
   - IP flows contain:
     - Categorical data (eg, protocol TCP/UDP)
     - Numeric data (eg, packet length, port numbers)
   - BERT processes everything as text, missing key relationships (eg, port 80 is closer to 443 than 22).

3. Inefficient for Structured Tabular Data  
   - BERT is computationally expensive and overkill for structured network logs.
   - It doesn’t naturally model hierarchical relationships (eg, subnet similarity in IPs).

4. No Built-in Anomaly Detection  
   - BERT embeddings are generic and don’t optimize for detecting unusual flows (eg, port scanning, DDoS).

### Why Custom Embeddings Work Better

Instead of treating IP flows as text, we use neural networks designed for structured/tabular data. These methods:

1. Preserve the structure of IPs, ports, and protocols.
2. Learn meaningful similarities (eg, nearby IPs, similar port usage).
3. Can be trained for specific tasks (eg, anomaly detection).
4. Are computationally efficient compared to large language models.


### Custom Embedding Methods

#### 1. MLP (Multi-Layer Perceptron) Embeddings
How it Works:  
- A simple feedforward neural network that takes raw features (IPs, ports, etc.) and maps them to a dense embedding space.
- Learns nonlinear relationships between features.

Why Better Than BERT?  
- Handles mixed data types (numeric + categorical).
- Lightweight and fast to train.
- Can be optimized for specific tasks (eg, clustering similar flows).

Best For:  
- Baseline performance comparison.
- When interpretability is needed.

#### 2. TabTransformer Embeddings
How it Works:  
- Uses transformers (like BERT) but specifically designed for tabular data.
- Each categorical feature (eg, ip.src, protocol) is embedded separately.
- Continuous features (eg, frame.len) are projected into the same space.
- A transformer layer models interactions between features.

Why Better Than BERT?  
- Respects tabular structure (unlike BERT, which flattens everything).
- Captures complex feature interactions (eg, protocol=TCP + port=80 → likely HTTP).
- More efficient than BERT since it’s designed for tables.

Best For:  
- Modeling relationships between IPs, ports, and protocols.
- Cases where feature interactions matter (eg, detecting unusual protocol-port combos).

#### 3. Autoencoder Embeddings
How it Works:  
- A neural network that compresses input data into a lower-dimensional embedding.
- Trained to reconstruct the original input from the embedding.
- The embedding (bottleneck layer) captures the most important patterns.

Why Better Than BERT?  
- Naturally learns a compressed representation.
- Can detect anomalies (flows with high reconstruction error are unusual).
- No need for labels (unsupervised).

Best For:  
- Anomaly detection (eg, rare flows stand out).
- Dimensionality reduction for visualization.


#### 4. Contrastive Embeddings
How it Works:  
- Trains a model to pull similar flows closer and push dissimilar flows apart.
- Requires defining a similarity metric (eg, flows from the same subnet are "similar").
- Uses a Siamese network structure.

Why Better Than BERT?  
- Optimized for similarity search (unlike BERT, which is generic).
- Can encode domain knowledge (eg, "similarity" based on subnet, not raw IPs).
- Works well for clustering.

Best For:  
- Finding similar flows (eg, all SSH traffic).
- Cases where manual similarity rules are known.


## 2. Working with Semantics
This time packets are converted to natural language descriptions, allowing queries like:
- "Find all TCP packets"
- "Show packets from 192.168.1.1"
- "Find large packets over 1000 bytes"

### What it does
- Converts raw packet data to semantic descriptions
- Supports multiple BERT-based embedding models
- Tracks performance metrics (time, CPU, memory)
- Enables natural language queries

### Next steps
- Have to measure its performance across embedding models
- Have to get the similarity scores

#### Note: 
I ran the CLI code on colab since my laptop was heating up a little too much and this was to just check out the semantic search space.