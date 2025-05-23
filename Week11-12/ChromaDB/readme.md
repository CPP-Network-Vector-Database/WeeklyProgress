# Benchmarking outputs for 10k records

## Summary
- Shorter sequences have better performance.
- Cosine similarity has best performance.
- MiniLM-L6-v2 has best throughput and memory usage. DistilRoBERTa-v1 is better in query quality.
- Smallest disk sizes are for 256 as max sequence length, ip as distance metric and all_miniLM-L12-v2 as the embedding model.

## Max Sequence Length

### Throughput (operations/sec)

| Max Seq Length | Insert Throughput (ops/sec) | Query Throughput (ops/sec) | Update Throughput (ops/sec) | Delete Throughput (ops/sec) |
|----------------|-------------------------------|------------------------------|-------------------------------|-------------------------------|
| 128            | 11.48                         | 11.43                        | 11.16                         | 382.52                        |
| 256            | 10.72                         | 10.61                        | 8.94                          | 229.52                        |
| 512            | 9.39                          | 9.08                         | 8.93                          | 170.02                        |


### Time Taken per Operation (seconds/op)

| Max Seq Length | Insert Time (sec/op) | Query Time (sec/op) | Update Time (sec/op) | Delete Time (sec/op) |
|----------------|------------------------|-----------------------|------------------------|------------------------|
| 128            | 0.087137               | 0.087478              | 0.089571               | 0.002614               |
| 256            | 0.093324               | 0.094214              | 0.111917               | 0.004357               |
| 512            | 0.106511               | 0.110118              | 0.112042               | 0.005882               |


### Memory Usage per Operation (MB/op)

| Max Seq Length | Insert Memory (MB/op) | Query Memory (MB/op) | Update Memory (MB/op) | Delete Memory (MB/op) |
|----------------|-------------------------|------------------------|-------------------------|-------------------------|
| 128            | 0.084677                | 0.085767               | 0.085875                | 0.014159                |
| 256            | 0.084898                | 0.086507               | 0.085307                | 0.013188                |
| 512            | 0.083303                | 0.086422               | 0.084511                | 0.011750                |


### Disk Size

| Max Seq Length | Disk Size (MB) |
|----------------|----------------|
| 128 (Base)     | 19.22          |
| 256            | 15.89          |
| 512            | 19.21          |


### Average Query Distance

| Max Seq Length | Avg. of First 5 Query Distances |
|----------------|-----------------------------------|
| 128            | 0.043736                          |
| 256            | 0.044438                          |
| 512            | 0.051079                          |


## Distance Metric

### Throughput (operations/sec)

| Distance Metric | Insert Throughput (ops/sec) | Query Throughput (ops/sec) | Update Throughput (ops/sec) | Delete Throughput (ops/sec) |
|-----------------|-------------------------------|------------------------------|-------------------------------|-------------------------------|
| L2              | 11.48                         | 11.43                        | 11.16                         | 382.52                        |
| IP              | 9.22                          | 9.88                         | 10.18                         | 319.40                        |
| Cosine          | 13.20                         | 13.66                        | 12.94                         | 565.69                        |


### Time Taken per Operation (seconds/op)


| Distance Metric | Insert Time (sec/op) | Query Time (sec/op) | Update Time (sec/op) | Delete Time (sec/op) |
|-----------------|------------------------|-----------------------|------------------------|------------------------|
| L2              | 0.087137               | 0.087478              | 0.089571               | 0.002614               |
| IP              | 0.108494               | 0.101181              | 0.098229               | 0.003131               |
| Cosine          | 0.075772               | 0.073230              | 0.077282               | 0.001768               |


### Memory Usage per Operation (MB/op)

| Distance Metric | Insert Memory (MB/op) | Query Memory (MB/op) | Update Memory (MB/op) | Delete Memory (MB/op) |
|-----------------|-------------------------|------------------------|-------------------------|-------------------------|
| L2              | 0.084677                | 0.085767               | 0.085875                | 0.014159                |
| IP              | 0.083463                | 0.086086               | 0.083701                | 0.011397                |
| Cosine          | 0.083050                | 0.089468               | 0.082748                | 0.011852                |


### Disk Size

| Distance Metric | Disk Size (MB) |
|-----------------|----------------|
| l2 (Base)       | 19.22          |
| ip              | 15.92          |
| cosine          | 19.24          |

### Average Query Distance

| Distance Metric | Avg. of First 5 Query Distances |
|-----------------|-----------------------------------|
| L2              | 0.043736                          |
| IP              | 0.026994                          |
| Cosine          | 0.025811                          |



## Embedding Function

### Throughput (operations/sec)

| Embedding Model    | Insert Throughput (ops/sec) | Query Throughput (ops/sec) | Update Throughput (ops/sec) | Delete Throughput (ops/sec) |
|--------------------|-------------------------------|------------------------------|-------------------------------|-------------------------------|
| MiniLM-L6-v2       | 11.48                         | 11.43                        | 11.16                         | 382.52                        |
| DistilRoBERTa-v1   | 7.97                          | 8.25                         | 8.05                          | 544.45                        |
| MPNet-base-v2      | 3.16                          | N/A                          | N/A                           | N/A                           |
| MiniLM-L12-v2      | 6.75                          | N/A                          | N/A                           | N/A                           |

### Time Taken per Operation (seconds/op)

| Embedding Model    | Insert Time (sec/op) | Query Time (sec/op) | Update Time (sec/op) | Delete Time (sec/op) |
|--------------------|------------------------|-----------------------|------------------------|------------------------|
| MiniLM-L6-v2       | 0.087137               | 0.087478              | 0.089571               | 0.002614               |
| DistilRoBERTa-v1   | 0.125515               | 0.121209              | 0.124158               | 0.001837               |
| MPNet-base-v2      | 0.316190               | N/A                   | N/A                    | N/A                    |
| MiniLM-L12-v2      | 0.148144               | N/A                   | N/A                    | N/A                    |

### Memory Usage per Operation (MB/op)

| Embedding Model    | Insert Memory (MB/op) | Query Memory (MB/op) | Update Memory (MB/op) | Delete Memory (MB/op) |
|--------------------|-------------------------|------------------------|-------------------------|-------------------------|
| MiniLM-L6-v2       | 0.084677                | 0.085767               | 0.085875                | 0.014159                |
| DistilRoBERTa-v1   | 0.179023                | 0.176989               | 0.177904                | 0.014464                |
| MPNet-base-v2      | 0.294426                | N/A                    | N/A                     | N/A                     |
| MiniLM-L12-v2      | 0.111239                | N/A                    | N/A                     | N/A                     |


### Disk Size

| Embedding Model        | Disk Size (MB) |
|------------------------|----------------|
| all-MiniLM-L6-v2 (Base)| 19.22          |
| all-distilroberta-v1   | 35.91          |
| all-mpnet-base-v2      | 33.09          |
| all-MiniLM-L12-v2      | 18.59          |

### Average Query Distance

| Embedding Model    | Avg. of First 5 Query Distances |
|--------------------|-----------------------------------|
| MiniLM-L6-v2       | 0.043736                          |
| DistilRoBERTa-v1   | 0.017730                          |
| MPNet-base-v2      | N/A                               |
| MiniLM-L12-v2      | N/A                               |



**Base configuration - MiniLM-L6-v2, l2, 128**
