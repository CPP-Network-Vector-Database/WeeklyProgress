ALBERT- Combines all features to text to generate embeddings
Time taken to retreive top 5 ip address-0.113904s

DistilBERT has been compared with generating embeddings as- Combining all features as text format and Converting all features as numerical values
Where the first method has shown better results with lower latency,as text format is the default format of DistilBERT to handle embeddings.
Time taken to retreive top 5 ip address(string)-0.008907s
Time taken to retreive top 5 ip address(encoding)-0.137075s

ELECTRA- Combined text format handling to generate embeddings.
Time taken to retreive top 5 ip address-0.159154s

DistilBERT was found to be the fastest in querying the same results of Number of frequently used Source IP addresses.


Multiple queries has been thus added to DistilBERT embeddings to retreive different results.

On trying CRUD operations - Large insert of 500 packets took 0.120377s using Distilbert.
