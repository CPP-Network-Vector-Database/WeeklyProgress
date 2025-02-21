import argparse
from ingest import extract_text_from_pdf, split_text, create_embeddings, insert_document_chunks
from query import semantic_query
from benchmark import benchmark_query
import warnings

def main():
    warnings.simplefilter("ignore",ResourceWarning)
    
    parser = argparse.ArgumentParser(description="Vector DB CLI for Document Ingestion and Querying")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for document ingestion
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("file_path")

    # Subparser for querying
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query_text")

    # Subparser for benchmarking
    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("query_text")

    args = parser.parse_args()

    if args.command == "ingest":
        text = extract_text_from_pdf(args.file_path)
        chunks = split_text(text)
        embeddings = create_embeddings(chunks)
        insert_document_chunks(chunks, embeddings)
        print("Document ingested successfully!")
    elif args.command == "query":
        result = semantic_query(args.query_text)
        print("Query Results:")
        for obj in result["data"]["Get"]["DocumentChunk"]:
            print(obj["content"])
    elif args.command == "benchmark":
        benchmark = benchmark_query(semantic_query, args.query_text)
        print("Benchmark Results:")
        print("Query Duration: {:.4f} seconds".format(benchmark["duration"]))
        print("CPU Usage: {}%".format(benchmark["cpu_usage"]))
        print("Memory Usage: {:.2f} MB".format(benchmark["memory_usage_MB"]))
        print("\n\n\n")
        print("Query Results:")
        for obj in benchmark["result"]["data"]["Get"]["DocumentChunk"]:
            print(obj["content"])

if __name__ == "__main__":
    main()
