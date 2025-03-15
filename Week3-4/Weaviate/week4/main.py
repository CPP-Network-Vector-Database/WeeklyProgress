import argparse
from ingest import extract_text_from_pdf, split_text, create_embeddings, insert_document_chunks
from query import semantic_query, batch_query
from benchmark import benchmark_query, start_cpu_monitor, stop_cpu_monitor
import warnings
import threading

def main():
    warnings.simplefilter("ignore", ResourceWarning)
    
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
    bench_parser.add_argument("query_text", nargs='+')  # to accept multiple queries

    args = parser.parse_args()

    start_cpu_monitor()

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
        benchmark = benchmark_query(batch_query, args.query_text)
        print("Benchmark Results:")
        print("Avg Query Duration: {:.4f} seconds".format(benchmark["duration"]))
        print("Avg CPU Usage: {:.2f}%".format(benchmark["cpu_usage"]))
        print("Avg Memory Usage: {:.2f} MB".format(benchmark["memory_usage_MB"]))
    
    stop_cpu_monitor()


if __name__ == "__main__":
    main()