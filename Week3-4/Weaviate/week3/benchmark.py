import time
import psutil

def benchmark_query(query_func, *args, **kwargs):
    start_time = time.time()
    result = query_func(*args, **kwargs)
    duration = time.time() - start_time

    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=0.5)

    rss_memory = process.memory_info().rss
    mem_usage = rss_memory / (1024 * 1024)  # in MB

    total_memory = psutil.virtual_memory().total  # Total system memory in bytes
    memory_percentage = (rss_memory / total_memory) * 100
    print(f"Memory Usage: {memory_percentage:.2f}% of system memory")

    total_cpu_usage = (cpu_usage / psutil.cpu_count(logical=True)) 
    print(f"CPU Usage: {total_cpu_usage:.2f}% of total system CPU")

    return {
        "result": result,
        "duration": duration,
        "cpu_usage": cpu_usage,
        "memory_usage_MB": mem_usage
    }



