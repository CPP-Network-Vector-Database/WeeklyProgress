import time
import psutil

def measure_performance(func, *args):
    start_time = time.time()
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)

    result = func(*args)

    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)
    end_time = time.time()

    print(f"Time: {end_time - start_time:.4f}s")
    print(f"CPU Usage: {cpu_after - cpu_before:.2f}%")
    print(f"Memory Usage: {mem_after - mem_before:.2f}MB")

    return result
