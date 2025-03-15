import time
import psutil
import threading

cpu_usage_data = []
recording = True                                                                                                                                                                                                                                                                          

def record_cpu_usage():
    global recording
    while recording:
        cpu_usage_data.append(psutil.cpu_percent(interval=0.01)/psutil.cpu_count())

def start_cpu_monitor():
    global recording
    recording = True
    monitor_thread = threading.Thread(target=record_cpu_usage, daemon=True)
    monitor_thread.start()

def stop_cpu_monitor():
    global recording
    recording = False  
    with open("cpu_usage_log.txt", "w") as f:
        for value in cpu_usage_data:
            f.write(f"{value}\n")
    print(f"CPU usage log saved. Recorded {len(cpu_usage_data)} entries.")

def benchmark_query(query_func, queries):
    start_time = time.time()
    results = query_func(queries)
    duration = time.time() - start_time
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=0.01)/psutil.cpu_count()
    mem_usage = process.memory_info().rss / (1024 * 1024)
    return {"result": results, "duration": duration / len(queries), "cpu_usage": cpu_usage, "memory_usage_MB": mem_usage}
