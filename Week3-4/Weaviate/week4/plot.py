import matplotlib.pyplot as plt


file_path = "./cpu_usage_log.txt"
with open(file_path, "r") as file:
    cpu_usage_data = [float(line.strip()) for line in file if line.strip()]

time_intervals = list(range(len(cpu_usage_data)))

plt.figure(figsize=(10, 5))
plt.plot(time_intervals, cpu_usage_data, marker='o', linestyle='-', color='b', label="CPU Usage (%)")


plt.xlabel("Time (in units of 0.01s)")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage Over Time")
plt.ylim(0, 100) 
plt.grid(True)
plt.legend()


plt.show()
