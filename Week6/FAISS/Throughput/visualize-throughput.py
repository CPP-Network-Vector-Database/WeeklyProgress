import pandas as pd
import matplotlib.pyplot as plt

# Load data
performance_df = pd.read_csv("/home/pes1ug22am100/Documents/WeeklyProgress/Week6/FAISS/Throughput/faiss_performance.csv")

# Filter for throughput operations
throughput_operations = [
    "Query Throughput",
    "Insertion Throughput",
    "Deletion Throughput",
    "Update Throughput"
]

# Pivot data for easier plotting
pivot_df = performance_df[performance_df["Operation"].isin(throughput_operations)].pivot(index="Operation", columns="Model", values="Throughput (ops/sec)")

# Plot
plt.figure(figsize=(12, 6))

for model in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[model], marker="o", label=model)

plt.xlabel("Operation")
plt.ylabel("Throughput (ops/sec)")
plt.title("Throughput Comparison for Different Models")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xticks(rotation=15)
plt.show()
