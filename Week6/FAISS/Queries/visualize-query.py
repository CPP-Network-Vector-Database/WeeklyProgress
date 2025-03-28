import pandas as pd
import matplotlib.pyplot as plt

performance_df = pd.read_csv("/home/pes1ug22am100/Documents/WeeklyProgress/Week6/FAISS/faiss_performance.csv")

# Filter only query operations
query_operations = [
    "Query Before Insertion", 
    "Query After Insertion", 
    "Query After Deletion", 
    "Query After Update"
]

pivot_df = performance_df[performance_df["Operation"].isin(query_operations)].pivot(index="Operation", columns="Model", values="Time (s)")

plt.figure(figsize=(12, 6))

for model in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[model], marker="o", label=model)

plt.xlabel("Operation")
plt.ylabel("Query Time (s)")
plt.title("Query Time Comparison for Different FAISS Operations")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xticks(rotation=15)
plt.show()
