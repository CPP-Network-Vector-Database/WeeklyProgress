### Fixing the old issue of negative CPU and Memory usage
tl;dr: pick up the PID of the process and only use psutils for that rather than making a measure of the whole system

This measure function helps accurately track CPU, memory usage, and execution time for any function. Previously, negative values appeared due to incorrect timing in resource measurement. 

#### **How It Works:**  
1. Captures CPU and memory usage before running the function.  
2. Runs the function with the given arguments.  
3. Captures CPU and memory usage again after execution.  
4. Calculates the difference to get actual resource usage.  

#### **Why This Fixes the Issue:**  
- `cpu_percent(interval=None)` ensures we don’t get misleading CPU readings.  
- Memory is tracked correctly in MB, avoiding fluctuations.  
- Execution time is measured precisely.  

