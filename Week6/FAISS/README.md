### Finetuning BERT
* The approach used was to basically convert each row into one sentence
* This was then tokenized using a pre-made tokeniser (it uses the wordpiece algorithm that basically just chunks the text into tokens)
* Then Masked Language Modelling was used to finetune bert on these embeddings
* The five tuple flow used had: source IP, destination IP, source port, destination port, and flow packets/s
* The dataset used was soruced from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/tor.html)
* The data exploration and concatenation was done to obtain one csv with ~100000 five tuple flows
* The pcap files were available as well, but CSV was used for ease of use
* If this proof of concept works, then bert can be finetuned with the flows extracted from the pcaps directly, which would help the final pipeline
* The data is available on [kaggle](https://www.kaggle.com/datasets/namitaachyuth/iscx-tor-nontor-2017-csvs)

Issues: 
* There seems to be some issues with finetuning BERT itself, I will have to look into this further
* Masked language modelling is an approach used by multiple papers for finetuning, but it is having trouble with ip flows- I think I might have to change the tokenizer instead of using the plug and play one
* Currently working with my professor to understand different embedding options

### Measuring query perfromance after large (250) insertions, deletions and updates
![image](https://github.com/user-attachments/assets/d8f5ca12-b06f-40aa-9865-c0edb74becea)
- Query Before Insertion is the slowest for some models (paraphrase-MiniLM)
    - implies that adding data improves FAISS's indexing efficiency? Not sure about the causal relationship here
- query time generally increases after insertion or update, but not uniformly across models
    - paraphrase-MiniLM shows a huge spike in query time after insertion, which means that its FAISS indexing is significantly impacted
    - all-MiniLM also has a noticeable increase in query time post-update
- Deletion has minimal impact on most models' query time, except for a slight rise in a few cases
    - FAISS efficiently handles deletions without significantly degrading query speed.
- microsoft/codebert-base experiences the highest query time after insertion
    - the FAISS index struggles to optimize query performance for this model after a large batch of insertions.
- Models like bert-base and distilbert are relatively stable, with small variations in query performance across operations

### Measuring throughput after large (250) insertions, deletions and updates
Approach used: 
- Record Start Time:
    - start_time = time.time() captures the time before executing the function.
- Execute the Function (func)
    - The function (insertion, deletion, query, update) is executed with the given arguments.
- Record End Time:
    - end_time = time.time() captures the time after execution.
- Compute Execution Time:
    - execution_time = end_time - start_time gives the total duration taken by the function
- Calculate Throughput:
    - throughput = num_operations / execution_time
    - We're basically calculating how many operations were completed per second
- and if execution time is zero (very rare but possible in edge cases), throughput is set to 0 to avoid division errors.
![image](https://github.com/user-attachments/assets/e2bc4ca0-42a9-4ba9-b46a-a74c8cee78ac)

**Key Takeaways**:
- Deletion throughput is the highest for all models (around 2M ops/sec).
    -  This could be because deleting an item from a FAISS index is often computationally simpler than inserting or querying
- insertion throughput is significantly lower than deletion
- query and update throughput values are relatively low, indicating that these operations take longer per operation compared to deletion (probably because of the partitions)



### Pipeline: Measuring model performance for *variable* insertions, deletions, and updates
1. paraphrase-miniLM
   ![image](https://github.com/user-attachments/assets/65d69e4e-f3dc-42f3-98ad-2728a4d1fd59)
2. all-MiniLM
   ![image](https://github.com/user-attachments/assets/9bb1fbf3-6c2f-46e3-80e8-91f51a8d4a67)
3. distilbert
   ![image](https://github.com/user-attachments/assets/1f67d91f-5d22-4f73-9d9f-ccf1b0d72dc6)
4. codebert
   ![image](https://github.com/user-attachments/assets/5efba466-73e1-4312-8786-78b6f2064041)
5. bert-base
   ![image](https://github.com/user-attachments/assets/3f9d66fa-3b43-4836-8307-c28144610e4f)
6. average-word-embeddings
   ![image](https://github.com/user-attachments/assets/4b5f12ce-3736-497b-a6de-58630f3362da)
7. all-mpnet
   ![image](https://github.com/user-attachments/assets/7cd59a03-4607-4849-95cf-08fd9184b21b)






