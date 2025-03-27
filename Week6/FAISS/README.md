This week, the following were my AAIs:
- Finetune BERT for IP Flows
- Push .ipynb code to .py
- Integrate FAISS with another vector database

The progress is as follows: 

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

### ipynb to py
This was done on the first day
