# FAERec
Official source code for SIGIR 2026 paper: Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation

# Data Preprocessing
```Preprocessing``` includes standard ID data handling and the acquisition of LLM embeddings. Using the Beauty dataset as an example, the steps are as follows:
1. Place raw dataset files in the ```Preprocessing/data/Beauty/raw``` directory
2. Run ```data_process.py```.
3. Run ```get_item_embedding.py``` in the ```Preprocessing/data/Beauty/handled``` to obtain LLM item embeddings.
4. Processed LLM embeddings should be placed in the ```FAERec/src/llm_emb/Beauty``` directory.

# Run the code
```Base``` and ```FAERec``` represent the backbone model and our proposed method, respectively.

For detailed hyperparameter settings, please refer to the log.

To save time, we have placed the pre-trained model weights and logs in ```SASRec/src/output``` and ```FAERec/src/output``` for direct execution.

The term "Food" in the files refers to "Grocery and Gourmet Food".

You can specify the corresponding backbone model by setting "--model_name" and "SASRec" is the default model.

Run FAERec in the ```FAERec/src``` directory:
```
python main.py  --gpu_id=0 --model_idx=1 --data_name=Beauty 
python main.py  --gpu_id=0 --model_idx=1 --data_name=Food --w_min=0.7 --period=20 --star_test=60 
python main.py  --gpu_id=0 --model_idx=1 --data_name=Yelp --w_min=0.3 --period=25 --star_test=50 
```

You can also run the backbone model (SASRec,FMLP-Rec,LRURec) in the ```Backbone/src``` directory using the following command:
```
python main.py  --gpu_id=0 --model_idx=1 --data_name=[DATA NAME]
```
