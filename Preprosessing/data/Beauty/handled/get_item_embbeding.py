import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

prompt_template = "The beauty item has following attributes: \n name is <TITLE>; brand is <BRAND>; price is <PRICE>. \n"
feat_template = "The item has following features: <CATEGORIES>. \n"
desc_template = "The item has following descriptions: <DESCRIPTION>. \n"

data = json.load(open("item2attributes.json", "r"))

all_feats = []

for user, user_attris in data.items():
    for feat_name in user_attris.keys():
        if feat_name not in all_feats:
            all_feats.append(feat_name)

def get_attri(item_str, attri, item_info):
    if attri not in item_info.keys():
        new_str = item_str.replace(f"<{attri.upper()}>", "unknown")
    else:
        new_str = item_str.replace(f"<{attri.upper()}>", str(item_info[attri]))
    return new_str

def get_feat(item_str, feat, item_info):
    if feat not in item_info.keys():
        return ""
    
    assert isinstance(item_info[feat], list)
    feat_str = ""
    for meta_feat in item_info[feat][0]:
        feat_str = feat_str + meta_feat + "; "
    new_str = item_str.replace(f"<{feat.upper()}>", feat_str)

    if len(new_str) > 2048:
        return new_str[:2048]
    return new_str

item_data = {}
for key, value in tqdm(data.items()):
    item_str = copy.deepcopy(prompt_template)
    item_str = get_attri(item_str, "title", value)
    item_str = get_attri(item_str, "brand", value)
    item_str = get_attri(item_str, "date", value)
    item_str = get_attri(item_str, "price", value)

    feat_str = copy.deepcopy(feat_template)
    feat_str = get_feat(feat_str, "categories", value)
    desc_str = copy.deepcopy(desc_template)
    desc_str = get_attri(desc_str, "description", value)
    
    item_data[key] = item_str + feat_str + desc_str

json.dump(item_data, open("item_str.json", "w"))

item_data = json.load(open("item_str.json", "r"))

def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open(data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)

id_map = json.load(open("id_map.json", "r"))["item2id"]
json_data = []
for key, value in item_data.items():
    json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

save_data("item_str.jsonline", json_data)

client = OpenAI(
    api_key='',   
    base_url=""
)

def get_response_batch(prompts, max_retries=3):
    for retry in range(max_retries):
        try:
            completion = client.embeddings.create(
                model="",
                input=prompts  
            )
            return [item.embedding for item in completion.data]
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(2 ** retry) 
                continue
            else:
                raise e

def process_batch(batch_items, item_emb):
    keys = []
    prompts = []
    
    for key, value in batch_items:
        if key not in item_emb:
            if len(value) > 4096:
                value = value[:4095]
            keys.append(key)
            prompts.append(value)
    
    if not prompts:
        return {}
    
    try:
        embeddings = get_response_batch(prompts)
        return dict(zip(keys, embeddings))
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return {}

def get_embeddings_optimized(item_data, batch_size=10, max_workers=5):
    if os.path.exists("item_emb.pkl"):
        item_emb = pickle.load(open("item_emb.pkl", "rb"))
    else:
        item_emb = {}
    
    pending_items = [(k, v) for k, v in item_data.items() if k not in item_emb]
    
    if not pending_items:
        return item_emb
     
    batches = [pending_items[i:i + batch_size] for i in range(0, len(pending_items), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, batch, item_emb): batch for batch in batches}
        
        with tqdm(total=len(pending_items), desc="Generate Embeddings") as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    item_emb.update(batch_results)
                    pbar.update(len(batch_results))
                    
                    if len(item_emb) % 100 == 0:
                        pickle.dump(item_emb, open("item_emb.pkl", "wb"))
                        
                except Exception as e:
                    print(f"Batch Processing Exception: {e}")
                    continue
    
    pickle.dump(item_emb, open("item_emb.pkl", "wb"))
    
    return item_emb

item_emb = get_embeddings_optimized(item_data, batch_size=30, max_workers=4)
id_map = json.load(open("id_map.json", "r"))["id2item"]
emb_list = []
for id in range(1, len(item_emb)+1):
    meta_emb = item_emb[id_map[str(id)]]
    emb_list.append(meta_emb)

emb_list = np.array(emb_list)
pickle.dump(emb_list, open("itm_emb_np.pkl", "wb"))
np.save("itm_emb_np.npy", emb_list)