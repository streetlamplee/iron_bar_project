import json
import os
import random

def train_valid_split(data_json_path:str, train_ratio:float = 0.8, valid_ratio:float = 0.2):
    if not os.path.exists(data_json_path):
        raise RuntimeError("No Data Json File Detected")

    with open(data_json_path, 'r') as f:
        data_json = json.load(f)


    d_key_list = [k for k in data_json.keys() if k != "len"]
    amount = len(d_key_list)

    train_key = random.sample(d_key_list, int(amount * train_ratio))
    valid_key = [k for k in d_key_list if k not in train_key]
    if train_ratio + valid_ratio == 1.0:
        pass

    else:
        tmp_ratio = valid_ratio / (1.0 - train_ratio)
        valid_key = random.sample(valid_key, int(amount * tmp_ratio))

    train_json = {}
    valid_json = {}

    for i, t_key in enumerate(train_key):
        train_json[f'{i}'] = data_json[t_key]
    for i, v_key in enumerate(valid_key):
        valid_json[f'{i}'] = data_json[v_key]

    return train_json, valid_json