"""
데이터 JSON을 학습용/검증용으로 나누는 유틸.
"""

import json
import os
import random

def train_valid_split(data_json_path:str, train_ratio:float = 0.8, valid_ratio:float = 0.2):
    """
    데이터 JSON을 학습용과 검증용으로 나눈다.

    :param train_ratio: 학습에 쓸 비율
    :param valid_ratio: 검증에 쓸 비율. 합이 1이 아니면 나머지는 사용하지 않는다
                        (데이터 일부만 쓰고 싶을 때).
    :return: (train_json, valid_json) - 각각 0부터 다시 번호를 매긴 dict
    """
    if not os.path.exists(data_json_path):
        raise RuntimeError("No Data Json File Detected")

    with open(data_json_path, 'r') as f:
        data_json = json.load(f)


    # "len"은 개수를 담은 항목이라 데이터가 아니므로 제외한다.
    d_key_list = [k for k in data_json.keys() if k != "len"]
    amount = len(d_key_list)

    # 무작위로 학습용을 뽑고, 나머지를 검증 후보로 삼는다.
    train_key = random.sample(d_key_list, int(amount * train_ratio))
    valid_key = [k for k in d_key_list if k not in train_key]
    if train_ratio + valid_ratio == 1.0:
        pass

    else:
        # 비율 합이 1보다 작으면, 남은 것 중에서 다시 필요한 만큼만 뽑는다.
        tmp_ratio = valid_ratio / (1.0 - train_ratio)
        valid_key = random.sample(valid_key, int(amount * tmp_ratio))

    # Dataset이 0부터의 연속된 번호로 접근하므로 키를 다시 매긴다.
    train_json = {}
    valid_json = {}

    for i, t_key in enumerate(train_key):
        train_json[f'{i}'] = data_json[t_key]
    for i, v_key in enumerate(valid_key):
        valid_json[f'{i}'] = data_json[v_key]

    return train_json, valid_json