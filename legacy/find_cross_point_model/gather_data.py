import random
import json
import os
import shutil

from numpy.ma.core import masked


def gather_data():
    train_json_file_path = './train/data.json'
    valid_json_file_path = './valid/data.json'
    train_image_folder_path = './train/image'
    valid_image_folder_path = './valid/image'
    output_folder = './data'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'image'), exist_ok=True)

    '''
    train 데이터 먼저 처리
    '''

    res_json = {}

    with open(train_json_file_path, 'r') as train_f:
        train_json = json.load(train_f)

    cnt = 0
    for k, v in train_json.items():
        if k == "len":
            continue

        sub_dict = {}

        file_name = v['filename']
        mask = v['mask']

        shutil.copy(file_name, os.path.join(output_folder, 'image') + f'/{cnt}.png')
        sub_dict['filename'] = os.path.join(output_folder, 'image') + f'/{cnt}.png'
        sub_dict['mask'] = mask

        res_json[f'{cnt}'] = sub_dict
        cnt += 1



    with open(valid_json_file_path, 'r') as valid_f:
        valid_json = json.load(valid_f)

    for k, v in valid_json.items():
        if k == "len":
            continue

        sub_dict = {}

        file_name = v['filename']
        mask = v['mask']

        shutil.copy(file_name, os.path.join(output_folder, 'image') + f'/{cnt}.png')
        sub_dict['filename'] = os.path.join(output_folder, 'image') + f'/{cnt}.png'
        sub_dict['mask'] = mask

        res_json[f'{cnt}'] = sub_dict
        cnt += 1

    with open(os.path.join(output_folder, 'data.json'), 'w') as f:
        json.dump(res_json, f, indent=2)

if __name__ == "__main__":
    gather_data()