import cv2
import numpy as np
import os
import json

def output_to_data(outputs_path:str):
    if len(os.listdir('data/image')) != len(os.listdir(outputs_path)):
        raise "num of picture differ with model outputs"
    with open('data/data.json', 'r') as f:
        json_image = json.load(f)
