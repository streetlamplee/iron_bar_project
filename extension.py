import os

import cv2
import random
import numpy as np
import torch


def get_latest_pth_file(base_dir, extension):
    latest_path = None
    latest_mtime = -1

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = full_path

    if latest_path:
        return os.path.relpath(latest_path, base_dir)  # 상대경로로 반환
    return None

def image_show(image, title='',  delay = 0):
    cv2.imshow(title, image)
    cv2.waitKey(delay)
    cv2.destroyWindow(title)

def set_seed(seed: int = 42):
    random.seed(seed)  # Python random seed 설정
    np.random.seed(seed)  # NumPy random seed 설정
    torch.manual_seed(seed)  # PyTorch CPU 시드 설정
    torch.cuda.manual_seed(seed)  # PyTorch GPU 시드 설정 (한 개의 GPU 사용 시)
    torch.cuda.manual_seed_all(seed)  # PyTorch 다중 GPU 사용 시 모든 GPU에 같은 seed 설정
    torch.backends.cudnn.deterministic = True  # CuDNN deterministic 설정
    torch.backends.cudnn.benchmark = False  # 성능보다 재현성을 우선할 경우 False로 설정


import os
import sys
def print_with(s:str):
    print(f"[{os.path.basename(sys.argv[0]):^20}] {s}")

def str_with(s:str):
    return f"[{os.path.basename(sys.argv[0]):^20}] {s}"