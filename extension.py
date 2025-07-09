import os

import cv2
import random
import numpy as np
import torch
import json
import re


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

def random_pallete(n, seed = None):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size = (n, 3))).astype(np.uint8)

def array_norm(array:np.ndarray):
    array_min = array.min()
    array_max = array.max()

    return (array - array_min) / (array_max - array_min)


def smart_json_dump(data, fp, indent=2):
    """
    JSON을 파일에 저장하되, 단순한 리스트는 한 줄로 유지.

    Args:
        data: dict 또는 list 등 JSON 직렬화 가능한 객체
        fp: 파일 객체 또는 파일 경로 (str)
        indent: 들여쓰기 레벨
    """
    raw = json.dumps(data, indent=indent, sort_keys=True)
    # 줄바꿈된 단순 리스트를 한 줄로 정리
    cleaned = re.sub(r'\[\s+([^\[\]\n]+?)\s+\]', r'[\1]', raw)

    if isinstance(fp, str):
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    else:
        fp.write(cleaned)