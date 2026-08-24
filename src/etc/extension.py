"""
프로젝트 전반에서 쓰는 공통 유틸 모음.

이미지 표시, 재현성을 위한 seed 고정, 학습 checkpoint 탐색, 로그 출력 등
특정 모듈에 속하지 않는 잡다한 함수들을 모아둔 곳이다.
"""

import os

import cv2
import random
import numpy as np
import torch
import json
import re


def get_latest_pth_file(base_dir, extension):
    """
    base_dir 아래를 모두 뒤져 가장 최근에 수정된 파일을 찾는다.

    학습 checkpoint(.pth)를 고를 때 사용한다. 즉 "마지막으로 학습한 모델"이 자동 선택되므로,
    특정 모델을 쓰려면 파일을 직접 지정하거나 최신 파일이 맞는지 확인해야 한다.

    :param base_dir: 탐색 시작 폴더
    :param extension: 찾을 확장자 (예: '.pth')
    :return: base_dir 기준 상대경로. 못 찾으면 None
    """
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
    """
    이미지를 창으로 띄운다. delay=0 이면 아무 키나 누를 때까지 멈춘다(블로킹).
    GUI가 없는 환경에서는 에러가 나므로 호출 지점의 isShow 플래그로 막아두고 쓴다.
    """
    cv2.imshow(title, image)
    cv2.waitKey(delay)
    cv2.destroyWindow(title)

def set_seed(seed: int = 42):
    """학습 결과를 재현할 수 있도록 모든 난수 생성기의 seed를 고정한다."""
    random.seed(seed)  # Python random seed 설정
    np.random.seed(seed)  # NumPy random seed 설정
    torch.manual_seed(seed)  # PyTorch CPU 시드 설정
    torch.cuda.manual_seed(seed)  # PyTorch GPU 시드 설정 (한 개의 GPU 사용 시)
    torch.cuda.manual_seed_all(seed)  # PyTorch 다중 GPU 사용 시 모든 GPU에 같은 seed 설정
    torch.backends.cudnn.deterministic = True  # CuDNN deterministic 설정
    torch.backends.cudnn.benchmark = False  # 성능보다 재현성을 우선할 경우 False로 설정


import os
import sys
# 로그 앞에 실행 스크립트 이름을 붙여, 어느 단계에서 나온 출력인지 구분하기 위한 함수들
def print_with(s:str):
    print(f"[{os.path.basename(sys.argv[0]):^20}] {s}")

def str_with(s:str):
    return f"[{os.path.basename(sys.argv[0]):^20}] {s}"

def random_pallete(n, seed = None):
    """서로 구분되는 색 n개를 무작위로 만든다. 검출된 철근을 색깔로 구분해 그릴 때 사용."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size = (n, 3))).astype(np.uint8)

def array_norm(array:np.ndarray):
    """배열을 최솟값 0, 최댓값 1 범위로 펴준다. 시각화 직전에 주로 사용한다."""
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

'''
@brief CamAraayHat을 이용하는 라즈베리파이 이미지를 분류하기 위한 Dict 생성 함수 (warp_point_finder에서 사용)
@param idx CamArray위치, (0: 좌상단, 1: 우상단, 2: 좌하단, 3: 우하단)
@return boolean masking 용 (2160, 3840) ndarray
'''
def CamArrayIdx(idx):
    # 4K(3840x2160) 한 장에 카메라 4대의 화면이 2x2로 붙어 들어온다.
    # 그중 idx 번째 카메라 영역의 [y시작, y끝, x시작, x끝]을 돌려준다.
    res = [0,0,0,0]
    if idx == 0:
        res[0] = 0
        res[1] = 1080
        res[2] = 0
        res[3] = 1920
    elif idx == 1:
        res[0] = 0
        res[1] = 1080
        res[2] = 1920
        res[3] = 3840
    elif idx == 2:
        res[0] = 1080
        res[1] = 2160
        res[2] = 0
        res[3] = 1920
    elif idx == 3:
        res[0] = 1080
        res[1] = 2160
        res[2] = 1920
        res[3] = 3840
    else:
        print(f"Not Valid param 'idx', 'idx' Must in range(0, 4)")
        return res

    return res
