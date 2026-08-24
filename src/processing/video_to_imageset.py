"""
동영상에서 프레임을 뽑아 이미지 데이터셋으로 만드는 유틸.

카메라 4대로 동시 촬영하는 대신, 한 대를 움직이며 찍은 동영상에서
서로 다른 각도의 사진 여러 장을 얻으려고 만들었다.
"""

import warnings

import cv2
import numpy as np
import os

def video_to_frame(video_path, output_dir):
    """
    동영상을 일정 간격으로 잘라 frame_00000.jpg 형식으로 저장한다.

    :param video_path: 원본 동영상 경로
    :param output_dir: 프레임을 저장할 폴더 (미리 존재해야 한다)
    """
    frame_interval = 5  # 30프레임마다 1장 저장

    cap = cv2.VideoCapture(video_path)
    frame_count = 0   # 읽은 전체 프레임 수
    saved_count = 0   # 실제로 저장한 프레임 수 (파일명 번호로 쓰인다)

    # 주의: 아래는 output_dir 이 아니라 항상 ./video_frame 을 만든다.
    #       output_dir 을 다른 곳으로 주려면 이 줄도 함께 고쳐야 한다.
    os.makedirs("./video_frame", exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 더 읽을 프레임이 없으면 종료
            break

        # frame_interval 장마다 한 장씩만 저장 (연속 프레임은 거의 동일하므로 건너뛴다)
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f'{saved_count}개의 프레임이 저장되었습니다.')

def target_frame(target_folder,target_frame_idx_list):
    """
    video_to_frame 으로 저장해둔 프레임 중 원하는 번호들만 골라 읽어온다.

    :param target_folder: 프레임이 저장된 폴더
    :param target_frame_idx_list: 사용할 프레임 번호 목록 (예: [0, 5, 11, 19])
    :return: RGB 이미지 리스트 (segmentation 모델이 RGB를 기대하므로 변환해서 반환)
    """
    ret = []
    target_folder_files = os.listdir(target_folder)
    for target_frame_idx in target_frame_idx_list:
        target_frame_name = f"./{target_folder}/frame_{target_frame_idx:05d}.jpg"
        image = cv2.imread(target_frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret.append(image)


    return ret

if __name__ == "__main__":
    # 프로젝트 루트의 video.mp4 를 잘라 ./video_frame 에 저장하는 예시
    pass
    video_to_frame("./video.mp4", "./video_frame")
