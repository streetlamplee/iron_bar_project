import warnings

import cv2
import numpy as np
import os

def video_to_frame(video_path, output_dir):
    frame_interval = 5  # 30프레임마다 1장 저장

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    os.makedirs("./video_frame", exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f'{saved_count}개의 프레임이 저장되었습니다.')

def target_frame(target_folder,target_frame_idx_list):
    ret = []
    target_folder_files = os.listdir(target_folder)
    for target_frame_idx in target_frame_idx_list:
        target_frame_name = f"./{target_folder}/frame_{target_frame_idx:05d}.jpg"
        image = cv2.imread(target_frame_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret.append(image)


    return ret

if __name__ == "__main__":
    pass
    video_to_frame("./video.mp4", "./video_frame")
