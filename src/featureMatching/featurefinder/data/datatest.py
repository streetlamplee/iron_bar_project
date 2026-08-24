"""
라벨링 결과(data.json)가 제대로 만들어졌는지 사진 위에 그려 확인하는 도구.
"""

import cv2
import json
from extension import image_show

def main():
    """data.json에 기록된 점 좌표가 사진 위 올바른 위치에 찍혀 있는지 눈으로 확인한다."""
    with open("./data.json", "r") as f:
        json_dict = json.load(f)

    for i in range(json_dict["len"]):
        image = cv2.imread(json_dict[f"{i}"]["filename"].replace("data/", ""))
        # "mask"에 담긴 교차점 좌표를 빨간 점으로 찍어본다.
        for p in json_dict[f"{i}"]["mask"]:
            cv2.circle(image, p, color = (0,0,255), thickness = -1, radius = 3)
        image_show(image)

if __name__ == "__main__":
    main()