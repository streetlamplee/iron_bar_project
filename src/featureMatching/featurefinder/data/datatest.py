import cv2
import json
from extension import image_show

def main():
    with open("./data.json", "r") as f:
        json_dict = json.load(f)

    for i in range(json_dict["len"]):
        image = cv2.imread(json_dict[f"{i}"]["filename"].replace("data/", ""))
        for p in json_dict[f"{i}"]["mask"]:
            cv2.circle(image, p, color = (0,0,255), thickness = -1, radius = 3)
        image_show(image)

if __name__ == "__main__":
    main()