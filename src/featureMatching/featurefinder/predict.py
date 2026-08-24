"""
학습된 교차점 검출 모델로 철근 교차점을 찾는 추론 코드.

predict()          : 폴더 안 이미지를 일괄 처리해 result/ 에 결과를 저장
predict_one_image(): 이미지 한 장을 받아 (교차점 이미지, 좌표 목록)을 돌려준다.
                     다른 모듈에서 호출하는 쪽은 이 함수다.

주의: 모델 경로가 예전 절대경로/상대경로로 하드코딩되어 있어 그대로는 동작하지 않는다.
"""

import os
from torchvision import transforms
import numpy as np
import extension
from extension import image_show
from featureMatching.featurefinder.model import pointFindingModel
import torch
import cv2
from find_cross_point_model.nms import nms

def extract_keypoints_from_tensor(tensor, image_size, max_points_per_cell=4, threshold=0.5):
    """
    모델이 낸 격자 텐서를 이미지 좌표 목록으로 되돌린다.
    (nms.py 의 export_point 와 같은 역할을 하는 사본이다.)
    """
    B, C, H, W = tensor.shape
    assert B == 1, "배치 크기는 1"
    grid_size = H
    stride = image_size // grid_size

    keypoints = []

    # 텐서 shape: (H, W, C)
    tensor = tensor.squeeze(0).permute(1, 2, 0)

    for gy in range(grid_size):
        for gx in range(grid_size):
            cell = tensor[gy, gx]  # shape: (C,)
            for i in range(max_points_per_cell):
                offset = i * 3
                rel_x = cell[offset + 0]
                rel_y = cell[offset + 1]
                objectness = cell[offset + 2]


                abs_x = min((gx + rel_x.item()) * stride, image_size - 1)
                abs_y = min((gy + rel_y.item()) * stride, image_size - 1)
                keypoints.append((abs_x, abs_y, float(objectness)))

    return keypoints


def predict():
    """폴더 안의 이미지를 모두 처리해, 교차점을 찍은 결과를 result/ 에 저장한다."""
    test_image_folder = '../warp_image'
    test_image_list = os.listdir(test_image_folder)
    model_folder = './models'
    # model_filename = os.path.join(model_folder, extension.get_latest_pth_file(model_folder, '.pth'))
    # 특정 checkpoint를 직접 지정한다 (위 주석은 최신 파일을 자동 선택하던 방식).
    model_filename = os.path.join(model_folder, '20250527_142708/epoch00808.pth')
    print(model_filename)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_filename)
    model = pointFindingModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    test_image_list.sort()
    for test_image in test_image_list:
        t_image = cv2.imread(os.path.join(test_image_folder, test_image))
        h, w, _ = t_image.shape
        t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB)
        # 학습 때와 같은 전처리: 0~1 스케일 -> ImageNet 정규화 -> (1, C, H, W)
        t_image = torch.tensor(t_image, dtype = torch.float32)
        t_image /= 255
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        t_image = t_image.permute(2,0,1).unsqueeze(0)
        t_image = normalize(t_image)
        t_image = t_image.to(device)
        with torch.no_grad():
            output = model(t_image)
            print(output.shape)
            output = torch.sigmoid(output)
            # 중복 예측 제거. 여기서는 50픽셀 이내의 점을 같은 점으로 본다.
            keypoints = nms(output, h, 50)
        res = cv2.imread(os.path.join(test_image_folder, test_image))
        # 주의: 아래에서 변수 h를 좌표로 덮어쓴다. 반복문 위쪽의 이미지 높이 h와 이름이 겹친다.
        for keypoint in keypoints:
            h, w, o = keypoint
            h = int(h)
            w = int(w)
            # 확신도 0.5 미만은 교차점으로 인정하지 않는다.
            if o >= .5:
                cv2.circle(res, (h,w), 2, (0,0,255), -1)
        os.makedirs('result', exist_ok=True)
        cv2.imwrite(os.path.join('result', test_image), res)
        # extension.image_show(res, title=test_image)
    return

def predict_one_image(image:np.ndarray, model_file = None): # 'find_cross_point_model/models/20250527_142708/epoch00808.pth'
    """
    이미지 한 장에서 교차점을 찾는다.

    :param image: 입력 이미지 (grayscale 또는 RGB)
    :param model_file: 사용할 checkpoint 경로. None이면 models 폴더의 최신 파일을 쓴다.
    :return: (교차점을 흰 점으로 찍은 이미지, 좌표 목록 [[x, y], ...])
    """
    model = pointFindingModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_file is None:
        model_folder = '/home/user/PycharmProjects/iron_bar_sample_project/featureMatching/featurefinder/models'
        model_filename = os.path.join(model_folder, extension.get_latest_pth_file(model_folder, '.pth'))
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # 어떤 형식이 들어와도 되도록 일단 grayscale로 통일해 크기를 얻은 뒤,
    # 모델 입력에 맞춰 다시 3채널로 되돌린다.
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor /= 255.
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output_logit = model(image_tensor)
        output = torch.sigmoid(output_logit)
        # 이쪽은 15픽셀 기준으로 중복을 거른다 (교차점 간격이 더 촘촘한 경우를 위해).
        keypoints = nms(output, h, 15)

    # 원본 위가 아니라 빈 검은 화면에 점만 찍어 돌려준다 (이후 처리에서 쓰기 쉽도록).
    res = np.zeros_like(image, dtype = np.uint8)
    res_point = []
    for key in keypoints:
        h, w, o = key
        h = int(h)
        w = int(w)
        if o > 0.5:
            cv2.circle(res, (h, w), 4, (255,255,255), -1)
            res_point.append([h, w])
    return res, res_point

    # ------------------------------------------------------------------
    # 아래 주석 블록은 큰 이미지를 256 조각으로 잘라 한 번에 추론하던 이전 방식이다.
    # 현재는 이미지 전체를 한 번에 넣는 방식으로 대체되었다.
    # ------------------------------------------------------------------
    # res_points = []
    # res = np.zeros_like(image)
    # h, w = image.shape
    # patches = []
    # position = []
    # for i in range(0, h, 256):
    #     for j in range(0, w, 256):
    #         h_end = min(h, i + 256)
    #         w_end = min(w, j + 256)
    #         c = image[i:h_end, j:w_end]
    #         if len(c.shape) == 2:
    #             c = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)
    #
    #         image_tensor = torch.tensor(c, dtype = torch.float32)
    #         image_tensor /= 255.
    #         image_tensor = image_tensor.permute(2,0,1)
    #
    #         patches.append(image_tensor)
    #         position.append((i,j))
    #
    # batch_tensor = torch.stack(patches).to(device)
    #
    #
    # normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # batch_tensor = normalize(batch_tensor)
    # with torch.no_grad():
    #     output_batch = model(batch_tensor)
    #     # print(output.shape)
    #     output_batch = torch.sigmoid(output_batch)
    #
    # for idx, output in enumerate(output_batch):
    #     i, j = position[idx]
    #     keypoints = nms(output.unsqueeze(0), output.shape[-1], 15)
    #     sub_res = np.zeros((256,256,3), dtype=np.uint8)
    #     # tmp_ = image[i:i+256, j:j+256].copy()
    #     for y, x, o in keypoints:
    #         if o >= 0.6:
    #             sub_res = cv2.circle(sub_res, (int(y), int(x)), 4, (255,255,255), -1)
    #             res_points.append([i+y, j+x])
    #             # tmp_ = cv2.circle(tmp_, (int(y), int(x)), 3, (0,0,255), -1)
    #     sub_res = cv2.cvtColor(sub_res, cv2.COLOR_BGR2GRAY)
    #     # tmp_ = tmp_.astype(np.uint8)
    #     # extension.image_show(tmp_)
    #     h_end = min(h, i + 256)
    #     w_end = min(w, j + 256)
    #     res[i:h_end, j:w_end] = sub_res[0:h_end - i, 0:w_end-j]
    #
    #
    # return res, res_pointsasdfasdfㅁㄴㅇㄻㄴㅇㄻㄴㅇㄹasdfasdfasdfasdf



if __name__ == "__main__":
    # 단독 실행 확인용: 4분할 사진에서 일부 영역만 잘라 교차점을 찍어본다.
    for i in range(3):
        image = cv2.imread('./test.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[0 + 1080 * (i // 2) :1080 + 1080 * (i // 2), 0 + 1920 * (i % 2):1920 + 1920 * (i % 2)]
        # image = cv2.resize(image, (512, 512))
        result, result_point = predict_one_image(image[512:512+512, 880:880+512])
        image_show(result)
        image = image[512:512+512, 880:880+512]
        for p in result_point:
            image = cv2.circle(image, p, radius=3, thickness=-1, color = (255, 0, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_show(image)