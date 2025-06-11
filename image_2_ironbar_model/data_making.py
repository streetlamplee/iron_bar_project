import json
import os
import shutil
import cv2
import numpy as np
import torch

import extension
import n_make_image_nice
from extension import smart_json_dump, image_show
from find_cross_point_model.predict import predict_one_image
from image_2_ironbar_model.manual_data_making import InteractiveLineDrawer, clicked_point_finder
from point_align import icp_rigid_trimmed_numpy

def make_target(image_list, image_size, save_filename=None):
    if len(image_list) == 0:
        return None, None, None
    src_image = cv2.imread(image_list[0])
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    res = np.zeros_like(src_image, dtype = np.float32)
    res += src_image * (1/len(image_list))

    _, src_point = predict_one_image(src_image)
    # image_show(_)

    for i in range(1, len(image_list)):
        tgt_image = cv2.imread(image_list[i])
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
        _, tgt_point = predict_one_image(tgt_image)
        # image_show(_)
        src, R, t, prev_error, trim_idx = icp_rigid_trimmed_numpy(src_point, tgt_point, max_iter=200, tolerance=1e-6, trim_ratio=0.90)

        M = np.hstack([R, t.reshape(2,1)])
        tgt_image_align = cv2.warpAffine(tgt_image, M, dsize=(image_size,image_size))
        res += tgt_image_align * (1/len(image_list))

    h_kernel = np.ones((1, 16), np.uint8)
    v_kernel = np.ones((16, 1), np.uint8)
    h_res = n_make_image_nice.custom_image_thresholding(res, h_kernel, int(16 * 0.3))
    v_res = n_make_image_nice.custom_image_thresholding(res, v_kernel, int(16 * 0.3))

    res_t = np.maximum(h_res, v_res)

    if save_filename is None:
        pass
    else:
        cv2.imwrite(f'{save_filename}_target_t.png', res_t)
        cv2.imwrite(f'{save_filename}_target.png', res)

    return res_t, res, f'{save_filename}_target.png'


def image_grouping(image_list:list, output_json_dict:dict, image_path:str):
    new_dict = dict()
    old_keys = []
    for image in image_list:
        group_num = image.split('_')[0]
        if group_num in output_json_dict.keys():
            old_keys.append(group_num)
            continue
        else:
            if group_num not in new_dict.keys():
                new_dict[group_num] = {}
                new_dict[group_num]['images'] = [os.path.join(image_path,image)]
            if len(new_dict[group_num]['images']) >= 5:
                continue
            new_dict[group_num]['images'].append(os.path.join(image_path,image))
            new_dict[group_num]['images'].sort()

    output_json_dict.update(new_dict)
    return output_json_dict, old_keys

def data_add(image_input_path, image_output_path, json_output_path):
    json_result = dict()
    output_image_list = []
    if  os.path.exists(json_output_path):
        with open(json_output_path, 'r') as f:
            json_result = json.load(f)
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path, exist_ok = True)
    else:
        output_image_list = os.listdir(image_output_path)
        output_image_list = [image for image in output_image_list if image.endswith('.png')]

    input_image_list = os.listdir(image_input_path)
    input_image_list = [image for image in input_image_list if image.endswith('.png')]

    json_result, old_keys = image_grouping(input_image_list, json_result, image_output_path)
    for filepath in input_image_list:
        filename = os.path.basename(filepath)
        target_path = os.path.join(image_output_path, filename)
        shutil.copy(os.path.join(image_input_path, filename), target_path)

    for key, value in json_result.items():
        input_list = [inp for inp in value['images'] if inp.split('/')[-1].split('_')[0] not in old_keys]
        target_t, target, tgt_filename = make_target(input_list,
                                                     256,
                                                     f'{image_output_path}/{key}')
        if target_t is None and target is None and tgt_filename is None:
            continue

        value['result'] = tgt_filename

        json_result[key] = value

    with open(json_output_path, 'w') as f:
        smart_json_dump(json_result, f, 2)

    return 1

def data_processing(h_lines, v_lines, image_size = (256, 256), grid_size = (8,8)):
    res = torch.zeros(size=(4, *grid_size), dtype = torch.float32)
    grid_height = image_size[0] / grid_size[0]
    grid_width = image_size[1] / grid_size[1]
    for h_line in h_lines:
        grid_num_line = int(h_line // grid_height)
        value_line = (h_line % grid_height) / grid_height

        res[0, grid_num_line, :] = 1.0
        res[1, grid_num_line, :] = value_line

    for v_line in v_lines:
        grid_num_line = int(v_line // grid_width)
        value_line = (v_line % grid_width) / grid_width

        res[2, :, grid_num_line] = 1.0
        res[3, :, grid_num_line] = value_line

    return res
IMAGE_SIZE = 1024
CROP_SIZE = 256
NUM_CROPS = 256
def crop_and_filter_points(points, crop_x, crop_y, crop_size=256):
    """
    이미지에서 crop 영역 내에 존재하는 점들만 필터링하는 함수
    """
    filtered_points = []
    for point in points:
        x, y = point
        if crop_x <= x < crop_x + crop_size and crop_y <= y < crop_y + crop_size:
            # crop 영역 내 좌표로 변환
            filtered_points.append([x - crop_x, y - crop_y])
    return filtered_points


def main():
    import random
    os.makedirs('data/origin', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 예시 그룹 번호 생성
    if len(os.listdir('data/processed')) == 0:
        group_num = 0
    else:
        group_num = sorted(list(map(int, [img.split("_")[0] for img in os.listdir('data/processed') if img.endswith('.png')])))[-1] +1

    # 1. 타겟 이미지 생성
    warp_image_path = '../warp_image'
    image_list = sorted(os.listdir(warp_image_path))
    image_list = [os.path.join(warp_image_path, img) for img in image_list if img.endswith('.png')]

    _, target_image, _ = make_target(image_list, IMAGE_SIZE)

    # 2. 타겟 이미지에서 점 클릭
    target_image_rgb = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    clicked_points = clicked_point_finder(target_image_rgb)

    # 3. 랜덤 crop 수행 및 각 crop에 해당하는 점 필터링
    data_json = {}
    json_path = 'data/processed/data.json'
    if  os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data_json = json.load(f)

    #[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]
    for c in range(group_num, group_num+NUM_CROPS):
        crop_x = random.randint(0, IMAGE_SIZE - CROP_SIZE)
        crop_y = random.randint(0, IMAGE_SIZE - CROP_SIZE)
        image_path_list = []

        # crop 이미지 생성
        for i, image_path in enumerate(image_list):
            image = cv2.imread(image_path)
            crop_filename = f"{c}_{i}.png"
            crop_image = image[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]
            cv2.imwrite(f"data/origin/{crop_filename}", crop_image)
            cv2.imwrite(f"data/processed/{crop_filename}", crop_image)
            image_path_list.append(f'data/processed/{crop_filename}')
            # crop 영역에 해당하는 점 필터링
            filtered_points = crop_and_filter_points(clicked_points, crop_x, crop_y)
        crop_target_image = target_image_rgb[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]
        cv2.imwrite(f"data/processed/{crop_filename.replace('.png', '_target.png')}", crop_target_image)
        # 아래 딕셔너리를 json 형태로 저장하면 됩니다
        sub_dict = {}
        sub_dict['images'] = image_path_list
        sub_dict['points'] = filtered_points
        sub_dict['result'] = f"data/processed/{crop_filename.replace('.png', '_target.png')}"

        data_json[c] = sub_dict

    # ✅ JSON으로 저장하고 싶을 경우 이 블럭을 사용하세요
    with open(json_path, 'w') as f:
        smart_json_dump(data_json, f, indent=2)

if __name__ == '__main__':
    main()
    # import random
    # os.makedirs('data/origin', exist_ok=True)
    # os.makedirs('data/processed', exist_ok=True)
    # images_path = '../warp_image'
    # image_list = os.listdir(images_path)
    # image_list = sorted(image_list)
    # if len(os.listdir('data/processed')) == 0:
    #     group_num = 0
    # else:
    #     group_num = int(sorted([img.split("_")[0] for img in os.listdir('data/processed') if img.split("_")[0] != 'data.json'])[-1]) + 1
    # c = 0
    # cnt = 16
    # while c < cnt:
    #     start_row = random.randint(0, 1023 - 256)
    #     start_col = random.randint(0, 1023 - 256)
    #     for i, image in enumerate(image_list):
    #         image = cv2.imread(os.path.join(images_path, image))
    #         image = image[start_row:start_row+256, start_col:start_col+256]
    #         cv2.imwrite(os.path.join('data/origin', f'{group_num}_{i}.png'), image)
    #
    #     c += 1
    #     group_num += 1
    #
    # data_add('data/origin', 'data/processed', 'data/processed/data.json')
    #
    # '''
    # manual data
    # '''
    # with open('data/processed/data.json', 'r') as f:
    #     data_json = json.load(f)
    #
    # for key in data_json.keys():
    #     if 'target_h' in data_json[key].keys() and 'target_v' in data_json[key].keys():
    #         continue
    #     image = cv2.imread(data_json[key]['result'])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     height, width, _ = image.shape
    #
    #     points = clicked_point_finder(image)
    #
    #     # data_json[key]['target_h'] = h_line
    #     # data_json[key]['target_v'] = v_line
    #
    #     data_json[key]['points'] = points
    #
    #
    # with open('data/processed/data.json', 'w') as f:
    #     extension.smart_json_dump(data_json, f, indent=2)
