import numpy as np
import cv2
from collections import deque
from predict import predict
import extension

def remove_too_close_neighbor(point_list:list):
    result = point_list.copy()
    need_del_idx_list = []
    median_list = []
    for ref_idx, point in enumerate(point_list):
        dist_list = []
        obj, ref_x, ref_y = point
        for idx, (_, x, y) in enumerate(point_list):
            dist = np.sqrt((ref_x - x) ** 2 + (ref_y - y) ** 2)
            dist_list.append([idx, dist])
        dist_list = sorted(dist_list, key = lambda x : x[1])[1:5]
        closest4_list =  [d for _, d in dist_list]
        closest_4_median = np.median(closest4_list)
        median_list.append(closest_4_median)

        for i in [i for i, d in dist_list if d < closest_4_median / 2]:
            if point_list[i][0] <= point[0]:
                need_del_idx_list.append(i)
            else:
                need_del_idx_list.append(ref_idx)

    for idx in sorted(list(set(need_del_idx_list)), reverse=True):
        del result[idx]

    return result, np.mean(median_list)




def draw_line(points, image_size = 1024, is_set_of_points:bool = False):
    if is_set_of_points:
        points = find_point_avg(points)
        points = np.array(points, dtype = np.int32).tolist()
    points, mean_median = remove_too_close_neighbor(points)
    points = [[row, col] for _, col, row in points]
    dist = 24
    result = np.zeros((1024, 1024, 3), dtype = np.uint8)
    # result = cv2.imread('test.png')
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    pallete = extension.random_pallete(99)
    p_idx = 0
    '''
    vertical drawing
    '''
    visited = []
    v_queue = deque()
    v_queue.extend(points)

    while v_queue:
        row, col = map(int, v_queue.pop())

        if [row, col] in visited:
            continue

        target_points_list = [p for p in points if max(0, col - dist) <= p[1] <= min(image_size - 1, col + dist)]

        target_points_list = sorted(target_points_list, key = lambda x : x[0])

        if len(target_points_list) <= 2:
            visited.extend(target_points_list)
            continue

        # start_point_row = target_points_list[0][0]
        # start_point_col = target_points_list[0][1]
        # end_point_row = target_points_list[-1][0]
        # end_point_col = target_points_list[-1][1]
        #
        # conditions = [start_point_row <= mean_median / 2,
        #               start_point_col <= mean_median / 2,
        #               end_point_row >= image_size - 1 - mean_median / 2,
        #               end_point_col >= image_size - 1 - mean_median / 2]
        #
        # if conditions[0]:
        #     target_points_list.insert(0, target_points_list[0].copy())
        #     target_points_list[0][0] = 0
        # if conditions[2]:
        #     target_points_list.append(target_points_list[-1].copy())
        #     target_points_list[-1][0] = image_size -1

        target_points_list.insert(0, target_points_list[0].copy())
        target_points_list[0][0] = 0
        target_points_list.append(target_points_list[-1].copy())
        target_points_list[-1][0] = image_size - 1

        color = tuple(int(c) for c in pallete[p_idx])
        p_idx += 1
        for i in range(len(target_points_list)- 1):
            result = cv2.line(result, target_points_list[i], target_points_list[i+1], thickness = 3, color = (255,255,255))

        for target_points in target_points_list:
            cv2.circle(result, target_points, color = (0,0,255), thickness = -1, radius = 5)

        visited.extend(target_points_list)

    '''
    horizontal drawing
    '''
    visited = []
    h_queue = deque()
    h_queue.extend(points)

    while h_queue:
        row, col = h_queue.pop()

        if [row, col] in visited:
            continue

        target_points_list = [p for p in points if max(0, row - dist) <= p[0] <= min(image_size - 1, row + dist)]

        target_points_list = sorted(target_points_list, key = lambda x : x[1])

        if len(target_points_list) <= 2:
            visited.extend(target_points_list)
            continue

        # start_point_row = target_points_list[0][0]
        # start_point_col = target_points_list[0][1]
        # end_point_row = target_points_list[-1][0]
        # end_point_col = target_points_list[-1][1]
        #
        # conditions = [start_point_row <= mean_median / 2,
        #               start_point_col <= mean_median / 2,
        #               end_point_row >= image_size - 1 - mean_median / 2,
        #               end_point_col >= image_size - 1 - mean_median / 2]
        #
        # if conditions[1]:
        #     target_points_list.insert(0, target_points_list[0].copy())
        #     target_points_list[0][1] = 0
        # if conditions[3]:
        #     target_points_list.append(target_points_list[-1].copy())
        #     target_points_list[-1][1] = image_size-1

        target_points_list.insert(0, target_points_list[0].copy())
        target_points_list[0][1] = 0
        target_points_list.append(target_points_list[-1].copy())
        target_points_list[-1][1] = image_size - 1

        color = tuple(int(c) for c in pallete[p_idx])
        p_idx += 1
        for i in range(len(target_points_list) - 1):
            result = cv2.line(result, target_points_list[i], target_points_list[i+1], thickness = 3, color = (255,255,255))

        for target_points in target_points_list:
            cv2.circle(result, target_points, color = (0,0,255), thickness = -1, radius = 3)

        visited.extend(target_points_list)


    return result

def find_point_avg(points_set):
    res = []

    length = len(points_set[0])

    for i in range(length):
        point_idx = []
        for points in points_set:
            point_idx.append(points[i])

        res.append(np.mean(point_idx, axis = 0))

    return res

if __name__ == '__main__':
    image_list = []
    for i in range(5):
        image = cv2.imread(f'../warp_image/{i+1}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_list.append(image)
    output, result, point_list = predict(image_list)
    output = draw_line(point_list, is_set_of_points=False)
    extension.image_show(output)