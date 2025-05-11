import numpy as np

def image_shift(image:np.ndarray, x_offset:int = 0, y_offset:int = 0):
    size = image.shape[:2]
    result = np.zeros(shape = size, dtype = np.uint8)

    x_start = max(0, x_offset)
    x_end = min(size[1], size[1] + x_offset)

    y_start = max(0, y_offset)
    y_end = min(size[0], size[0] + y_offset)

    src_x_start = max(0, -x_offset)
    src_x_end = src_x_start + (x_end - x_start)

    src_y_start = max(0, -y_offset)
    src_y_end = src_y_start + (y_end - y_start)

    result[y_start:y_end, x_start:x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]

    return result

def check_condition(x, x_offset, y, y_offset, size):
    res = True
    x_ = x - x_offset
    y_ = y - y_offset
    condition1 = x_ < 0
    condition2 = y_ < 0
    condition3 = x_ >= size[0]
    condition4 = y_ >= size[1]

    if condition1 or condition2 or condition3 or condition4:
        res = False

    return res