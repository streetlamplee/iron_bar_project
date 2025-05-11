import numpy as np
import cv2
import matplotlib.pyplot as plt

threshold_value = 128

def define_layer():
    ### 디버그용 입력 인자 설정 (끝나면 인자로 추가할 것)
    input = []
    for i in range(4):
        img = cv2.imread(f"tmp_warp/warp{i}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input.append(img)
    ###
    input = np.array(input)

    for idx, inp in enumerate(input):
        inp = np.where((inp < threshold_value), 0, 255) # 철근이면 0, 배경이면 255
        h, w = inp.shape

        vertical_gap_list = []
        vertical_gap = 0
        vertical_gap_interval = 1
        ret = find_gap(inp, 1, idx)
        x_values = list(range(120, 120+len(ret)))
        plt.plot(x_values, ret, marker='o', linestyle='-', color='b')
        plt.xlabel("gap")
        plt.ylabel("non-zero")
        plt.title("non-zero per gap")
        plt.grid(True)
        plt.show()
        break


def find_gap(input, gap_interval, num_of_image):
    '''
    :param input: warp perspective / binary 처리 된 사진 1장
    :param gap_interval: gap 을 찾을 때 늘어나는 간격 조정
    :return: 가능한 gap의 min 값
    '''
    gap = 120
    h, w = input.shape
    ret = []
    if np.max(input) != 1:
        input = np.astype(input, np.float32)
        input = input / 255
    if h != w:
        raise "1:1 비율의 warp perspective 이미지를 넣어주세요"
    for _ in range(100):
        result = np.zeros((h,w), np.float32)
        for i in range(h):
            for j in range(w):
                top_left = input[i - gap][j - gap]  if i-gap >= 0 and j-gap >= 0 else 1.0
                top_middle = input[i - gap][j]      if i-gap >= 0 else 1.0
                top_right = input[i - gap][j + gap] if i-gap >= 0 and j+gap < h else 1.0
                middle_left = input[i][j - gap]     if j-gap >= 0 else 1.0
                middle_middle = input[i][j]
                middle_right = input[i][j + gap]    if j + gap < h else 1.0
                btm_left = input[i + gap][j - gap]  if i + gap < h and j-gap >= 0 else 1.0
                btm_middle = input[i + gap][j]      if i + gap < h else 1.0
                btm_right = input[i + gap][j + gap] if i + gap < h and j + gap < h else 1.0

                like_aspp = np.array([[top_left, top_middle, top_right],
                                      [middle_left, middle_middle, middle_right],
                                      [btm_left, btm_middle, btm_right]], dtype = np.float32)
                kernel = np.array([[1,1,1],
                                   [1,1,1],
                                   [1,1,1]], dtype = np.float32)

                res = np.sum(like_aspp * kernel) / 9.
                result[i][j] = res
        print(f"now gap is {gap}")
        result *= 255
        result = result.astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        result = cv2.putText(result, f"gap: {gap}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,200,230), 2)
        # cv2.imshow("result", result)
        # cv2.waitKey(5000)
        # cv2.destroyWindow("result")
        cv2.imwrite(f"tmp_result/{num_of_image}_gap{gap}.png", result)
        gap += gap_interval
        ret.append(np.count_nonzero(result))
    return ret


if __name__ == "__main__":
    define_layer()