import cv2
import threading
import time
from collections import deque
from iron_bar_segmentation.predict import predict as seg_predict

frames = deque(maxlen=10)
lock = threading.Lock()
def get_frame():
    url = "rtsp://admin:q1w2e3r4@192.168.1.100:554/Streaming/Channels/201/"
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1600, 900))
        with lock:
            frames.append(frame)

    cap.release()

def main():
    thr = threading.Thread(target = get_frame, daemon=True)
    thr.start()

    time.sleep(1)

    while True:
        if not frames:
            time.sleep(0.01)
            continue
        with lock:
            output = frames.pop()
        output = seg_predict(output)
        cv2.imshow('NVR Stream', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()