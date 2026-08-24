"""
라즈베리파이에서 촬영 사진을 받아오는 모듈.

카메라 4대가 붙은 라즈베리파이가 한 장의 큰 사진(2x2로 붙은 형태)을 저장하면,
그 파일을 네트워크로 받아 카메라별 4장으로 잘라 돌려준다.

추가 패키지 필요: paramiko(SFTP), python-nmap(IP 검색)
"""

import paramiko
import nmap
import cv2

def find_raspi_ip(target):
    """
    같은 네트워크 대역을 훑어 라즈베리파이로 보이는 IP를 찾는다.
    IP가 고정되어 있지 않아 매번 검색한다.

    :param target: 검색할 대역 (예: "10.42.0.0/24")
    :return: 게이트웨이(.1)를 제외한 응답 IP 목록
    """
    scanner = nmap.PortScanner()
    scanner.scan(hosts=target)

    open_ips = []
    for host in scanner.all_hosts():
        if not host.endswith(".1"):
            open_ips.append(host)

    return open_ips

def get_picture_from_raspi(is_connected:bool):
    """
    :param is_connected: True면 라즈베리파이에 접속해 새 사진을 받아온다.
                         False면 이전에 받아둔 raspi_image.jpg 를 그대로 사용한다
                         (장비 없이 개발할 때 쓰는 경로).
    :return: 카메라별로 나눈 이미지 4장
    """
    if is_connected:
        host = find_raspi_ip("10.42.0.0/24")[0]
        port = 5901
        transprot = paramiko.transport.Transport(host, port)
        userId = "user"
        passwd = "q1w2e3r4"

        # 주의: 접속 정보가 코드에 그대로 적혀 있다. 외부에 공개할 경우 반드시 분리할 것.
        transprot.connect(username=userId, password=passwd)
        sftp = paramiko.SFTPClient.from_transport(transprot)

        # 라즈베리파이가 찍어둔 사진을 현재 폴더로 내려받는다.
        sftp.get(remotepath="/home/user/test/test_image.jpg", localpath="./raspi_image.jpg")

        sftp.close()
        transprot.close()

    ret = split_image_arr_2_2("./raspi_image.jpg")

    return ret

def split_image_arr_2_2(filepath:str):
    """
    2x2로 붙어 있는 한 장의 사진을 카메라 4대의 사진으로 잘라낸다.

    :return: [좌상단, 우상단, 좌하단, 우하단] 순서의 RGB 이미지 4장
    """
    image_arr = cv2.imread(filepath)
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    height, width = image_arr.shape[:2]
    image1 = image_arr.copy()[0         :height//2  ,0         :width//2]
    image2 = image_arr.copy()[0         :height//2  ,width//2  :width]
    image3 = image_arr.copy()[height//2 :height     ,0         :width//2]
    image4 = image_arr.copy()[height//2 :height     ,width//2  :width]

    ret = [image1, image2, image3, image4]

    return ret

if __name__ == "__main__":
    get_picture_from_raspi()
