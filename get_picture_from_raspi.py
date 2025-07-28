import paramiko
import nmap
import cv2

def find_raspi_ip(target):
    scanner = nmap.PortScanner()
    scanner.scan(hosts=target)

    open_ips = []
    for host in scanner.all_hosts():
        if not host.endswith(".1"):
            open_ips.append(host)

    return open_ips

def get_picture_from_raspi(is_connected:bool):
    if is_connected:
        host = find_raspi_ip("10.42.0.0/24")[0]
        port = 5901
        transprot = paramiko.transport.Transport(host, port)
        userId = "user"
        passwd = "q1w2e3r4"

        transprot.connect(username=userId, password=passwd)
        sftp = paramiko.SFTPClient.from_transport(transprot)

        sftp.get(remotepath="/home/user/test/test_image.jpg", localpath="./raspi_image.jpg")

        sftp.close()
        transprot.close()

    ret = split_image_arr_2_2("./raspi_image.jpg")

    return ret

def split_image_arr_2_2(filepath:str):
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
