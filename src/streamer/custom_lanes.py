import http.client
import requests
import cv2
import numpy as np
import io
from PIL import Image
import time
import sys
sys.path.append("..")
from pynput.keyboard import Listener, Controller


HOST = '192.168.5.1'
PORT = '8000'

BASE_URL = 'http://' + HOST + ':' + PORT + '/'


class QueryImage:

    def __init__(self, host, port=8080, argv='/?action=snapshot'):
        self.host = host
        self.port = port
        self.argv = argv

    def queryImage(self):
        http_data = http.client.HTTPConnection(self.host, self.port)
        http_data.putrequest('GET', self.argv)
        http_data.putheader('Host', self.host)
        http_data.putheader('User-agent', 'python-http.client')
        http_data.putheader('Content-type', 'image/jpeg')
        http_data.endheaders()
        returnmsg = http_data.getresponse()
        returnmsg = returnmsg.read()
        returnmsg = io.BytesIO(returnmsg)
        returnmsg = Image.open(returnmsg)
        img = np.array(returnmsg)

        return img


def __request__(url, times=10):
    for x in range(times):
        try:
            requests.get(url)
            return 0
        except:
            return
    return -1


def run_action(cmd):
    url = BASE_URL + 'run/?action=' + cmd
    __request__(url)


def run_speed(speed):
    url = BASE_URL + 'run/?speed=' + speed
    __request__(url)


def on_press(key):
    if '{0}'.format(key) == "'w'":
        run_action('forward')
    if '{0}'.format(key) == "'a'":
        run_action('fwleft')
    if '{0}'.format(key) == "'s'":
        run_action('backward')
    if '{0}'.format(key) == "'d'":
        run_action('fwright')


def on_release(key):
    if '{0}'.format(key) == "'w'":
        run_action('stop')
    if '{0}'.format(key) == "'a'":
        run_action('fwstraight')
    if '{0}'.format(key) == "'s'":
        run_action('stop')
    if '{0}'.format(key) == "'d'":
        run_action('fwstraight')


def connection_ok():
    cmd = 'connection_test'
    url = BASE_URL + cmd

    try:
        r = requests.get(url)
        if r.text == 'OK':
            return True
    except:
        return False


def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


def draw_lines(image, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), (100, 100, 255), 2)
    except:
        pass


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=40, threshold2=40)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    vertices = np.array([[0, 240], [0, 160], [80, 80], [240, 80], [320, 160], [320, 240], [200, 240], [160, 140], [120, 240]])
    processed_img = roi(processed_img, [vertices])

    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 60, 5)
    # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    draw_lines(original_image, lines)

    return original_image


queryImage = QueryImage(HOST)

keyboard = Controller()
Listener(on_press=on_press, on_release=on_release).start()

sum = 0
count = 0

# cap = cv2.VideoCapture("./project_video.mp4")
cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', 900, 720)

# last_time = time.time()
while(True):
    image = queryImage.queryImage()
    # image = np.array(Image.open('test_images/screen.png'))
    new_image = process_img(image)

    cv2.imshow('Stream', new_image)

    # diff = time.time() - last_time
    # fps = 1 / diff
    # sum += fps
    # count += 1
    # if count == 10:
    #     print('{0:.3g}'.format(sum / count))
    #     sum = 0
    #     count = 0
    # last_time = time.time()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
