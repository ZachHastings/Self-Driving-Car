import http.client
import requests
import cv2
import numpy as np
import io
from PIL import Image
import LaneDetection
from LaneFollow import LaneFollow as lf
from pynput.keyboard import Listener, Controller


HOST = '192.168.5.1'
PORT = '8000'
BASE_URL = 'http://' + HOST + ':' + PORT + '/'
pid = lf()
running = False


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
    if '{0}'.format(key) == "'g'":
        go()
    if '{0}'.format(key) == "'h'":
        stop()


def on_release(key):
    if '{0}'.format(key) == "'w'":
        run_action('stop')
    if '{0}'.format(key) == "'a'":
        run_action('fwstraight')
    if '{0}'.format(key) == "'s'":
        run_action('stop')
    if '{0}'.format(key) == "'d'":
        run_action('fwstraight')


def go():
    run_action('forward')
    global running
    running = True


def stop():
    run_action('stop')
    global running
    running = False


def connection_ok():
    cmd = 'connection_test'
    url = BASE_URL + cmd

    try:
        r = requests.get(url)
        if r.text == 'OK':
            return True
    except:
        return False


if connection_ok():
    run_action('fwready')
    run_action('bwready')
    run_speed('30')
    queryImage = QueryImage(HOST)
else:
    exit()

keyboard = Controller()
Listener(on_press=on_press, on_release=on_release).start()

cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', 900, 720)

while True:
    image = queryImage.queryImage()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    warped = image
    if (running):
        try:
            image, warped, error = LaneDetection.process_img(image)
            angle = pid.calcAngle(error)
            angle = np.int(angle)
            run_action('fwturn:{0}'.format(angle))
        except ValueError:
            pass

    cv2.imshow('Stream', image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

