import http.client
import requests
import cv2
import numpy as np
import io
from PIL import Image
import time
import test
from LaneFollow import LaneFollow as lf
import os
import sys
sys.path.append("..")
from pynput.keyboard import Listener, Controller
from streamer.object_detection import detection

from streamer.object_detection.utils import visualization_utils as vis_util
from streamer.object_detection.utils import label_map_util


HOST = '192.168.5.1'
PORT = '8000'

BASE_URL = 'http://' + HOST + ':' + PORT + '/'

pid = lf()

running = False

PATH_TO_LABELS  = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


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


def stop():
    run_action('stop')


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

while(True):
    image = queryImage.queryImage()
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, warped, error = test.process_img(image)

        angle = pid.calcAngle(error)
        angle = np.int(angle)
        run_action('fwturn:{0}'.format(angle))
    except ValueError:
        pass
    # cv2.imshow('Stream', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = test.process_img(image)
    #
    # vertices = np.array([[0, 240], [60, 160], [260, 160], [320, 240]])
    #
    # mask = np.zeros_like(image)
    # cv2.fillPoly(mask, [vertices], (255, 255, 255))
    # masked = cv2.bitwise_and(image, mask)

    #masked = roi(image, [vertices])

    cv2.imshow('Stream', image)
    # cv2.imshow('warped', warped)

    #image_np_expanded = np.expand_dims(image, axis=0)

    # image = cv2.resize(image, (160, 120))
    #
    # output_dict = detection.run_inference_for_single_image(image)
    #
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks'),
    #     use_normalized_coordinates=True,
    #     line_thickness=1
    # )

    # cv2.imshow('Stream', image)
    # cv2.imshow('Stream', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # if cap.grab():
    #     flag, frame = cap.retrieve()
    #     if not flag:
    #         continue
    #     else:
    #         image = np.array(frame)
    #         image = test.process_img(image)
    #         cv2.imshow('test', image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


# PATH = 'test_images'
# TEST_IMAGES = [os.path.join(PATH, 'test{}.jpg'.format(i)) for i in range(1, 9)]
#
#
# #start_time = time.time()
# for image_path in TEST_IMAGES:
#     #plt.figure(figsize=(16, 9))
#     image = np.array(Im.open(image_path))
#     new_image = process_img(image)
#     #plt.imshow(new_image)
#     new_image = Im.fromarray(new_image, 'RGB')
#     new_image.show()
#     #cv2.imshow('window', new_image)
#     #print(time.time() - start_time)
#     #start_time = time.time()
