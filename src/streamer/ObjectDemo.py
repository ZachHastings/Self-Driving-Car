import http.client
import requests
import cv2
import numpy as np
import io
from PIL import Image
import os
import sys
sys.path.append("..")
from streamer.object_detection import detection

from streamer.object_detection.utils import visualization_utils as vis_util
from streamer.object_detection.utils import label_map_util


HOST = '192.168.5.1'
PORT = '8000'

BASE_URL = 'http://' + HOST + ':' + PORT + '/'

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
    queryImage = QueryImage(HOST)
else:
    exit()


cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', 900, 720)

while(True):
    image = queryImage.queryImage()
    image = cv2.resize(image, (160, 120))

    output_dict = detection.run_inference_for_single_image(image)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=1
    )

    cv2.imshow('Stream', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

