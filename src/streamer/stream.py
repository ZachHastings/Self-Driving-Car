from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
import http.client
import requests
import sys

screen = "screen.ui"

Ui_Screen, QtBaseClass = uic.loadUiType(screen)

TIMEOUT = 200
MAX_SPEED = 100
MIN_SPEED = 40
SPEED_LEVEL_1 = MIN_SPEED
SPEED_LEVEL_2 = (MAX_SPEED - MIN_SPEED) / 4 * 1 + MIN_SPEED
SPEED_LEVEL_3 = (MAX_SPEED - MIN_SPEED) / 4 * 2 + MIN_SPEED
SPEED_LEVEL_4 = (MAX_SPEED - MIN_SPEED) / 4 * 3 + MIN_SPEED
SPEED_LEVEL_5 = MAX_SPEED
SPEED = [0, SPEED_LEVEL_1, SPEED_LEVEL_2, SPEED_LEVEL_3, SPEED_LEVEL_4, SPEED_LEVEL_5]

HOST = '192.168.5.1'
PORT = '8000'

BASE_URL = 'http://' + HOST + ':' + PORT + '/'

def __relash_url__():
    global BASE_URL
    BASE_URL = 'http://' + HOST + ':' + PORT + '/'


class Screen(QtWidgets.QDialog, Ui_Screen):

    def __init__(self):

        QtWidgets.QDialog.__init__(self)
        Ui_Screen.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("Self Driving Car Client")

        self.speed_level = 0

        self.level_btn_show(self.speed_level)
        self.btn_back.setStyleSheet("border-image: url(./images/back_unpressed.png);")
        self.btn_setting.setStyleSheet("border-image: url(./images/settings_unpressed.png);")

        self.setup()

    def setup(self):
        if connection_ok() == True:
            self.start_stream()
            return True
        else:
            return False

    def start_stream(self):
        self.queryImage = QueryImage(HOST)
        self.timer = QTimer(timeout=self.reflash_frame)
        self.timer.start(TIMEOUT)

        run_action('fwready')
        run_action('bwready')

    def stop_stream(self):
        self.timer.stop()

    def transToPixmap(self):
        data = self.queryImage.queryImage()
        if not data:
            return None
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        return pixmap

    def reflash_frame(self):
        pixmap = self.transToPixmap()
        if pixmap:
            self.label_snapshot.setPixmap(pixmap)
        else:
            return

    def set_speed_level(self, speed):
        run_speed(speed)

    def level_btn_show(self, speed_level):
        """Reflash the view of level_btn

        Whit this function call, all level_btns change to a unpressed status except one that be clicked recently

        Args:
            1~5, the argument speed_level  means the button be clicked recently
        """
        # set all buttons stylesheet unpressed
        self.level1.setStyleSheet("border-image: url(./images/speed_level_1_unpressed.png);")
        self.level2.setStyleSheet("border-image: url(./images/speed_level_2_unpressed.png);")
        self.level3.setStyleSheet("border-image: url(./images/speed_level_3_unpressed.png);")
        self.level4.setStyleSheet("border-image: url(./images/speed_level_4_unpressed.png);")
        self.level5.setStyleSheet("border-image: url(./images/speed_level_5_unpressed.png);")
        if speed_level == 1:  # level 1 button is pressed
            self.level1.setStyleSheet("border-image: url(./images/speed_level_1_pressed.png);")
        elif speed_level == 2:  # level 2 button is pressed
            self.level2.setStyleSheet("border-image: url(./images/speed_level_2_pressed.png);")
        elif speed_level == 3:  # level 3 button is pressed
            self.level3.setStyleSheet("border-image: url(./images/speed_level_3_pressed.png);")
        elif speed_level == 4:  # level 4 button is pressed
            self.level4.setStyleSheet("border-image: url(./images/speed_level_4_pressed.png);")
        elif speed_level == 5:  # level 5 button is pressed
            self.level5.setStyleSheet("border-image: url(./images/speed_level_5_pressed.png);")

    def keyPressEvent(self, event):
        """Keyboard press event

        Effective key: W, A, S, D, ↑,  ↓,  ←,  →
        Press a key on keyboard, the function will get an event, if the condition is met, call the function
        run_action().

        Args:
            event, this argument will get when an event of keyboard pressed occured

        """
        key_press = event.key()

        # don't need autorepeat, while haven't released, just run once
        if not event.isAutoRepeat():
            if key_press == Qt.Key_Up:  # up
                run_action('camup')
            elif key_press == Qt.Key_Right:  # right
                run_action('camright')
            elif key_press == Qt.Key_Down:  # down
                run_action('camdown')
            elif key_press == Qt.Key_Left:  # left
                run_action('camleft')
            elif key_press == Qt.Key_W:  # W
                run_action('forward')
            elif key_press == Qt.Key_A:  # A
                run_action('fwleft')
            elif key_press == Qt.Key_S:  # S
                run_action('backward')
            elif key_press == Qt.Key_D:  # D
                run_action('fwright')

    def keyReleaseEvent(self, event):
        """Keyboard released event

        Effective key: W,A,S,D, ↑,  ↓,  ←,  →
        Release a key on keyboard, the function will get an event, if the condition is met, call the function
        run_action().

        Args:
            event, this argument will get when an event of keyboard release occured

        """
        # don't need autorepeat, while haven't pressed, just run once
        key_release = event.key()
        if not event.isAutoRepeat():
            if key_release == Qt.Key_Up:  # up
                run_action('camready')
            elif key_release == Qt.Key_Right:  # right
                run_action('camready')
            elif key_release == Qt.Key_Down:  # down
                run_action('camready')
            elif key_release == Qt.Key_Left:  # left
                run_action('camready')
            elif key_release == Qt.Key_W:  # W
                run_action('stop')
            elif key_release == Qt.Key_A:  # A
                run_action('fwstraight')
            elif key_release == Qt.Key_S:  # S
                run_action('stop')
            elif key_release == Qt.Key_D:  # D
                run_action('fwstraight')


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

        return returnmsg.read()


def connection_ok():
    cmd = 'connection_test'
    url = BASE_URL + cmd

    try:
        r = requests.get(url)
        if r.text == 'OK':
            return True
    except:
        return False


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


def main():
    app = QtWidgets.QApplication(sys.argv)

    screen = Screen()

    screen.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    screen = Screen()

    screen.show()

    sys.exit(app.exec_())
