import cv2
import test
import numpy as np

cap = cv2.VideoCapture("./project_video.mp4")

while True:
    if cap.grab():
        flag, frame = cap.retrieve()
        if not flag:
            continue
        else:
            image = np.array(frame)
            image = test.process_img(image)
            cv2.imshow('test', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
