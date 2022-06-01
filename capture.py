import cv2
import os

filePath = os.path.join("./", "walking.mp4")
capture = cv2.VideoCapture(filePath)

targetFrame = 150
frameLength = 2
count = 0
while capture.isOpened() and count <= targetFrame:
    _, _ = capture.read()
    count += 1

count = 0

while capture.isOpened() and count <= frameLength:
    _, image = capture.read()
    if capture.get(1):
        cv2.imwrite("./img/img%d.png" % count, image)
        count += 1

capture.release()
