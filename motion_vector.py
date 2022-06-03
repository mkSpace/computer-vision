import cv2
import numpy as np
import numpy.linalg


WINDOW_SIZE = 9
DERIVATIVE_FILTER_X = [[-1, 0, 1]]
DERIVATIVE_FILTER_Y = [[-1], [0], [1]]
t0_image = cv2.imread("img/img0.png", 1)
t0_image_grey = cv2.imread("img/img0.png", 0)
t1_image_grey = cv2.imread("img/img1.png", 0)

height, width, _ = t0_image.shape
dx = cv2.filter2D(t0_image_grey, kernel=np.array(DERIVATIVE_FILTER_X), ddepth=cv2.CV_64F)
dy = cv2.filter2D(t0_image_grey, kernel=np.array(DERIVATIVE_FILTER_Y), ddepth=cv2.CV_64F)
dt = np.zeros((height, width))

for y in range(height):
    for x in range(width):
        dt[y][x] = float(t1_image_grey[y][x]) - float(t0_image_grey[y][x])
N = int(WINDOW_SIZE / 2)
result = np.zeros(shape=(height, width, 2))
for y in range(N, height - N):
    for x in range(N, width - N):
        derivative_y = dy[(y - N):(y + N + 1), (x - N):(x + N + 1)]
        derivative_x = dx[(y - N):(y + N + 1), (x - N):(x + N + 1)]
        derivative_t = dt[(y - N):(y + N + 1), (x - N):(x + N + 1)]

        A = np.concatenate(
            [np.reshape(derivative_y, (WINDOW_SIZE ** 2, 1)), np.reshape(derivative_x, (WINDOW_SIZE ** 2, 1))],
            axis=-1
        )
        b = -np.reshape(derivative_t, (WINDOW_SIZE ** 2, 1))

        A_TA = np.matmul(A.T, A)
        A_TB = np.matmul(A.T, b)
        try:
            v_T = np.matmul(np.linalg.inv(A_TA), A_TB)
            result[y][x] = v_T.T
        except:
            result[y][x] = np.array([0., 0.])

for y in range(N, height - N, N):
    for x in range(N, width - N, N):
        v, u = result[y][x]
        source = (x, y)
        destination = (x + int(u), y + int(v))
        if source != destination:
            background = cv2.arrowedLine(img=t0_image, pt1=source, pt2=destination,
                                         color=(0, 0, 255), tipLength=0.1)
cv2.imshow('result', t0_image)
cv2.waitKey()
