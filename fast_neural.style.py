import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

def style_transfer(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    return output

result = style_transfer(cv2.imread('images/game.jpg'))

cv2.imwrite('images/game_result.jpg', result * 255.0)
cv2.imshow('style', result)
cv2.waitKey(0)
