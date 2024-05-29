import cv2

# 读取图像
image = cv2.imread('../src/texture/target5.jpg')

# 指定缩放比例（例如，宽度和高度都缩小到原来的50%）
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
dim = (500,500)
# 调整图像大小
resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# 显示原始图像和缩小后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
cv2.imwrite('../src/texture/target5_resized.jpg', resized_image)

# 等待按键按下并关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()