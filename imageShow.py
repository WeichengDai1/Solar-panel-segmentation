import cv2
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt
# import PIL
width = 800
height = 800

# wholeImg = np.random.randint(0, 2, (64, 128, 128, 1))
batch = eval(input("please input the batch size: "))
wholeImg = np.load('tests/predict%s.npy' % batch)
wholeTrue = np.load('tests/true%s.npy' % batch)
wholeInput = np.load('tests/input%s.npy' % batch)
feature1 = np.load('feature1/%s.npy' % batch)
feature2 = np.load('feature2/%s.npy' % batch)
feature3 = np.load('feature3/%s.npy' % batch)
feature4 = np.load('feature4/%s.npy' % batch)

value = eval(input("please input the image number(0~63): "))
img = wholeImg[value:value + 1, :, :, :]
true = wholeTrue[value:value + 1, :, :, :]
myInput = wholeInput[value:value + 1, :, :, :]
feature1 = feature1[value:value + 1, :, :, :]
feature2 = feature2[value:value + 1, :, :, :]
feature3 = feature3[value:value + 1, :, :, :]
feature4 = feature4[value:value + 1, :, :, :]

img = np.squeeze(img)
true = np.squeeze(true)
myInput = np.squeeze(myInput) / 255

cv2.namedWindow('input', 0)
cv2.resizeWindow('input', 256, 256)
cv2.imshow('input', myInput)

cv2.namedWindow('predicted', 0)
cv2.resizeWindow('predicted', 256, 256)
cv2.imshow('predicted', img)

cv2.namedWindow('true', 0)
cv2.resizeWindow('true', 256, 256)
cv2.imshow('true', true)

# for i in range(64):
#     plt.subplot(8, 8, i+1)
#     plt.imshow(np.squeeze(feature1[:, :, :, i:i+1]))
#     plt.title(i, fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
#     # cv2.imshow('features%s' % i, np.squeeze(feature3[:, :, :, i:i+1]))
# plt.show()

# for i in range(128):
#     plt.subplot(16, 8, i+1)
#     plt.imshow(np.squeeze(feature2[:, :, :, i:i+1]))
#     # plt.title(i, fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
#     # cv2.imshow('features%s' % i, np.squeeze(feature3[:, :, :, i:i+1]))
# plt.show()

# for i in range(256):
#     plt.subplot(16, 16, i+1)
#     plt.imshow(np.squeeze(feature3[:, :, :, i:i+1]))
#     # plt.title(i, fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
#     # cv2.imshow('features%s' % i, np.squeeze(feature3[:, :, :, i:i+1]))
# plt.show()

# for i in range(512):
#     plt.subplot(32, 16, i+1)
#     plt.imshow(np.squeeze(feature4[:, :, :, i:i+1]))
#     # plt.title(i, fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
#     # cv2.imshow('features%s' % i, np.squeeze(feature3[:, :, :, i:i+1]))
# plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
