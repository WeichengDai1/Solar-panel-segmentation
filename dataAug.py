import numpy as np
import cv2

X_train = np.load('X_train.npy')
# print(X_train.shape)
x1 = np.squeeze(X_train[0:0 + 1, :, :, :])
# cv2.imshow("ori", x1/255)
mat90 = np.rot90(x1, 1)
# cv2.imshow("rot", mat90/255)
mat180 = np.rot90(mat90, 1)
mat270 = np.rot90(mat180, 1)
x1 = x1[np.newaxis, :, :, :]
mat90 = mat90[np.newaxis, :, :, :]
mat180 = mat180[np.newaxis, :, :, :]
mat270 = mat270[np.newaxis, :, :, :]
x1 = np.concatenate((x1, mat90), axis=0)
x1 = np.concatenate((x1, mat180), axis=0)
x1 = np.concatenate((x1, mat270), axis=0)

for i in range(X_train.shape[0]):
    x2 = np.squeeze(X_train[i:i + 1, :, :, :])
    # cv2.imshow("ori", x1/255)
    mat90 = np.rot90(x2, 1)
    mat180 = np.rot90(mat90, 1)
    mat270 = np.rot90(mat180, 1)
    x2 = x2[np.newaxis, :, :, :]
    mat90 = mat90[np.newaxis, :, :, :]
    mat180 = mat180[np.newaxis, :, :, :]
    mat270 = mat270[np.newaxis, :, :, :]
    x1 = np.concatenate((x1, x2), axis=0)
    x1 = np.concatenate((x1, mat90), axis=0)
    x1 = np.concatenate((x1, mat180), axis=0)
    x1 = np.concatenate((x1, mat270), axis=0)
    print(str(i)+' finished.\n')

np.save('X_train_aug.npy', x1)
del x1
x1 = np.load('X_train_aug.npy')
print(x1.shape)

