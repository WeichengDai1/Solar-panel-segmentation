import numpy as np
import torch
X_train = np.load('X_train.npy') / 255
Y_train = np.load('Y_train_aug.npy')

X_test = np.load('X_test.npy') / 255
Y_test = np.load('Y_test.npy')


def IOU(output, label):
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    iou = torch.sum(output[output == label])

    IOU = iou / (torch.sum(output) + torch.sum(label) - iou + 1e-6)
    return IOU.item()


def process_dist(X_test=X_test, X_train=X_train):
    dist_list = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            dist_list[i, j] = np.linalg.norm(X_test[i]-X_train[j])
        print(str(i)+" finished.")
    np.save('dist_list.npy', dist_list)


def knn(k=3, Y_train=Y_train, Y_test=Y_test):
    dist_list = np.load('dist_list.npy')
    accuracy = []
    for i in range(dist_list.shape[0]):
        idx = np.argpartition(dist_list[i].ravel(), -1*k)[0:k]
        ans = np.zeros((idx.shape[0], Y_train.shape[1], Y_train.shape[2], Y_train.shape[3]))
        for j in range(len(idx)):
            # ans = np.add(ans, Y_train[j, :, :, :])
            ans[j] += Y_train[idx[j], :, :, :]
        # ans = np.true_divide(ans, idx.shape[0])
        # ans = np.true_divide(ans, 1)
        accuracy.append(IOU(torch.from_numpy(ans), torch.from_numpy(Y_test[i])))
        print(accuracy[-1])

    return sum(accuracy) / len(accuracy)


# dist_list = np.load('dist_list.npy')
# print(np.argpartition(dist_list[0].ravel(), -1*3)[0:3])
# print('The iou of knn when k=3 is '+str(knn(k=3)))
# print('The iou of knn when k=5 is '+str(knn(k=5)))
print('The iou of knn when k=7 is '+str(knn(k=7)))