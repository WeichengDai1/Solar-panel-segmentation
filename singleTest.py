import numpy as np
# from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

transform_train = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

class UNet(nn.Module):
    def __init__(self,colordim =1):
        super(UNet, self).__init__()
        self.dropout = nn.Dropout(p = 0.3)
        #self.bn0 = nn.BatchNorm2d(3)
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding = 1, padding_mode = 'replicate')  # input of (n,n,3), output of (n,n,64)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding = 1, padding_mode = 'replicate')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding = 1, padding_mode = 'replicate')
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding = 1, padding_mode = 'replicate')
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding = 1, padding_mode = 'replicate')
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding = 1, padding_mode = 'replicate')
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding = 1, padding_mode = 'replicate')
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1, padding_mode = 'replicate')
        self.upconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn5_out = nn.BatchNorm2d(512)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding = 1, padding_mode = 'replicate')
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding = 1, padding_mode = 'replicate')
        # self.upconv6 = nn.Conv2d(512, 256, 1)
        self.upconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn6_out = nn.BatchNorm2d(256)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding = 1, padding_mode = 'replicate')
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding = 1, padding_mode = 'replicate')
        # self.upconv7 = nn.Conv2d(256, 128, 1)
        self.upconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn7_out = nn.BatchNorm2d(128)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        # self.upconv8 = nn.Conv2d(128, 64, 1)
        self.upconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn8_out = nn.BatchNorm2d(64)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding = 1, padding_mode = 'replicate')
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding = 1, padding_mode = 'replicate')
        self.conv9_3 = nn.Conv2d(32, colordim, 1)
        self.bn9 = nn.BatchNorm2d(colordim)
        self.bn10 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def getf1(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x1)))))
        return x1

    def getf2(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(self.bn0(x1))))))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        return x2

    def getf3(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(self.bn0(x1))))))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        x3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.maxpool(x2))))))
        return x3

    def getf4(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(self.bn0(x1))))))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        x3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.maxpool(x2))))))
        x4 = F.relu(self.bn4(self.conv4_2(F.relu(self.conv4_1(self.maxpool(x3))))))
        return x4 

    def getfinal(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(self.bn0(x1))))))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        x3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.maxpool(x2))))))
        x4 = F.relu(self.bn4(self.conv4_2(F.relu(self.conv4_1(self.maxpool(x3))))))
        xup = F.relu(self.conv5_2(F.relu(self.conv5_1(self.maxpool(x4)))))  # x5
        xup = self.bn5(self.upconv5(xup))  # x6in
        # print('x6 size: %d'%(xup.size(2)))
        cropidx = (x4.size(2) - xup.size(2)) // 2
        x4 = x4[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x4.size(2),xup.size(2)))
        xup = self.bn5_out(torch.cat((x4, xup), 1))  # x6 cat x4
        xup = F.relu(self.conv6_2(F.relu(self.conv6_1(xup))))  # x6out

        xup = self.dropout(self.bn6(self.upconv6(xup)))  # x7in
        # print('xup1: %d%d'%(torch.max(xup),torch.min(xup)))
        cropidx = (x3.size(2) - xup.size(2)) // 2
        x3 = x3[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x3.size(2),xup.size(2)))
        xup = self.bn6_out(torch.cat((x3, xup), 1))  # x7 cat x3
        xup = F.relu(self.conv7_2(F.relu(self.conv7_1(xup))))  # x7out

        xup = self.dropout(self.bn7(self.upconv7(xup)))  # x8in
        # print('xup2: %d%d'%(torch.max(xup),torch.min(xup)))
        cropidx = (x2.size(2) - xup.size(2)) // 2
        x2 = x2[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x2.size(2),xup.size(2)))
        xup = self.bn7_out(torch.cat((x2, xup), 1))  # x8 cat x2
        xup = F.relu(self.conv8_2(F.relu(self.conv8_1(xup))))  # x8out

        xup = self.dropout(self.bn8(self.upconv8(xup)))  # x9in
        # print('xup3: %d%d'%(torch.max(xup),torch.min(xup)))
        cropidx = (x1.size(2) - xup.size(2)) // 2
        x1 = x1[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x1.size(2),xup.size(2)))
        xup = self.bn8_out(torch.cat((x1, xup), 1))  # x9 cat x1
        # print('xup4.1:%d,%d'%(torch.max(xup),torch.min(xup)))
        xup = F.relu(self.bn10((self.conv9_1(xup))))

        # print('xup4.2:%d,%d'%(torch.max(xup),torch.min(xup)))
        xup = F.relu(self.bn11((self.conv9_2(xup))))
        # print('xup4.3:%d,%d'%(torch.max(xup),torch.min(xup)))
        xup = self.conv9_3(xup)
        # print('xup4.4:%d,%d'%(torch.max(xup),torch.min(xup)))
        # xup = self.conv9_3(F.relu(self.conv9_2(F.relu(self.conv9_1(xup)))))  # x9out
        return xup

    def forward(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(self.bn0(x1))))))
        # self.f1 = x1
        # print('x1 size: %d'%(x1.size(2)))
        # print('x1: %d%d'%(torch.max(x1),torch.min(x1)))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        # self.f2 = x2
        # print('x2 size: %d'%(x2.size(2)))
        # print('x2: %d%d'%(torch.max(x2),torch.min(x2)))
        x3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.maxpool(x2))))))
        # self.f3 = x3
        # print('x3 size: %d'%(x3.size(2)))
        # print('x3: %d%d'%(torch.max(x3),torch.min(x3)))
        x4 = F.relu(self.bn4(self.conv4_2(F.relu(self.conv4_1(self.maxpool(x3))))))
        # self.f4 = x4
        # print('x4 size: %d'%(x4.size(2)))
        # print('x4: %d%d'%(torch.max(x4),torch.min(x4)))
        xup = F.relu(self.conv5_2(F.relu(self.conv5_1(self.maxpool(x4)))))  # x5
        # print('x5 size: %d'%(xup.size(2)))

        xup = self.bn5(self.upconv5(xup))  # x6in
        # print('x6 size: %d'%(xup.size(2)))
        cropidx = (x4.size(2) - xup.size(2)) // 2
        x4 = x4[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x4.size(2),xup.size(2)))
        xup = self.bn5_out(torch.cat((x4, xup), 1))  # x6 cat x4
        xup = F.relu(self.conv6_2(F.relu(self.conv6_1(xup))))  # x6out

        xup = self.dropout(self.bn6(self.upconv6(xup)))  # x7in
        # print('xup1: %d%d'%(torch.max(xup),torch.min(xup)))
        cropidx = (x3.size(2) - xup.size(2)) // 2
        x3 = x3[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x3.size(2),xup.size(2)))
        xup = self.bn6_out(torch.cat((x3, xup), 1))  # x7 cat x3
        xup = F.relu(self.conv7_2(F.relu(self.conv7_1(xup))))  # x7out

        xup = self.dropout(self.bn7(self.upconv7(xup)))  # x8in
        # print('xup2: %d%d'%(torch.max(xup),torch.min(xup)))
        cropidx = (x2.size(2) - xup.size(2)) // 2
        x2 = x2[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x2.size(2),xup.size(2)))
        xup = self.bn7_out(torch.cat((x2, xup), 1))  # x8 cat x2
        xup = F.relu(self.conv8_2(F.relu(self.conv8_1(xup))))  # x8out

        xup = self.dropout(self.bn8(self.upconv8(xup)))  # x9in
        # print('xup3: %d%d'%(torch.max(xup),torch.min(xup)))
        cropidx = (x1.size(2) - xup.size(2)) // 2
        x1 = x1[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x1.size(2),xup.size(2)))
        xup = self.bn8_out(torch.cat((x1, xup), 1))  # x9 cat x1
        # print('xup4.1:%d,%d'%(torch.max(xup),torch.min(xup)))
        xup = F.relu(self.bn10((self.conv9_1(xup))))

        # print('xup4.2:%d,%d'%(torch.max(xup),torch.min(xup)))
        xup = F.relu(self.bn11((self.conv9_2(xup))))
        # print('xup4.3:%d,%d'%(torch.max(xup),torch.min(xup)))
        xup = self.conv9_3(xup)
        # print('xup4.4:%d,%d'%(torch.max(xup),torch.min(xup)))
        # xup = self.conv9_3(F.relu(self.conv9_2(F.relu(self.conv9_1(xup)))))  # x9out
        # print(xup.shape)
        # print('xup4: %d,%d'%(torch.min(xup),torch.max(xup)))
        # print(xup[0,0,30,30])
        xup = torch.sigmoid(self.bn12(xup))
        # print(xup[0,0,30,30])
        # xup = xup.view(32*128*128)
        # xup = F.softmax(xup,dim = 0)
        return xup


unet = UNet()
X_test=np.load('X_test.npy')
Y_test=np.load('Y_test.npy')

X_test = X_test.transpose((0,3,1,2))
Y_test = Y_test.transpose((0,3,1,2))

batch_size=64
torch_dataset = Data.TensorDataset(torch.from_numpy(X_test).float(),torch.from_numpy(Y_test).float())
loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size)

unet = nn.DataParallel(unet, device_ids = [0, 1], output_device = 0)
unet = unet.cuda(0)
unet.load_state_dict(torch.load('model/97.pt'))

i = 0
for x, y in loader:
    x, y = transform_train(x).cuda(0), y.cuda(0)
    with torch.no_grad():
        predict = unet(x)
    # x = x.cpu().numpy().transpose((0, 2, 3, 1))
    # if(isinstance(unet, torch.nn.DataParallel)):
    #     unet = unet.module
    x1 = unet.module.getf1(x).data.cpu().numpy().transpose((0, 2, 3, 1))
    x2 = unet.module.getf2(x).data.cpu().numpy().transpose((0, 2, 3, 1))
    x3 = unet.module.getf3(x).data.cpu().numpy().transpose((0, 2, 3, 1))
    x4 = unet.module.getf4(x).data.cpu().numpy().transpose((0, 2, 3, 1))
    xfinal = unet.module.getfinal(x).data.cpu().numpy().transpose((0, 2, 3, 1))
    predict = predict.cpu().numpy().transpose((0, 2, 3, 1))
    predict[predict>=0.5]=1
    predict[predict<0.5]=0
    y = y.cpu().numpy().transpose((0, 2, 3, 1))
    np.save('tests/input%s' % i, x)
    np.save('tests/predict%s' % i, predict)
    np.save('tests/true%s' % i, y)
    np.save('feature1/%s' % i, x1)
    np.save('feature2/%s' % i, x2)
    np.save('feature3/%s' % i, x3)
    np.save('feature4/%s' % i, x4)
    np.save('featurefinal/%s' % i, xfinal)
    print('Number %s finished.' % i)
    i+=1

print('finished operating')
