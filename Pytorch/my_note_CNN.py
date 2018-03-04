import numpy as np
from torch.autograd import Variable
import torch as t



# hyper parameters
LR = 0.001
BATCH_SIZE = 1000
EPOCH = 50


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        picdict = pickle.load(fo, encoding='bytes')
    return picdict


def showImage(row_img):
    import cv2
    import opencvlib
    img = row_img.reshape(3, 32, 32)
    r, g, b = img[0], img[1], img[2]
    img = cv2.merge([b, g, r])
    img = opencvlib.Resize(img, 200, 200)
    cv2.imshow("win", img)
    opencvlib.WaitEscToExit()
    cv2.destroyAllWindows()




dataDict = unpickle("./cifar-10-data/cifar-10-batches-py/data_batch_1")
train_set_x = dataDict[b'data'].reshape(10000, 3, 32, 32)
train_set_y = dataDict[b'labels']

for i in range(2, 6):
    path = "./cifar-10-data/cifar-10-batches-py/data_batch_" + str(i)
    dataDict = unpickle(path)
    batch_x = dataDict[b'data'].reshape(10000, 3, 32, 32)
    batch_y = dataDict[b'labels']
    train_set_x = np.vstack((train_set_x, batch_x))
    train_set_y += batch_y

dataDict = unpickle("./cifar-10-data/cifar-10-batches-py/test_batch")
test_set_x = dataDict[b'data'].reshape(10000, 3, 32, 32)
test_set_y = dataDict[b'labels']






import torch.utils.data as utdata
train_set_x = t.FloatTensor(train_set_x)/255
train_set_y = t.LongTensor(train_set_y)
dataSet = utdata.TensorDataset(data_tensor=train_set_x, target_tensor=train_set_y)
dataLoader = utdata.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

test_set_x = t.FloatTensor(test_set_x)/255




# model
class Conv(t.nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.model = t.nn.Sequential(
            t.nn.Conv2d(3, 32, (3, 3)),  # 32x32 -> 30x30
            t.nn.ReLU(),
            t.nn.MaxPool2d(2, stride=2),  # 30x30 -> 15x15

            t.nn.Conv2d(32, 64, (3, 3)),  # 15x15 -> 13x13
            t.nn.ReLU(),
            t.nn.MaxPool2d(2, stride=2),  # 13x13 -> 6x6

            t.nn.Conv2d(64, 64, (3, 3)),  # 6x6 -> 4x4
            t.nn.ReLU()
        )

    def forward(self, input):
        output = self.model(input)
        return output




class FullConnect(t.nn.Module):
    def __init__(self):
        super(FullConnect, self).__init__()
        self.model = t.nn.Sequential(
            t.nn.Linear(64 * 4 * 4, 128),
            t.nn.BatchNorm1d(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 64),
            t.nn.BatchNorm1d(64),
            t.nn.ReLU(),
            t.nn.Linear(64, 10),
            t.nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output








if __name__ == '__main__':
    conv_model = Conv().cuda()
    fc_model = FullConnect().cuda()

    # loss + cuda
    loss_func = t.nn.NLLLoss().cuda()

    # optimizer
    opt_conv = t.optim.Adam(conv_model.parameters(), lr=LR)
    opt_fc = t.optim.Adamax(fc_model.parameters(), lr=LR)

    for eopch in range(EPOCH):
        for batch_Nbr, (batch_xs, batch_ys) in enumerate(dataLoader):
            batch_xs_cuda = Variable(batch_xs).cuda()
            batch_ys_cuda = Variable(batch_ys).cuda()

            opt_conv.zero_grad()
            opt_fc.zero_grad()
            y_pred = conv_model(batch_xs_cuda)
            y_pred = y_pred.view(-1, 64*4*4)
            y_pred = fc_model(y_pred)

            loss = loss_func(y_pred, batch_ys_cuda)

            if (batch_Nbr % (BATCH_SIZE/2) == 0):
                print("loss = %.4f" % (loss.data.cpu()[0]))
                predition = conv_model(Variable(test_set_x[0:1000]).cuda())
                predition = predition.view(-1, 64 * 4 * 4)
                predition = fc_model(predition)
                predition = predition.cpu().data.numpy()
                pred_labels = np.argmax(predition, axis=1)
                true_labels = test_set_y[0:1000]
                acc = np.sum(pred_labels == true_labels) / 1000
                print("acc = %.4f" % acc)

            loss.backward()
            opt_conv.step()
            opt_fc.step()



    y_pred = conv_model(Variable(train_set_x[0:2]).cuda())
    y_pred = y_pred.view(-1, 64*4*4)
    y_pred = fc_model(y_pred)
    print(y_pred)
    print(train_set_y[0:2])


