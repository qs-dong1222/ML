import torch as t
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



####################
# data set
####################
n_samples = 500
n_features = 3
noise = np.random.rand(n_samples, n_features)*12
x1 = noise + np.array([2, 3, 4])
y1_true = np.ones((n_samples, 1)) * 0
x2 = noise + np.array([-7, 10, -4])
y2_true = np.ones((n_samples, 1)) * 1
x3 = noise + np.array([10, -9, 6])
y3_true = np.ones((n_samples, 1)) * 2


ax = plt.figure(1).add_subplot(111, projection='3d')
ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=100, c='r', marker='.')
ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], s=100, c='g', marker='.')
ax.scatter(x3[:, 0], x3[:, 1], x3[:, 2], s=100, c='b', marker='.')



class Model(t.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = t.nn.Linear(3, 10)
        self.fc2 = t.nn.Linear(10, 10)
        self.fc3 = t.nn.Linear(10, 3)

    def forward(self, *input):
        x = input[0]
        x = F.relu(self.fc1.forward(x))
        x = F.relu(self.fc2.forward(x))
        output = F.log_softmax(self.fc3.forward(x))
        return output



model = Model()



x = np.vstack((np.vstack((x1, x2)), x3))
y_true = np.vstack((np.vstack((y1_true, y2_true)), y3_true)).flatten()

x = Variable(t.FloatTensor(x))
y_true = Variable(t.LongTensor(y_true))





import torch.optim as optim
opt = optim.Adam(model.parameters(), lr=0.001)
loss_func = t.nn.NLLLoss()


x = x.cuda()
y_true = y_true.cuda()
loss_func = loss_func.cuda()
model = model.cuda()


# training loop:
for train_step in range(1000):
    opt.zero_grad()   # zero the gradient buffers
    y_pred = model(x)
    loss = loss_func(y_pred, y_true)
    if train_step % 50 == 0:
        print("loss = ", loss)
        predition = y_pred.cpu().data.numpy()
        pred_labels = np.argmax(predition, axis=1)
        true_labels = y_true.cpu().data.numpy()
        acc = np.sum(pred_labels == true_labels) / (n_samples*3)
        print("acc = ", acc)

    loss.backward()
    opt.step()    # Does the update





predition = y_pred.cpu().data.numpy()
pred_labels = np.argmax(predition, axis=1)

# ax = plt.figure(2).add_subplot(111, projection='3d')
x = x.cpu().data.numpy()

for i in range(len(pred_labels)):
    if (pred_labels[i] == 0) :
        color = 'y'
    elif (pred_labels[i] == 1):
        color = 'k'
    else:
        color = 'c'
    ax.scatter(x[i][0], x[i][1], x[i][2], s=100, c=color, marker='.')

plt.show()







# predition = model(x)
# predition = predition.cpu().data.numpy()
# pred_labels = np.argmax(predition, axis=1)
# print(pred_labels)

# x1_v = Variable(t.FloatTensor(x1))
# predition = model.cpu()(x1_v).data.numpy()
# pred_labels = np.argmax(predition, axis=1)
# print(pred_labels)
