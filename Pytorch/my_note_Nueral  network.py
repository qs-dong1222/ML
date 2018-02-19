import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



############################
# given data
############################
x = np.linspace(-10, 10, 100).reshape(100, 1)
x = Variable(t.FloatTensor(x))
noise = Variable(t.rand(100, 1) * 0.3)
y_true = 1e-6 * (-0.7*x + 3*x**2 - 11.7*x**5) + noise

import matplotlib.pyplot as plt
plt.ion()
plt.figure()







class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # conventional mandatory
        # instantiate a fully connected layer object(y=wx+b): n_input=1, n_output=5
        self.fc1 = nn.Linear(1, 5)
        # instantiate a fully connected layer object(y=wx+b): n_input=5, n_output=3
        self.fc2 = nn.Linear(5, 3)
        # instantiate a fully connected layer object(y=wx+b): n_input=3, n_output=1
        self.fc3 = nn.Linear(3, 1)

    def forward(self, *input):
        featureSet = input[0]
        fc1_out = self.fc1(featureSet)  # 这里其实很奇怪,我们用nn.Linear()去实例化了一个对象fc,但这个fc并不是一个method
        fc1_act_out = F.relu(fc1_out)   # 理论上说它其实不能像method一样带入参数,可是这里它却可以,它在后台直接调用了fc的内部forward()
        fc2_out = self.fc2(fc1_act_out)
        fc2_act_out = F.relu(fc2_out)
        fc3_out = self.fc3(fc2_act_out)
        ###  fc1_out = self.fc1.forward(featureSet)  # 同上, 但是更具体
        ###  fc1_act_out = F.relu(fc1_out)
        ###  fc2_out = self.fc2.forward(fc1_act_out)
        ###  fc2_act_out = F.relu(fc2_out)
        ###  fc3_out = self.fc3.forward(fc2_act_out)
        return fc3_out




# construct model
model = Model()



# show model layer information
print(model)
"""
Model(
  (fc1): Linear(in_features=1, out_features=5)
  (fc2): Linear(in_features=5, out_features=3)
  (fc3): Linear(in_features=3, out_features=1)
)
"""


# show concrete params info in each layer
for param in model.parameters():
    print(param)
    """
    ---------------------------------
    | 1st layer weights param[0][0] |
    ---------------------------------
    Parameter containing:
    -0.8299
    -0.1563
    -0.6600
    -0.3073
    -0.3275
    [torch.FloatTensor of size 5x1]

    ------------------------------
    | 1st layer bias param[0][1] |
    ------------------------------
    Parameter containing:
    -0.3176
    -0.8726
    -0.4748
     0.9796
    -0.5064
    [torch.FloatTensor of size 5]

    ---------------------------------
    | 2nd layer weights param[1][0] |
    ---------------------------------
    Parameter containing:
     0.3889 -0.0554 -0.2784  0.1555  0.2139
     0.0540 -0.2617  0.4375  0.3975 -0.3555
     0.4037 -0.0578  0.3562 -0.1874 -0.0269
    [torch.FloatTensor of size 3x5]

    ------------------------------
    | 2nd layer bias param[1][1] |
    ------------------------------
    Parameter containing:
     0.1434
    -0.3156
     0.1610
    [torch.FloatTensor of size 3]

    -------------------------
    | 3rd layer weights ... |
    -------------------------
    Parameter containing:
     0.5598 -0.5605  0.4874
    [torch.FloatTensor of size 1x3]

    ---------------------
    | 3rd layer bias ... |
    ---------------------
    Parameter containing:
    -0.4154
    [torch.FloatTensor of size 1]
    """



y_pred = model(x)
print("initial y_pred:\n", y_pred)



import torch.optim as optim
opt = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()

x = x.cuda()
y_true = y_true.cuda()
loss_func = loss_func.cuda()
model = model.cuda()


# training loop:
for train_step in range(5000):
    opt.zero_grad()   # zero the gradient buffers
    y_pred = model(x)
    loss = loss_func(y_pred, y_true)
    if train_step % 50 == 0:
        plt.cla()
        plt.scatter(x.data.cpu().numpy(), y_true.data.cpu().numpy())
        plt.plot(x.data.cpu().numpy(), model(x).data.cpu().numpy(), 'r-', lw=5)
        plt.text(5, 1, "loss=%.4f" % (loss.data.cpu()[0]))
        plt.pause(0.01)

    loss.backward()
    opt.step()    # Does the update







