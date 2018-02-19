import numpy as np
import torch as t
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt


# hyper params
BATCH_SIZE = 50
EPOCH = 1000
LR = 0.001

############################
# given data
############################
n_samples = 200
x = t.FloatTensor(np.linspace(-10, 10, n_samples).reshape(n_samples, 1))
noise = t.rand(n_samples, 1) * 0.3
y_true = 1e-6 * (-0.7*x + 3*x**2 - 11.7*x**5) + noise


dataSet = data.TensorDataset(data_tensor=x, target_tensor=y_true)
dataLoader = data.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True)  # num_workers is number
                                                                                  # of processes we want to apply to do this batch job




# model + cuda
model = t.nn.Sequential(
    t.nn.Linear(1, 10),
    t.nn.BatchNorm1d(10),
    t.nn.ReLU(),
    t.nn.Linear(10, 10),
    t.nn.BatchNorm1d(10),
    t.nn.ReLU(),
    t.nn.Linear(10, 1)
).cuda()

# loss + cuda
loss_func = t.nn.MSELoss().cuda()

# optimizer
opt = t.optim.Adam(model.parameters(), lr=LR)


plt.ion()
plt.figure(1)



for eopch in range(EPOCH):
    for batch_Nbr, (batch_xs, batch_ys) in enumerate(dataLoader):
        batch_xs_cuda = Variable(batch_xs).cuda()
        batch_ys_cuda = Variable(batch_ys).cuda()

        opt.zero_grad()
        y_pred = model(batch_xs_cuda)
        loss = loss_func(y_pred, batch_ys_cuda)
        if (batch_Nbr%(BATCH_SIZE/2)==0):
            plt.cla()
            plt.scatter(x.numpy(), y_true.numpy())
            plt.plot(x.numpy(), model(Variable(x).cuda()).data.cpu().numpy(), 'r-', lw=5)
            plt.text(5, 1, "loss=%.4f" % (loss.data.cpu()[0]))
            plt.pause(0.01)
        loss.backward()
        opt.step()



