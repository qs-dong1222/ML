import torch as t
#import cupy as xp
import numpy as xp


batch_size = 50
in_size = 512
EPOCH = 5000


class Net(t.nn.Module):
    def __init__(self, in_size):
        super(Net, self).__init__()
        self.fc = t.nn.Sequential(
            t.nn.Linear(in_size, 1024),
            t.nn.LeakyReLU(),
            t.nn.BatchNorm1d(1024),
            t.nn.Linear(1024, 1024),
            t.nn.LeakyReLU(),
            t.nn.BatchNorm1d(1024),
            t.nn.Linear(1024, 1024),
            t.nn.ReLU(),
            t.nn.Linear(1024, 30)
        )

    def forward(self, x):
        return self.fc(x)



model = Net(in_size).cuda()

lossfunc = t.nn.CrossEntropyLoss().cuda()

opt = t.optim.Adam(model.parameters())


for epoch in range(EPOCH):
    x = xp.random.rand(batch_size, in_size)
    x = xp.sqrt(x ** 3 + 2 * x)
    x = t.Tensor(x).cuda()
    y_true = t.LongTensor(xp.random.randint(0, 30, batch_size)).cuda()
    pred = model(x)
    loss = lossfunc(pred, y_true)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if(epoch%50 == 0):
        print(loss)

