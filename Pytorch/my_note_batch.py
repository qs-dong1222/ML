import numpy as np
import torch as t
from torch.autograd import Variable
import torch.utils.data as data

BATCH_SIZE = 10

############################
# given data
############################
x = np.linspace(-10, 10, 100).reshape(100, 1)
x = t.FloatTensor(x)
noise = t.rand(100, 1) * 0.3
y_true = 1e-6 * (-0.7*x + 3*x**2 - 11.7*x**5) + noise


dataSet = data.TensorDataset(data_tensor=x, target_tensor=y_true)
dataLoader = data.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # num_workers is number
                                                                                  # of processes we want to apply to do this batch job

for batch_Nbr, (batch_xs, batch_ys) in enumerate(dataLoader):
    print("batch_Nbr = ", batch_Nbr)
    print("batch_xs = ", batch_xs)
    print("batch_ys = ", batch_ys)


