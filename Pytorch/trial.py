# import numpy as np
# import matplotlib.pyplot as plt



# # data preparation
# train_images = np.load("./MNIST/npy-format-data/train_images.npy")/255
# train_labels = np.load("./MNIST/npy-format-data/train_labels.npy")
# test_images = np.load("./MNIST/npy-format-data/test_images.npy")/255
# test_labels = np.load("./MNIST/npy-format-data/test_labels.npy")


# idxs = np.where(train_labels==8)
# train_images = train_images[idxs]
# train_labels = train_labels[idxs]



# col_size = 5
# grid_imgs = np.vstack([ np.hstack([img for img in train_images[s:s+col_size]]) for s in range(0, col_size*5, col_size)])


# plt.imshow(grid_imgs, cmap='gray')
# plt.show()



import torch as t
from torch.autograd import Variable


true_xs = Variable(t.randn(100, 5))
noise = Variable(t.randn(100, 2))
print("true xs:\n", true_xs)
print("noise xs:\n", noise)






class Disc(t.nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.m = t.nn.Sequential(
            t.nn.Linear(5, 1),
            t.nn.BatchNorm1d(1),
            t.nn.Sigmoid()
        )

    def forward(self, input):
        output = self.m(input)
        return output





class Gen(t.nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.m = t.nn.Sequential(
            t.nn.Linear(2, 5)
        )

    def forward(self, input):
        output = self.m(input)
        return output




class Improved_Wloss(t.nn.Module):
    def __init__(self, G_model, D_model):
        super(Improved_Wloss, self).__init__()
        self.G = G_model
        self.D = D_model
        return

    def forward(self, _true_xs, _G_in_noise, _gamma):
        fake_xs = self.G(_G_in_noise)
        ori_w_loss = t.mean(self.D(_true_xs)) - t.mean(self.D(fake_xs))

        alpha = Variable(t.rand(fake_xs.size()))
        penalty_xs = alpha * _true_xs - (1 - alpha) * fake_xs

        import my_ext_api as api
        penalty_grad = api.calcGrad_wrt_varin(penalty_xs, self.D)
        penalty = _gamma * t.mean(t.pow(t.norm(penalty_grad) - 1, 2))

        w_loss = - (ori_w_loss - penalty)
        return w_loss





D = Disc()
G = Gen()

gamma = 0.001
print("gamma = ", gamma)


# optimizer
opt_D = t.optim.Adam(D.parameters(), lr=0.0002)


loss_log = []
for i in range(5000):
    fake_xs = G(noise)
    ori_w_loss = t.mean(D(true_xs)) - t.mean(D(fake_xs))
    alpha = Variable(t.rand(fake_xs.size()))
    penalty_xs = alpha * true_xs - (1 - alpha) * fake_xs
    penalty_out = D(penalty_xs)
    penalty_xs_grad = t.autograd.grad(penalty_out, penalty_xs, t.ones(penalty_out.size()), retain_graph=True)
    penalty = gamma * t.mean(t.pow(t.norm(penalty_xs_grad[0]) - 1, 2))

    opt_D.zero_grad()
    penalty.volatile = False
    loss = ori_w_loss - penalty
    loss.backward()
    opt_D.step()
    loss_log.append(loss.data[0])


import matplotlib.pyplot as plt
plt.plot(range(5000), loss_log)
plt.show()


# penalty_xs.retain_grad()
# penalty_out.backward(t.ones(penalty_out.size()), retain_graph=True)  # penalty_xs' grad is computed





# opt_D.zero_grad()
#
#
#
#
# penalty.volatile = False
# loss = ori_w_loss - penalty
#
# for param in D.parameters():
#     print("before:", param)
#
# loss.backward()
#
# for param in D.parameters():
#     print("later:", param)





# x = Variable(t.FloatTensor([1, 2, 3]), requires_grad=True)
# y = x*2
#
# a = Variable(t.FloatTensor(1))
# b = a**2 + 2
#
# z = b + y
#
# z.backward(t.ones(z.size()))
# print("b.grad", b.grad)


