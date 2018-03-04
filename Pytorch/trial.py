import torch as t
from torch.autograd import Variable
from my_ext_api import *





# # hyper params
# EPOCH = 10
#
#
#
#
#
# class Gen(t.nn.Module):
#     def __init__(self, noise_dim, out_dim):
#         super(Gen, self).__init__()
#         self.l1 = t.nn.Sequential(
#             t.nn.Linear(noise_dim, 10),
#             t.nn.BatchNorm1d(10),
#             t.nn.ReLU(),
#             t.nn.Linear(10, out_dim),
#             t.nn.Tanh()
#         )
#         return
#
#     def forward(self, input):
#         output = self.l1(input)
#         return output
#
#
#
#
# class Dis(t.nn.Module):
#     def __init__(self, in_dim):
#         super(Dis, self).__init__()
#         self.l1 = t.nn.Sequential(
#             t.nn.Linear(in_dim, 10),
#             t.nn.BatchNorm1d(10),
#             t.nn.ReLU(),
#             t.nn.Linear(10, 1)
#         )
#
#     def forward(self, input):
#         output = self.l1(input)
#         return output
#
#
#
# # 100 -> G -> 20 -> D -> 1
# #                   ^
# #                   |
# #                   20
#
# G = Gen(noise_dim=100, out_dim=20).cuda()
# D = Dis(in_dim=20).cuda()
#
# print(G)
# print(D)
#
#
#
#
# noise = t.randn(5, 100)
# noise_v = Variable(noise).cuda()
# true_xs = t.randn(5, 20)
#
# fake_xs = G(noise_v)
# true_xs = Variable(true_xs).cuda()
#
# w_gp_loss = Wasserstein_GP_Loss().cuda()
# w_loss = w_gp_loss(D_model=D, fxs=fake_xs, rxs=true_xs, LAMBDA=10)
#
#
# print("w_loss:\n", w_loss)
#
#
# for params in D.parameters():
#     print("before:\n", params.grad)
#
#
# w_loss.backward()
#
#
#
# for params in D.parameters():
#     print("after:\n", params.grad)

