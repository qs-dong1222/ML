import torch as t
import numpy as np
from my_ext_api import *


class Gen(t.nn.Module):
    def __init__(self, z_dim, y_dim):
        super(Gen, self).__init__()
        self.l1 = t.nn.Sequential(
            t.nn.Linear(z_dim+y_dim, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 784),
            t.nn.Sigmoid()
        )

    def forward(self, zy):
        output = self.l1(zy)
        return output



class Dis(t.nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Dis, self).__init__()
        self.l1 = t.nn.Sequential(
            t.nn.Linear(x_dim+y_dim, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, 1),
            t.nn.Sigmoid()
        )

    def forward(self, xy):
        output = self.l1(xy)
        return output


        
        
        


# G = Gen(100, 10)
# D = Dis(784, 10)
# init_model_weight(G, xavier_uniform)
# init_model_weight(D, xavier_uniform)
# print(G)
# print(D)
#
#
# z = Variable(t.rand(3, 100))
# y = Variable(t.rand(3, 10))
#
# fake = G(t.cat((z, y), dim=1))
# print(fake)
#
# pred = D(t.cat((fake, y), dim=1))
# print(pred)









# class Gen(t.nn.Module):
#     def __init__(self, z_dim, y_dim):
#         super(Gen, self).__init__()
#         self.z = t.nn.Sequential(
#             t.nn.Linear(z_dim, 200),
#             t.nn.ReLU()
#         )
#         self.y = t.nn.Sequential(
#             t.nn.Linear(y_dim, 1000),
#             t.nn.ReLU()
#         )
#         self.l1 = t.nn.Sequential(
#             t.nn.Linear(1200, 784),
#             t.nn.Sigmoid()
#         )
#         return
#
#     def forward(self, z_in, y_in):
#         z_out = self.z(z_in)
#         y_out = self.y(y_in)
#         zy = t.cat((z_out, y_out), dim=1)
#         # output = self.l1(zy).view(-1, 28, 28)
#         output = self.l1(zy)
#         return output
#
#
#
# class Dis(t.nn.Module):
#     def __init__(self, x_dim, y_dim):
#         super(Dis, self).__init__()
#         # self.x = my_ext_api.Maxout(x_dim, 240, 5)
#         # self.y = my_ext_api.Maxout(y_dim, 50, 5)
#         # self.xy = my_ext_api.Maxout(240 + 50, 240, 4)
#         super(Dis, self).__init__()
#         self.x = t.nn.Linear(x_dim, 240)
#         self.y = t.nn.Linear(y_dim, 50)
#         self.xy = t.nn.Linear(240 + 50, 240)
#         self.flat = t.nn.Linear(240, 1)
#
#     def forward(self, x_in, y_in):
#         x_out = self.x(x_in)
#         y_out = self.y(y_in)
#         xy_in = t.cat((x_out, y_out), dim=1)
#         xy_out = self.xy(xy_in)
#         output = self.flat(xy_out)
#         output = F.sigmoid(output)
#         return output